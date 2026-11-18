//! `lucy` command line: run, check-determinism, snapshot, bench-scaling.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Parser, Subcommand};
use lucy_core::kernel::Engine;
use lucy_core::observe::{
    Observatory, state_hash, unigram_entropy_bits, unigram_top1_accuracy, write_snapshot,
};
use lucy_core::probe::{ProbeConfig, Prober};
use lucy_core::rules::Params;
use lucy_core::seed::{self, SeedConfig};
use serde_json::json;

mod sweep;

#[derive(Parser)]
#[command(
    name = "lucy",
    version,
    about = "Lucy: sparse, event-driven, CPU-only learning substrate"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Stream a corpus through Lucy, writing metrics JSONL.
    Run {
        #[command(flatten)]
        a: RunArgs,
        #[arg(long, default_value = "runs/metrics.jsonl")]
        metrics: PathBuf,
        /// Also dump per-cell stats at the end.
        #[arg(long)]
        snapshot: Option<PathBuf>,
    },
    /// Run with the probe ceiling measured every --probe-every bytes (default 100000): metrics
    /// and probe rows go to --metrics; --dump / --snapshots add per-checkpoint files beside it.
    Probe {
        #[command(flatten)]
        a: RunArgs,
        #[arg(long, default_value = "runs/probe/metrics.jsonl")]
        metrics: PathBuf,
        /// Write each checkpoint's raw probe windows (`probe-<bytes>.*`, scripts/probe.py format).
        #[arg(long)]
        dump: bool,
        /// Write a per-cell snapshot at each checkpoint.
        #[arg(long)]
        snapshots: bool,
    },
    /// Run twice and compare hashes of all deterministic metrics + final state.
    CheckDeterminism {
        #[command(flatten)]
        a: RunArgs,
    },
    /// Run, then dump per-cell stats (JSONL) for offline analysis.
    Snapshot {
        #[command(flatten)]
        a: RunArgs,
        #[arg(long, default_value = "runs/snapshot.jsonl")]
        out: PathBuf,
    },
    /// Run seeds x parameter grids as parallel `lucy run` processes and print a summary table.
    /// Shares all run flags (corpus, ticks, graph size, base --params); --seed is ignored.
    Sweep {
        #[command(flatten)]
        a: RunArgs,
        #[arg(long, value_delimiter = ',', default_value = "0")]
        seeds: Vec<u64>,
        /// Parameter axis `name=v1,v2,...` (repeatable; the grid is their product).
        #[arg(long = "param")]
        axes: Vec<String>,
        /// Concurrent runs (default: available cores).
        #[arg(long)]
        jobs: Option<usize>,
        #[arg(long, default_value = "runs/sweep")]
        out: PathBuf,
    },
    /// Same activity, more total cells: cost must stay ~constant.
    BenchScaling {
        #[command(flatten)]
        a: RunArgs,
        #[arg(long, value_delimiter = ',', default_value = "1,2,4,8")]
        multipliers: Vec<usize>,
    },
}

#[derive(Args, Clone)]
struct RunArgs {
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Logical ticks to run (stops early if the corpus ends and activity dies out).
    #[arg(long)]
    ticks: u64,
    /// Hidden cells.
    #[arg(long, default_value_t = SeedConfig::default().hidden)]
    cells: usize,
    #[arg(long, default_value_t = SeedConfig::default().fanout)]
    fanout: usize,
    #[arg(long, default_value_t = SeedConfig::default().input_fanout)]
    input_fanout: usize,
    #[arg(long, default_value_t = SeedConfig::default().predict_fanin)]
    predict_fanin: usize,
    #[arg(long, default_value_t = SeedConfig::default().ticks_per_byte)]
    ticks_per_byte: u64,
    #[arg(long, default_value_t = SeedConfig::default().ring_cap)]
    ring: usize,
    #[arg(long, default_value_t = 10_000)]
    metrics_every: u64,
    /// Worker threads for the tick-synchronous parallel fire phase (results are bit-identical).
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// JSON file overriding any subset of rule constants (see docs/DESIGN.md).
    #[arg(long)]
    params: Option<PathBuf>,
    /// Probe ceiling every this many bytes (0 = off; `lucy probe` defaults it to 100000).
    #[arg(long, default_value_t = 0)]
    probe_every: u64,
    /// Most recent byte windows each probe trains/tests on (80/20 temporal split).
    #[arg(long, default_value_t = 200_000)]
    probe_samples: usize,
    /// Forgetting measure at each checkpoint: re-read the first N corpus bytes through a frozen
    /// clone and probe them (0 = off).
    #[arg(long, default_value_t = 0)]
    probe_old: usize,
    /// Add time-resolved and bigram-path probes and the path audit to each checkpoint.
    #[arg(long)]
    probe_paths: bool,
}

struct Setup {
    corpus: Vec<u8>,
    cfg: SeedConfig,
    p: Params,
}

impl RunArgs {
    /// The flags a child `lucy run` needs to reproduce this configuration (minus seed/params).
    fn forwarded(&self) -> Vec<String> {
        let mut v = vec!["--corpus".to_string(), self.corpus.display().to_string()];
        for (flag, val) in [
            ("--ticks", self.ticks.to_string()),
            ("--cells", self.cells.to_string()),
            ("--fanout", self.fanout.to_string()),
            ("--input-fanout", self.input_fanout.to_string()),
            ("--predict-fanin", self.predict_fanin.to_string()),
            ("--ticks-per-byte", self.ticks_per_byte.to_string()),
            ("--ring", self.ring.to_string()),
            ("--metrics-every", self.metrics_every.to_string()),
            ("--probe-every", self.probe_every.to_string()),
            ("--probe-samples", self.probe_samples.to_string()),
            ("--probe-old", self.probe_old.to_string()),
        ] {
            v.push(flag.to_string());
            v.push(val);
        }
        if self.probe_paths {
            v.push("--probe-paths".to_string());
        }
        v
    }

    fn setup(&self) -> Result<Setup> {
        // shortcut: whole-file read (fine up to ~1 GB); switch to BufReader streaming for larger corpora.
        let corpus = fs::read(&self.corpus)
            .with_context(|| format!("reading corpus {}", self.corpus.display()))?;
        ensure!(
            !corpus.is_empty(),
            "corpus {} is empty",
            self.corpus.display()
        );
        ensure!(self.metrics_every > 0, "--metrics-every must be > 0");
        ensure!(self.threads >= 1, "--threads must be >= 1");
        ensure!(
            self.probe_samples >= 1000,
            "--probe-samples must be >= 1000"
        );
        if self.threads > 1 {
            // The global pool can only be built once per process; later calls are no-ops.
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads)
                .build_global();
        }
        let p = match &self.params {
            Some(path) => {
                let text = fs::read_to_string(path)
                    .with_context(|| format!("reading {}", path.display()))?;
                serde_json::from_str(&text)
                    .with_context(|| format!("parsing {}", path.display()))?
            }
            None => Params::default(),
        };
        p.validate()
            .map_err(anyhow::Error::msg)
            .context("invalid params")?;
        let cfg = SeedConfig {
            seed: self.seed,
            hidden: self.cells,
            fanout: self.fanout,
            input_fanout: self.input_fanout,
            predict_fanin: self.predict_fanin,
            ticks_per_byte: self.ticks_per_byte,
            ring_cap: self.ring,
            ..SeedConfig::default()
        };
        cfg.validate()
            .map_err(anyhow::Error::msg)
            .context("invalid graph config")?;
        Ok(Setup { corpus, cfg, p })
    }
}

struct Outcome {
    eng: Engine,
    obs: Observatory,
    secs: f64,
    unigram: f64,
}

fn simulate(
    a: &RunArgs,
    s: &Setup,
    extra_cells: usize,
    out: Option<Box<dyn Write>>,
    artifacts: Option<(PathBuf, bool, bool)>,
) -> Result<Outcome> {
    let mut eng = seed::build(&s.cfg, s.p.clone());
    eng.parallel = a.threads > 1;
    if extra_cells > 0 {
        seed::add_dormant(&mut eng, &s.cfg, extra_cells);
    }
    let unigram = unigram_entropy_bits(&s.corpus);
    let mut obs = Observatory::new(&eng, a.metrics_every, s.cfg.seed, out);
    obs.header(&json!({
        "kind": "header",
        "corpus": a.corpus.display().to_string(),
        "corpus_bytes": s.corpus.len(),
        "unigram_entropy_bits": unigram,
        "unigram_top1_accuracy": unigram_top1_accuracy(&s.corpus),
        "ticks": a.ticks,
        "extra_cells": extra_cells,
        "seed_config": s.cfg,
        "params": s.p,
    }))?;
    if a.probe_every > 0 {
        let (dir, dump, snapshots) =
            artifacts.map_or((None, false, false), |(d, x, y)| (Some(d), x, y));
        let cfg = ProbeConfig {
            every: a.probe_every,
            samples: a.probe_samples,
            dir,
            dump,
            snapshots,
            old: s.corpus[..a.probe_old.min(s.corpus.len())].to_vec(),
            paths: a.probe_paths,
        };
        obs.prober = Some(Prober::new(cfg, s.corpus.clone(), s.cfg.ticks_per_byte));
    }
    let t0 = Instant::now();
    eng.run(&mut s.corpus.iter().copied(), a.ticks, &mut obs)?;
    Ok(Outcome {
        eng,
        obs,
        secs: t0.elapsed().as_secs_f64(),
        unigram,
    })
}

fn create(path: &Path) -> Result<BufWriter<File>> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    Ok(BufWriter::new(
        File::create(path).with_context(|| format!("creating {}", path.display()))?,
    ))
}

fn summary(o: &Outcome) {
    let st = &o.eng.stats;
    let fires = o.eng.ring.total();
    println!(
        "ticks {}  bytes {}  events {}  fires {}  cells {}",
        o.eng.now,
        st.bytes_total,
        st.events,
        fires,
        o.eng.g.cells.slots.live()
    );
    println!(
        "wall {:.2}s  events/sec {:.0}  fires/sec {:.0}",
        o.secs,
        st.events as f64 / o.secs,
        fires as f64 / o.secs
    );
    if let Some(m) = &o.obs.last {
        let f = |x: Option<f64>| x.map_or("-".to_string(), |v| format!("{v:.3}"));
        println!(
            "last window: surprise {} bits (unigram baseline {:.3})  accuracy {}  tension {}  hop depth {}  fires/tick {}",
            f(m.surprise_bits),
            o.unigram,
            f(m.accuracy),
            f(m.tension),
            f(m.hop_depth),
            f(m.fires_per_tick),
        );
    }
    if let Some(r) = o.obs.probes.last() {
        println!(
            "headline @ {} bytes: routing (readout) {:.4}  tissue (probe ceiling) {:.4}  | uni {:.4} bi {:.4} tri {:.4} (probe split)",
            r.bytes_seen,
            r.readout_accuracy,
            r.probe_accuracy,
            r.baseline_accuracy[0],
            r.baseline_accuracy[1],
            r.baseline_accuracy[2],
        );
    }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Run {
            a,
            metrics,
            snapshot,
        } => {
            let s = a.setup()?;
            let o = simulate(&a, &s, 0, Some(Box::new(create(&metrics)?)), None)?;
            summary(&o);
            println!(
                "state hash {:016x}  metrics -> {}",
                state_hash(&o.eng),
                metrics.display()
            );
            if let Some(path) = snapshot {
                write_snapshot(&o.eng, &mut create(&path)?)?;
                println!("snapshot -> {}", path.display());
            }
        }
        Cmd::Probe {
            mut a,
            metrics,
            dump,
            snapshots,
        } => {
            if a.probe_every == 0 {
                a.probe_every = 100_000;
            }
            let s = a.setup()?;
            let dir = metrics
                .parent()
                .filter(|d| !d.as_os_str().is_empty())
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
            let w = Box::new(create(&metrics)?);
            let o = simulate(&a, &s, 0, Some(w), Some((dir, dump, snapshots)))?;
            summary(&o);
            println!(
                "state hash {:016x}  metrics + probe -> {}",
                state_hash(&o.eng),
                metrics.display()
            );
        }
        Cmd::CheckDeterminism { a } => {
            let s = a.setup()?;
            // With --threads > 1 the first run is single-threaded: the check then also proves the
            // parallel engine is bit-identical to the sequential one.
            let mut hashes = Vec::new();
            for run in 1..=2 {
                let single = RunArgs {
                    threads: 1,
                    ..a.clone()
                };
                let o = simulate(if run == 1 { &single } else { &a }, &s, 0, None, None)?;
                let h = (o.obs.metrics_hash(), state_hash(&o.eng));
                println!(
                    "run {run}: metrics {:016x}  state {:016x}  ({} events, {:.2}s)",
                    h.0, h.1, o.eng.stats.events, o.secs
                );
                hashes.push(h);
            }
            if hashes[0] != hashes[1] {
                bail!("NONDETERMINISTIC: hashes differ");
            }
            println!("deterministic: OK");
        }
        Cmd::Snapshot { a, out } => {
            let s = a.setup()?;
            let o = simulate(&a, &s, 0, None, None)?;
            summary(&o);
            write_snapshot(&o.eng, &mut create(&out)?)?;
            println!("snapshot -> {}", out.display());
        }
        Cmd::Sweep {
            a,
            seeds,
            axes,
            jobs,
            out,
        } => {
            a.setup()?; // validate the shared configuration once, before spawning anything
            let jobs =
                jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
            sweep::Sweep {
                run_args: a.forwarded(),
                seeds: &seeds,
                axes: &axes,
                base_params: a.params.as_deref(),
                jobs,
                out: &out,
            }
            .run()?;
        }
        Cmd::BenchScaling { a, multipliers } => {
            let s = a.setup()?;
            println!(
                "{:>5} {:>8} {:>11} {:>10} {:>9} {:>9} {:>7}",
                "mult", "cells", "events", "fires", "ns/event", "ns/fire", "ratio"
            );
            let mut base: Option<(u64, f64)> = None;
            for &k in &multipliers {
                ensure!(k >= 1, "multipliers must be >= 1");
                let o = simulate(&a, &s, (k - 1) * a.cells, None, None)?;
                let (events, fires) = (o.eng.stats.events, o.eng.ring.total());
                let ns_event = o.secs * 1e9 / events as f64;
                let (base_events, base_ns) = *base.get_or_insert((events, ns_event));
                println!(
                    "{:>4}x {:>8} {:>11} {:>10} {:>9.1} {:>9.1} {:>6.2}x",
                    k,
                    o.eng.g.cells.slots.live(),
                    events,
                    fires,
                    ns_event,
                    o.secs * 1e9 / fires as f64,
                    ns_event / base_ns
                );
                ensure!(
                    events == base_events,
                    "activity changed with dormant cells: locality violated"
                );
            }
        }
    }
    Ok(())
}
