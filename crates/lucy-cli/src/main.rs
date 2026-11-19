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
use lucy_core::rules::{Params, Tissue};
use lucy_core::seed::{self, SeedConfig};
use serde_json::json;

mod sweep;
mod talk;

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
    /// Offline upper bound for class cells: pass 1 runs without classes, its final contexts are
    /// clustered in hindsight (Brown-style exchange), pass 2 reruns with that fixed map through
    /// the same back-off path (`cls_mode = 3`). Metrics are pass 2's.
    ClassBound {
        #[command(flatten)]
        a: RunArgs,
        #[arg(long, default_value = "runs/class-bound.jsonl")]
        metrics: PathBuf,
        #[arg(long, default_value_t = 4096)]
        classes: usize,
        #[arg(long, default_value_t = 5)]
        sweeps: usize,
    },
    /// Talk to a saved context tissue: each typed line is learned, then Lucy continues the
    /// text (her own bytes are context, never evidence). /sleep, /save, /quit; saves on exit.
    Talk {
        /// The tissue to wake (a `--save` of a context-tissue run).
        #[arg(long)]
        resume: PathBuf,
        /// Where to save (default: `<resume>-talk.bin`, never the womb itself).
        #[arg(long)]
        save: Option<PathBuf>,
        /// Sampling temperature (0 = always the top-1 byte).
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,
        /// Most bytes Lucy writes per turn (she also stops at a newline).
        #[arg(long, default_value_t = 200)]
        max_bytes: usize,
        /// Under each reply, the entropy of every byte in whole bits.
        #[arg(long)]
        entropy: bool,
        /// Sleep after this many idle seconds (0 = only on /sleep).
        #[arg(long, default_value_t = 120)]
        idle: u64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Each typed byte counts this many times a womb byte (default 1 = off).
        #[arg(long, default_value_t = 1.0)]
        kael_weight: f64,
        /// Recency: earlier evidence (womb and older talk) halves in weight every H talk bytes,
        /// relative to new talk bytes (0 = off).
        #[arg(long, default_value_t = 0)]
        recency: u64,
        /// Speaker tag for Kael's lines (framing).
        #[arg(long, default_value = "MOT")]
        you: String,
        /// Speaker tag that opens Lucy's turn (framing).
        #[arg(long, default_value = "CHI")]
        me: String,
        /// No framing: lines are learned as typed and Lucy picks her own speaker.
        #[arg(long)]
        raw: bool,
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
    /// Save the whole run state here at the end (before the final partial report) and every
    /// --save-every bytes; `--resume` continues it bit-identically.
    #[arg(long)]
    save: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    save_every: u64,
    /// Continue a saved run: graph flags, seed and --params come from the save; --ticks is the
    /// absolute end; metrics are appended.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Byte-stream tissue, overriding --params: `context` (context cells, default) or
    /// `reservoir` (the Phase 1 W/Q hidden cells and input-cell readout).
    #[arg(long, value_parser = ["context", "reservoir"])]
    tissue: Option<String>,
}

struct Setup {
    corpus: Vec<u8>,
    cfg: SeedConfig,
    p: Params,
    /// Fixed class map for `cls_mode = 3` (`class-bound` pass 2).
    oracle: Option<lucy_core::context::ClassMap>,
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
        if let Some(t) = &self.tissue {
            v.extend(["--tissue".to_string(), t.clone()]);
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
        ensure!(
            self.save_every == 0 || self.save.is_some(),
            "--save-every needs --save"
        );
        if self.threads > 1 {
            // The global pool can only be built once per process; later calls are no-ops.
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads)
                .build_global();
        }
        let mut p: Params = match &self.params {
            Some(path) => {
                let text = fs::read_to_string(path)
                    .with_context(|| format!("reading {}", path.display()))?;
                serde_json::from_str(&text)
                    .with_context(|| format!("parsing {}", path.display()))?
            }
            None => Params::default(),
        };
        if let Some(t) = &self.tissue {
            p.tissue = serde_json::from_value(serde_json::Value::String(t.clone()))?;
        }
        p.validate()
            .map_err(anyhow::Error::msg)
            .context("invalid params")?;
        let reservoir = p.tissue == Tissue::Reservoir;
        ensure!(
            reservoir || self.probe_every == 0,
            "the probe measures the reservoir: add --tissue reservoir"
        );
        // Context cells: the reservoir is off the byte path, so none is built.
        let n = |x: usize| if reservoir { x } else { 0 };
        let cfg = SeedConfig {
            seed: self.seed,
            hidden: n(self.cells),
            fanout: n(self.fanout),
            input_fanout: n(self.input_fanout),
            predict_fanin: n(self.predict_fanin),
            ticks_per_byte: self.ticks_per_byte,
            ring_cap: self.ring,
            ..SeedConfig::default()
        };
        cfg.validate()
            .map_err(anyhow::Error::msg)
            .context("invalid graph config")?;
        Ok(Setup {
            corpus,
            cfg,
            p,
            oracle: None,
        })
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
    let unigram = unigram_entropy_bits(&s.corpus);
    let (mut eng, mut obs) = match &a.resume {
        Some(path) => {
            let (mut eng, mut obs) = lucy_core::save::load(path)
                .with_context(|| format!("loading {}", path.display()))?;
            obs.set_out(out);
            let seen = (eng.stats.bytes_total as usize).min(s.corpus.len());
            eng.stats.baselines.rebuild(&s.corpus[..seen]);
            (eng, obs)
        }
        None => {
            let mut eng = seed::build(&s.cfg, s.p.clone());
            if extra_cells > 0 {
                seed::add_dormant(&mut eng, &s.cfg, extra_cells);
            }
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
            (eng, obs)
        }
    };
    eng.parallel = a.threads > 1;
    if let Some(m) = &s.oracle {
        eng.ctx.set_oracle(m.clone());
    }
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
        let k = eng.senses.ticks_per_byte;
        match &mut obs.prober {
            Some(p) => p.reattach(cfg, s.corpus.clone()),
            None => obs.prober = Some(Prober::new(cfg, s.corpus.clone(), k)),
        }
    }
    let t0 = Instant::now();
    // Bytes already consumed by a resumed run are skipped (one byte per arrival).
    let mut src = s
        .corpus
        .iter()
        .copied()
        .skip(eng.stats.bytes_total as usize);
    let step = a.save_every * eng.senses.ticks_per_byte;
    // First save boundary after the next pending event, then every `step` ticks.
    let mut mark = match step {
        0 => a.ticks,
        _ => (eng.next_time().unwrap_or(eng.now) / step + 1) * step,
    };
    loop {
        let until = mark.min(a.ticks);
        mark += step;
        eng.run_until(&mut src, until, &mut obs)?;
        if let Some(path) = &a.save {
            lucy_core::save::save(path, &eng, &obs)
                .with_context(|| format!("saving {}", path.display()))?;
        }
        if until >= a.ticks || eng.next_time().is_none() {
            break;
        }
    }
    obs.finish(&mut eng)?;
    Ok(Outcome {
        eng,
        obs,
        secs: t0.elapsed().as_secs_f64(),
        unigram,
    })
}

/// The metrics file: appended to when resuming (the first run wrote the header).
fn metrics_out(path: &Path, a: &RunArgs) -> Result<Box<dyn Write>> {
    if a.resume.is_none() {
        return Ok(Box::new(create(path)?));
    }
    let f = fs::OpenOptions::new()
        .append(true)
        .open(path)
        .with_context(|| format!("appending to {}", path.display()))?;
    Ok(Box::new(BufWriter::new(f)))
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
        "ticks {}  bytes {}  events {}  fires {}  cells {}  context cells {}",
        o.eng.now,
        st.bytes_total,
        st.events,
        fires,
        o.eng.g.cells.slots.live(),
        o.eng.ctx.cells()
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
            let o = simulate(&a, &s, 0, Some(metrics_out(&metrics, &a)?), None)?;
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
            let w = metrics_out(&metrics, &a)?;
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
        Cmd::ClassBound {
            a,
            metrics,
            classes,
            sweeps,
        } => {
            let mut s = a.setup()?;
            ensure!(
                s.p.tissue == Tissue::Context,
                "class-bound needs the context tissue"
            );
            let p = s.p.clone();
            s.p.cls_mode = 0;
            let o = simulate(&a, &s, 0, None, None)?;
            let t0 = Instant::now();
            let map = o.eng.ctx.batch_classes(&p, classes, sweeps);
            let used: std::collections::BTreeSet<u32> = map.values().copied().collect();
            println!(
                "pass 1: {} contexts -> {} classes in {:.1}s",
                map.len(),
                used.len(),
                t0.elapsed().as_secs_f64()
            );
            s.p = Params { cls_mode: 3, ..p };
            s.oracle = Some(map);
            let o = simulate(&a, &s, 0, Some(metrics_out(&metrics, &a)?), None)?;
            summary(&o);
            println!("pass 2 metrics -> {}", metrics.display());
        }
        Cmd::Talk {
            resume,
            save,
            temperature,
            max_bytes,
            entropy,
            idle,
            seed,
            kael_weight,
            recency,
            you,
            me,
            raw,
        } => talk::Talk {
            resume,
            save,
            temperature,
            max_bytes,
            entropy,
            idle_secs: idle,
            seed,
            kael_weight,
            recency,
            you,
            me,
            raw,
        }
        .run()?,
        Cmd::BenchScaling { a, multipliers } => {
            let s = a.setup()?;
            ensure!(
                s.p.tissue == Tissue::Reservoir,
                "bench-scaling adds dormant reservoir cells: add --tissue reservoir"
            );
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
