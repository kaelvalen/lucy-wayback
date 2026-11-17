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
use lucy_core::rules::Params;
use lucy_core::seed::{self, SeedConfig};
use serde_json::json;

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
    /// JSON file overriding any subset of rule constants (see docs/DESIGN.md).
    #[arg(long)]
    params: Option<PathBuf>,
}

struct Setup {
    corpus: Vec<u8>,
    cfg: SeedConfig,
    p: Params,
}

impl RunArgs {
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
) -> Result<Outcome> {
    let mut eng = seed::build(&s.cfg, s.p.clone());
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
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Run {
            a,
            metrics,
            snapshot,
        } => {
            let s = a.setup()?;
            let o = simulate(&a, &s, 0, Some(Box::new(create(&metrics)?)))?;
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
        Cmd::CheckDeterminism { a } => {
            let s = a.setup()?;
            let mut hashes = Vec::new();
            for run in 1..=2 {
                let o = simulate(&a, &s, 0, None)?;
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
            let o = simulate(&a, &s, 0, None)?;
            summary(&o);
            write_snapshot(&o.eng, &mut create(&out)?)?;
            println!("snapshot -> {}", out.display());
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
                let o = simulate(&a, &s, (k - 1) * a.cells, None)?;
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
