//! Probe ceiling (observatory only, never feeds back): how much next-byte information a linear
//! reader could extract from which hidden cells fired in the K ticks before each byte.
//!
//! Same protocol as `scripts/probe.py` (which stays as the offline reference):
//!   - one sample per byte: the distinct hidden cells fired in `[(i-1)K, iK)`, label = byte i;
//!   - temporal split, first 80 % train / last 20 % test (rows from index 2, like the counts);
//!   - softmax regression on L2-normalized binary rows, minibatch SGD (batch 256, 6 epochs,
//!     lr 0.2 / (1 + epoch)), weight gradient summed over the batch, bias gradient averaged;
//!   - a control probe on a one-hot of the previous byte (must land near the bigram), and count
//!     baselines with back-off fitted on train only.
//!
//! The `Prober` records one window per byte as the run goes (O(fires per byte), from the ring)
//! and trains at every `every`-byte checkpoint on the last `samples` windows, so the ceiling is a
//! curve over the run.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io;
use std::path::PathBuf;

use rand::seq::SliceRandom;
use serde::Serialize;

use crate::kernel::Engine;
use crate::seed::{STREAM_PROBE, stream};
use crate::store::Kind;

const CLASSES: usize = 256;
pub const EPOCHS: usize = 6;
pub const LR: f32 = 0.2;
pub const BATCH: usize = 256;

/// Sparse binary rows (CSR) with byte labels.
#[derive(Debug, Clone, Default)]
pub struct Sparse {
    pub indptr: Vec<u32>,
    pub indices: Vec<u32>,
    pub labels: Vec<u8>,
    pub n_feat: usize,
}

impl Sparse {
    fn row(&self, i: usize) -> &[u32] {
        &self.indices[self.indptr[i] as usize..self.indptr[i + 1] as usize]
    }

    /// Write the `probe_dump` file format (`scripts/probe.py PREFIX` re-runs it offline).
    pub fn write(&self, prefix: &str, meta: &serde_json::Value) -> io::Result<()> {
        let le = |xs: &[u32]| xs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut idx16 = Vec::with_capacity(self.indices.len() * 2);
        for &x in &self.indices {
            let x = u16::try_from(x).map_err(|_| io::Error::other("cell index over u16"))?;
            idx16.extend(x.to_le_bytes());
        }
        fs::write(format!("{prefix}.indptr"), le(&self.indptr))?;
        fs::write(format!("{prefix}.indices"), idx16)?;
        fs::write(format!("{prefix}.labels"), &self.labels)?;
        fs::write(format!("{prefix}.meta.json"), meta.to_string())
    }
}

/// Logits `b + sum_j W[j] / sqrt(nnz)` for one row.
fn logits(w: &[f32], b: &[f32], feats: &[u32], z: &mut [f32; CLASSES]) {
    z.copy_from_slice(b);
    if feats.is_empty() {
        return;
    }
    let val = 1.0 / (feats.len() as f32).sqrt();
    for &j in feats {
        let row = &w[j as usize * CLASSES..][..CLASSES];
        for (z, &w) in z.iter_mut().zip(row) {
            *z += w * val;
        }
    }
}

fn softmax(z: &mut [f32; CLASSES]) {
    let mx = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for x in z.iter_mut() {
        *x = (*x - mx).exp();
        s += *x;
    }
    for x in z.iter_mut() {
        *x /= s;
    }
}

fn argmax(z: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in z.iter().enumerate() {
        if x > z[best] {
            best = i;
        }
    }
    best
}

/// (top-1, bits per byte) on `rows`.
fn evaluate(w: &[f32], b: &[f32], d: &Sparse, rows: &[u32]) -> (f64, f64) {
    let (mut correct, mut bits) = (0usize, 0.0f64);
    let mut z = [0.0; CLASSES];
    for &r in rows {
        let y = d.labels[r as usize] as usize;
        logits(w, b, d.row(r as usize), &mut z);
        softmax(&mut z);
        correct += usize::from(argmax(&z) == y);
        bits -= (z[y].max(1e-12) as f64).log2();
    }
    let n = rows.len().max(1) as f64;
    (correct as f64 / n, bits / n)
}

/// Softmax regression by minibatch SGD; returns test (top-1, bits) after each epoch.
/// `mask` (n_feat x 256, 0/1): weight (j, b) may only be nonzero where mask = 1 (W starts at 0
/// and masked steps are dropped), i.e. byte b's logit only reads the allowed features.
pub fn train(
    d: &Sparse,
    train: &[u32],
    test: &[u32],
    seed: u64,
    mask: Option<&[f32]>,
) -> Vec<(f64, f64)> {
    let mut w = vec![0.0f32; d.n_feat * CLASSES];
    let mut b = [0.0f32; CLASSES];
    let mut rng = stream(seed, STREAM_PROBE);
    let mut perm = train.to_vec();
    let mut g = vec![[0.0f32; CLASSES]; BATCH];
    let mut history = Vec::with_capacity(EPOCHS);
    for ep in 0..EPOCHS {
        let step = LR / (1 + ep) as f32;
        perm.shuffle(&mut rng);
        for rows in perm.chunks(BATCH) {
            // All logits with the weights before the step (minibatch), then apply the step.
            for (gi, &r) in g.iter_mut().zip(rows) {
                logits(&w, &b, d.row(r as usize), gi);
                softmax(gi);
                gi[d.labels[r as usize] as usize] -= 1.0;
            }
            for (gi, &r) in g.iter().zip(rows) {
                let feats = d.row(r as usize);
                if feats.is_empty() {
                    continue;
                }
                let s = step / (feats.len() as f32).sqrt();
                for &j in feats {
                    let row = &mut w[j as usize * CLASSES..][..CLASSES];
                    match mask {
                        None => {
                            for (w, &g) in row.iter_mut().zip(gi) {
                                *w -= s * g;
                            }
                        }
                        Some(m) => {
                            let m = &m[j as usize * CLASSES..][..CLASSES];
                            for ((w, &g), &m) in row.iter_mut().zip(gi).zip(m) {
                                *w -= s * g * m;
                            }
                        }
                    }
                }
            }
            let s = step / rows.len() as f32;
            for gi in &g[..rows.len()] {
                for (b, &g) in b.iter_mut().zip(gi) {
                    *b -= s * g;
                }
            }
        }
        history.push(evaluate(&w, &b, d, test));
    }
    history
}

/// Order-0/1/2 count models fitted on `labels[2..cut]`, scored on `labels[cut..]` with back-off
/// to shorter contexts when unseen; bits use `(n_y + 1/2) / (n + 128)`.
#[allow(clippy::needless_range_loop)] // i indexes both the label and its context
pub fn count_baselines(labels: &[u8], cut: usize) -> [(f64, f64); 3] {
    let ctx = |i: usize, k: usize| -> u32 {
        labels[i - k..i]
            .iter()
            .fold(k as u32, |a, &x| (a << 8) | x as u32)
    };
    let mut tables: HashMap<u32, Vec<u32>> = HashMap::new();
    for i in 2..cut {
        for k in 0..3 {
            tables.entry(ctx(i, k)).or_insert_with(|| vec![0; CLASSES])[labels[i] as usize] += 1;
        }
    }
    let mut out = [(0.0, 0.0); 3];
    let n = (labels.len() - cut).max(1) as f64;
    for (k, o) in out.iter_mut().enumerate() {
        let (mut correct, mut bits) = (0usize, 0.0f64);
        for i in cut..labels.len() {
            let row = (0..=k)
                .rev()
                .find_map(|kk| tables.get(&ctx(i, kk)))
                .expect("unigram context exists");
            let y = labels[i] as usize;
            let total: u32 = row.iter().sum();
            let best = (0..CLASSES).fold(0, |a, c| if row[c] > row[a] { c } else { a });
            correct += usize::from(best == y);
            bits -= ((row[y] as f64 + 0.5) / (total as f64 + 0.5 * CLASSES as f64)).log2();
        }
        *o = (correct as f64 / n, bits / n);
    }
    out
}

/// One probe evaluation (a checkpoint).
#[derive(Serialize, Debug, Clone)]
pub struct ProbeReport {
    pub kind: &'static str,
    pub bytes_seen: u64,
    pub tick: u64,
    pub samples: usize,
    pub active_cells: f64,
    /// Routing: Lucy's own readout top-1 on the probe's test bytes (same bytes, same split).
    pub readout_accuracy: f64,
    /// Probe ceiling: test top-1 and bits/byte of the reservoir probe (last epoch).
    pub probe_accuracy: f64,
    pub probe_bits: f64,
    /// Edge-restricted probe: byte b's logit may only read hidden cells that have an edge into
    /// input cell b at the checkpoint (the ceiling of the current topology).
    pub edge_probe_accuracy: Option<f64>,
    pub edge_probe_bits: Option<f64>,
    pub control_accuracy: f64,
    pub control_bits: f64,
    /// Count baselines on the same split: [unigram, bigram, trigram].
    pub baseline_accuracy: [f64; 3],
    pub baseline_bits: [f64; 3],
    pub probe_history: Vec<(f64, f64)>,
    /// Forgetting measure: a clone of the engine with consolidation off (W frozen) re-reads the
    /// first `old` corpus bytes; full probe and Lucy's readout on that fixed early segment.
    pub old_probe_accuracy: Option<f64>,
    pub old_probe_bits: Option<f64>,
    pub old_readout_accuracy: Option<f64>,
    pub train_secs: f64,
}

/// Full protocol on one segment.
/// `cum[j]`: Lucy's correct guesses before row j's byte (scores routing on the test rows).
pub fn evaluate_segment(
    d: &Sparse,
    cum: &[u64],
    edge_mask: Option<&[f32]>,
    bytes_seen: u64,
    tick: u64,
) -> ProbeReport {
    let t0 = std::time::Instant::now();
    let n = d.labels.len();
    let cut = n * 4 / 5;
    let train_rows: Vec<u32> = (2..cut as u32).collect();
    let test_rows: Vec<u32> = (cut as u32..n as u32).collect();
    let base = count_baselines(&d.labels, cut);
    let control = Sparse {
        indptr: (0..=n as u32).collect(),
        indices: std::iter::once(0)
            .chain(d.labels[..n - 1].iter().map(|&x| x as u32))
            .collect(),
        labels: d.labels.clone(),
        n_feat: CLASSES,
    };
    let c = train(&control, &train_rows, &test_rows, 0, None);
    let h = train(d, &train_rows, &test_rows, 0, None);
    let em = edge_mask.map(|m| train(d, &train_rows, &test_rows, 0, Some(m))[EPOCHS - 1]);
    ProbeReport {
        kind: "probe",
        bytes_seen,
        tick,
        samples: n,
        active_cells: d.indices.len() as f64 / n.max(1) as f64,
        readout_accuracy: match cum {
            [] => f64::NAN,
            _ => (cum[n - 1] - cum[cut]) as f64 / (n - 1 - cut) as f64,
        },
        probe_accuracy: h[EPOCHS - 1].0,
        probe_bits: h[EPOCHS - 1].1,
        edge_probe_accuracy: em.map(|x| x.0),
        edge_probe_bits: em.map(|x| x.1),
        control_accuracy: c[EPOCHS - 1].0,
        control_bits: c[EPOCHS - 1].1,
        baseline_accuracy: base.map(|x| x.0),
        baseline_bits: base.map(|x| x.1),
        probe_history: h,
        old_probe_accuracy: None,
        old_probe_bits: None,
        old_readout_accuracy: None,
        train_secs: t0.elapsed().as_secs_f64(),
    }
}

#[derive(Debug, Clone)]
pub struct ProbeConfig {
    /// Checkpoint every this many bytes.
    pub every: u64,
    /// Windows per probe (the most recent ones).
    pub samples: usize,
    /// Where checkpoint artifacts go: raw windows (`probe-<bytes>.*`) and cell snapshots.
    pub dir: Option<PathBuf>,
    pub dump: bool,
    pub snapshots: bool,
    /// Fixed early segment for the forgetting measure (empty = off).
    pub old: Vec<u8>,
}

/// Collect one window per byte while a clone of `eng` (consolidation off) reads `old`, then
/// probe it: (probe top-1, probe bits, Lucy readout top-1 on the test bytes).
pub fn old_segment_probe(eng: &Engine, old: &[u8]) -> Option<(f64, f64, f64)> {
    let mut e = eng.clone();
    e.p.consolidate_every = 0;
    e.parallel = false;
    let k = e.senses.ticks_per_byte;
    let mut next = e
        .scheduler()
        .sorted()
        .into_iter()
        .find(|ev| matches!(ev.payload, crate::scheduler::Payload::Arrive))?
        .t;
    let mut src = old.iter().copied();
    let mut d = Sparse {
        indptr: vec![0],
        n_feat: e.g.cells.slots.capacity(),
        ..Sparse::default()
    };
    let mut cum = Vec::with_capacity(old.len());
    let mut buf = Vec::new();
    for &label in old {
        while e.next_time().is_some_and(|t| t < next) {
            e.step(&mut src);
        }
        buf.clear();
        let mut seq = e.ring.total();
        while seq > 0 {
            seq -= 1;
            let Some(ev) = e.ring.get(seq) else { break };
            if ev.t + k < next {
                break;
            }
            if e.g.cells.kind[ev.cell.idx as usize] == Kind::Hidden {
                buf.push(ev.cell.idx);
            }
        }
        buf.sort_unstable();
        buf.dedup();
        d.indices.extend_from_slice(&buf);
        d.indptr.push(d.indices.len() as u32);
        d.labels.push(label);
        cum.push(e.stats.correct_total);
        next += k;
    }
    let r = evaluate_segment(&d, &cum, None, 0, 0);
    Some((r.probe_accuracy, r.probe_bits, r.readout_accuracy))
}

/// Records windows during a run and probes at checkpoints. Reads the engine only.
pub struct Prober {
    cfg: ProbeConfig,
    corpus: Vec<u8>,
    k: u64,
    next: u64,
    rows: VecDeque<Box<[u32]>>,
    labels: VecDeque<u8>,
    cum: VecDeque<u64>,
    since_last: u64,
    buf: Vec<u32>,
}

impl Prober {
    pub fn new(cfg: ProbeConfig, corpus: Vec<u8>, ticks_per_byte: u64) -> Self {
        assert!(cfg.every > 0 && cfg.samples >= 100, "probe: bad config");
        Self {
            cfg,
            corpus,
            k: ticks_per_byte,
            next: 1,
            rows: VecDeque::new(),
            labels: VecDeque::new(),
            cum: VecDeque::new(),
            since_last: 0,
            buf: Vec::new(),
        }
    }

    /// Called before the engine processes events at `t`: record the window of every byte that
    /// arrives at or before `t`; returns the reports of checkpoints completed on the way.
    pub fn observe_until(&mut self, eng: &Engine, t: u64) -> io::Result<Vec<ProbeReport>> {
        let mut out = Vec::new();
        while (self.next as usize) < self.corpus.len() && self.next * self.k <= t {
            self.record(eng, self.next);
            self.next += 1;
            if self.next.is_multiple_of(self.cfg.every) {
                out.push(self.checkpoint(eng, t)?);
            }
        }
        Ok(out)
    }

    /// Probe whatever was recorded since the last checkpoint (end of run).
    pub fn finish(&mut self, eng: &Engine) -> io::Result<Option<ProbeReport>> {
        if self.since_last < 1000 {
            return Ok(None);
        }
        self.checkpoint(eng, eng.now).map(Some)
    }

    fn record(&mut self, eng: &Engine, i: u64) {
        let from = i * self.k - self.k;
        self.buf.clear();
        let mut seq = eng.ring.total();
        while seq > 0 {
            seq -= 1;
            let Some(ev) = eng.ring.get(seq) else { break };
            if ev.t < from {
                break;
            }
            if eng.g.cells.kind[ev.cell.idx as usize] == Kind::Hidden {
                self.buf.push(ev.cell.idx);
            }
        }
        self.buf.sort_unstable();
        self.buf.dedup();
        if self.rows.len() == self.cfg.samples {
            self.rows.pop_front();
            self.labels.pop_front();
            self.cum.pop_front();
        }
        self.rows.push_back(self.buf.as_slice().into());
        self.labels.push_back(self.corpus[i as usize]);
        self.cum.push_back(eng.stats.correct_total);
        self.since_last += 1;
    }

    fn checkpoint(&mut self, eng: &Engine, t: u64) -> io::Result<ProbeReport> {
        self.since_last = 0;
        let mut d = Sparse {
            indptr: vec![0],
            n_feat: eng.g.cells.slots.capacity(),
            labels: self.labels.iter().copied().collect(),
            ..Sparse::default()
        };
        for r in &self.rows {
            d.indices.extend_from_slice(r);
            d.indptr.push(d.indices.len() as u32);
        }
        let bytes = self.next;
        if let Some(dir) = &self.cfg.dir {
            fs::create_dir_all(dir)?;
            if self.cfg.dump {
                let prefix = dir.join(format!("probe-{bytes}")).display().to_string();
                d.write(
                    &prefix,
                    &serde_json::json!({
                        "start_byte": bytes - d.labels.len() as u64, "n": d.labels.len(),
                        "ticks_per_byte": self.k, "cells": d.n_feat, "first_hidden": 0,
                    }),
                )?;
            }
            if self.cfg.snapshots {
                let path = dir.join(format!("snapshot-{bytes}.jsonl"));
                crate::observe::write_snapshot(
                    eng,
                    &mut io::BufWriter::new(fs::File::create(path)?),
                )?;
            }
        }
        let old = (!self.cfg.old.is_empty())
            .then(|| old_segment_probe(eng, &self.cfg.old))
            .flatten();
        let cum: Vec<u64> = self.cum.iter().copied().collect();
        let mut mask = vec![0.0f32; d.n_feat * CLASSES];
        for b in 0..crate::senses::N_BYTES {
            for &e in &eng.g.cells.ins[b] {
                mask[eng.g.edges.src[e as usize] as usize * CLASSES + b] = 1.0;
            }
        }
        let mut r = evaluate_segment(&d, &cum, Some(&mask), bytes, t);
        if let Some((a, b, ro)) = old {
            (
                r.old_probe_accuracy,
                r.old_probe_bits,
                r.old_readout_accuracy,
            ) = (Some(a), Some(b), Some(ro));
        }
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Label = (feature cell) mod 7: a linear probe must read it; counts on random labels can't.
    #[test]
    fn probe_reads_a_linear_code_and_counts_match_hand_values() {
        let mut d = Sparse {
            indptr: vec![0],
            n_feat: 50,
            ..Sparse::default()
        };
        let mut rng = stream(1, 99);
        for _ in 0..3000 {
            let f = rng.random_range(0..49u32);
            d.indices.extend([f, 49]); // a constant feature too: normalization must cope
            d.indptr.push(d.indices.len() as u32);
            d.labels.push((f % 7) as u8);
        }
        // Edge mask: class k may read feature j only if j % 7 == k: still perfect. Forbidding
        // classes 0 and 1 their features leaves them indistinguishable (the bias can make
        // "no evidence" mean one of them, not both).
        let mut ok = vec![0.0; 50 * CLASSES];
        let mut no0 = vec![0.0; 50 * CLASSES];
        for j in 0..49 {
            ok[j * CLASSES + j % 7] = 1.0;
            if j % 7 > 1 {
                no0[j * CLASSES + j % 7] = 1.0;
            }
        }
        let r = evaluate_segment(&d, &[], Some(&ok), 0, 0);
        assert!(r.probe_accuracy > 0.99, "{r:?}");
        assert!(r.edge_probe_accuracy.unwrap() > 0.99, "{r:?}");
        let r0 = evaluate_segment(&d, &[], Some(&no0), 0, 0);
        let frac = |k| d.labels[2400..].iter().filter(|&&y| y == k).count() as f64 / 600.0;
        let acc0 = r0.edge_probe_accuracy.unwrap();
        assert!(
            acc0 < 1.0 - frac(0).min(frac(1)) + 0.02 && acc0 > 0.8,
            "{r0:?}"
        );
        assert!(r.baseline_accuracy[0] < 0.25);

        // "abab...": bigram and trigram are perfect on the test part, unigram is a coin flip.
        let labels: Vec<u8> = b"ab".repeat(50).to_vec();
        let c = count_baselines(&labels, 80);
        assert_eq!((c[1].0, c[2].0), (1.0, 1.0));
        assert!((c[0].0 - 0.5).abs() < 1e-9);
    }
}
