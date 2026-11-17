//! Observatory: windowed counters fed by the kernel (O(1) per event), periodic JSONL reports,
//! a fixed reservoir sample for population stats (never a pass over all cells), the
//! determinism hash, and offline snapshots. Observing only reads dynamics state.

use std::io::{self, Write};
use std::time::Instant;

use rand::Rng;
use serde::Serialize;
use serde_json::json;

use crate::kernel::Engine;
use crate::num::{S, norm};
use crate::rules;
use crate::scheduler::Payload;
use crate::seed::{STREAM_RESERVOIR, stream};
use crate::store::{Kind, Wave};

pub const RESERVOIR: usize = 1024;

/// Online fit of the softmax inverse temperature `beta = 1 / tau` that turns readout scores into
/// a calibrated next-byte distribution. Observatory only: the dynamics never read it.
///
/// Each byte is scored with the current beta first (prequential, so the reported surprise is
/// honest), then beta takes a damped stochastic Newton step on the log-loss:
/// `dL/dbeta = E_p[s] - s_true`, `d2L/dbeta2 = Var_p[s]` (running mean).
#[derive(Debug, Clone)]
pub struct Calibrator {
    pub beta: f64,
    curvature: f64,
}

/// Step and averaging rate of the calibrator: ~100 bytes of memory.
const CALIBRATION_RATE: f64 = 0.01;

impl Default for Calibrator {
    fn default() -> Self {
        Self {
            beta: 1.0,
            curvature: 0.0,
        }
    }
}

impl Calibrator {
    /// Surprise in bits of `truth` under `softmax(beta * scores)`, then update beta.
    pub fn observe(&mut self, scores: &[S], truth: usize) -> f64 {
        let logits: Vec<f64> = scores.iter().map(|&s| self.beta * s as f64).collect();
        let mx = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
        let z: f64 = w.iter().sum();
        let surprise = (mx + z.ln() - logits[truth]) / std::f64::consts::LN_2;
        let mean: f64 = w
            .iter()
            .zip(scores)
            .map(|(w, &s)| w * s as f64)
            .sum::<f64>()
            / z;
        let var: f64 = w
            .iter()
            .zip(scores)
            .map(|(w, &s)| w * (s as f64 - mean).powi(2))
            .sum::<f64>()
            / z;
        if var > 0.0 {
            self.curvature = if self.curvature == 0.0 {
                var
            } else {
                self.curvature + CALIBRATION_RATE * (var - self.curvature)
            };
            let grad = mean - scores[truth] as f64;
            self.beta = (self.beta - CALIBRATION_RATE * grad / self.curvature).max(0.0);
        }
        surprise
    }
}

/// Online, prequential count model over the byte stream with an `ORDER`-byte context
/// (0 = unigram, 1 = bigram, 2 = trigram): predicts, then learns. These are the real targets for
/// Lucy's next-byte accuracy and surprise, measured on exactly the same stream and windows.
/// Top-1 is a running argmax per context (first byte to reach the max wins ties); code length
/// uses the KT estimator `(n_ctx,b + 1/2) / (n_ctx + 128)`.
#[derive(Debug, Clone)]
pub struct CountModel<const ORDER: usize> {
    counts: Vec<u32>,
    totals: Vec<u32>,
    best: Vec<u8>,
    ctx: usize,
    seen: usize,
}

impl<const ORDER: usize> Default for CountModel<ORDER> {
    fn default() -> Self {
        let contexts = 1 << (8 * ORDER);
        Self {
            counts: vec![0; contexts * 256],
            totals: vec![0; contexts],
            best: vec![0; contexts],
            ctx: 0,
            seen: 0,
        }
    }
}

impl<const ORDER: usize> CountModel<ORDER> {
    /// Score `b` under the model (correct top-1?, bits), then count it.
    pub fn observe(&mut self, b: u8) -> (bool, f64) {
        let row = self.ctx * 256;
        let n = self.totals[self.ctx];
        let k = self.counts[row + b as usize];
        let correct = n > 0 && self.best[self.ctx] == b;
        let bits = -((k as f64 + 0.5) / (n as f64 + 128.0)).log2();
        self.counts[row + b as usize] = k + 1;
        self.totals[self.ctx] = n + 1;
        if k + 1 > self.counts[row + self.best[self.ctx] as usize] {
            self.best[self.ctx] = b;
        }
        if ORDER > 0 {
            self.ctx = ((self.ctx << 8) | b as usize) & ((1 << (8 * ORDER)) - 1);
        }
        self.seen += 1;
        (correct, bits)
    }
}

/// Unigram, bigram and trigram baselines with per-window accumulators.
#[derive(Debug, Clone, Default)]
pub struct Baselines {
    uni: CountModel<0>,
    bi: CountModel<1>,
    tri: CountModel<2>,
    pub correct: [u64; 3],
    pub bits: [f64; 3],
}

impl Baselines {
    pub fn observe(&mut self, b: u8) {
        let r = [self.uni.observe(b), self.bi.observe(b), self.tri.observe(b)];
        for (i, (ok, bits)) in r.into_iter().enumerate() {
            self.correct[i] += u64::from(ok);
            self.bits[i] += bits;
        }
    }

    fn reset_window(&mut self) {
        self.correct = [0; 3];
        self.bits = [0.0; 3];
    }
}

/// Counters the kernel bumps as it runs. Window fields reset at each report.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub events: u64,
    pub bytes_total: u64,
    pub fires: u64,
    pub hidden_fires: u64,
    pub e2_sum: f64,
    pub bytes: u64,
    pub correct: u64,
    pub surprise_sum: f64,
    pub calibrator: Calibrator,
    pub baselines: Baselines,
    pub depth_sum: u64,
    pub depth_windows: u64,
    /// Hidden fires whose freshest ancestry is an older byte (or none): reverberation.
    pub reverb_fires: u64,
    /// Readout coverage: hot input cells summed over bytes, and bytes whose true cell was hot.
    pub hot_inputs: u64,
    pub truth_hot: u64,
    pub max_tick_fires: u64,
    /// Distinct cells fired this window (stamp dedup).
    pub fired: Vec<u32>,
    stamp: Vec<u32>,
    window: u32,
    byte_depth: u32,
    cur_tick: u64,
    cur_tick_fires: u64,
}

impl Stats {
    pub fn on_fire(&mut self, c: usize, t: u64) {
        self.fires += 1;
        if t != self.cur_tick {
            self.cur_tick = t;
            self.cur_tick_fires = 0;
        }
        self.cur_tick_fires += 1;
        self.max_tick_fires = self.max_tick_fires.max(self.cur_tick_fires);
        if self.stamp.len() <= c {
            self.stamp.resize(c + 1, u32::MAX);
        }
        if self.stamp[c] != self.window {
            self.stamp[c] = self.window;
            self.fired.push(c as u32);
        }
    }

    /// Before a byte is scored: how many input cells heard anything, and whether the true one did.
    pub fn on_coverage(&mut self, hot: usize, truth_hot: bool) {
        self.hot_inputs += hot as u64;
        self.truth_hot += u64::from(truth_hot);
    }

    pub fn on_hidden_fire(&mut self, e2: S, wave: Wave) {
        self.hidden_fires += 1;
        self.e2_sum += e2 as f64;
        if wave.origin == self.bytes_total && wave.origin > 0 {
            self.byte_depth = self.byte_depth.max(wave.depth);
        } else {
            self.reverb_fires += 1;
        }
    }

    /// A byte arrived: close the previous inter-byte window (hop depth), score the guess, and
    /// measure calibrated surprise from the 256 readout scores.
    pub fn on_byte(&mut self, correct: bool, scores: &[S], truth: usize) {
        let surprise = self.calibrator.observe(scores, truth);
        self.baselines.observe(truth as u8);
        if self.bytes_total > 0 {
            self.depth_sum += self.byte_depth as u64;
            self.depth_windows += 1;
        }
        self.byte_depth = 0;
        self.bytes_total += 1;
        self.bytes += 1;
        self.correct += u64::from(correct);
        self.surprise_sum += surprise;
    }

    fn reset_window(&mut self) {
        self.fires = 0;
        self.hidden_fires = 0;
        self.e2_sum = 0.0;
        self.bytes = 0;
        self.correct = 0;
        self.surprise_sum = 0.0;
        self.baselines.reset_window();
        self.depth_sum = 0;
        self.depth_windows = 0;
        self.reverb_fires = 0;
        self.hot_inputs = 0;
        self.truth_hot = 0;
        self.max_tick_fires = 0;
        self.cur_tick_fires = 0;
        self.fired.clear();
        self.window = self.window.wrapping_add(1);
    }
}

#[derive(Serialize, Debug, PartialEq)]
pub struct Dist {
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
}

pub fn dist(xs: &mut [f64]) -> Option<Dist> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(f64::total_cmp);
    let pct = |q: f64| xs[(q * (xs.len() - 1) as f64).round() as usize];
    Some(Dist {
        mean: xs.iter().sum::<f64>() / xs.len() as f64,
        p50: pct(0.5),
        p90: pct(0.9),
        p99: pct(0.99),
    })
}

#[derive(Serialize, Debug)]
pub struct PopStats {
    pub n: usize,
    pub tau: Option<Dist>,
    pub theta_mean: Option<f64>,
    pub v_mean: Option<f64>,
    pub rate_mean: Option<f64>,
}

fn pop_stats(eng: &Engine, cells: impl Iterator<Item = usize>, t: u64) -> PopStats {
    let (mut taus, mut theta, mut v, mut rate) = (Vec::new(), 0.0, 0.0, 0.0);
    for c in cells {
        let s = eng.peek(c, t);
        taus.push(eng.g.cells.tau[c] as f64);
        theta += rules::theta_eff(&s) as f64;
        v += s.v as f64;
        rate += s.rate as f64;
    }
    let n = taus.len();
    let mean = |x: f64| (n > 0).then(|| x / n as f64);
    PopStats {
        n,
        tau: dist(&mut taus),
        theta_mean: mean(theta),
        v_mean: mean(v),
        rate_mean: mean(rate),
    }
}

/// One deterministic metrics line (wall-clock fields are added separately and never hashed).
#[derive(Serialize, Debug)]
pub struct Metrics {
    pub kind: &'static str,
    pub tick: u64,
    pub bytes_seen: u64,
    pub cells_total: usize,
    pub cells_fired_in_window: usize,
    pub fires_in_window: u64,
    pub fires_per_tick: Option<f64>,
    pub max_fires_per_tick: u64,
    pub hidden_fires_in_window: u64,
    /// Next-byte guess accuracy over bytes in the window.
    pub accuracy: Option<f64>,
    /// Mean -log2 softmax(score / tau)[true byte] over bytes in the window, tau fitted online.
    pub surprise_bits: Option<f64>,
    /// Calibrated softmax temperature at the end of the window.
    pub surprise_temperature: Option<f64>,
    /// Online count baselines on the same windows: [unigram, bigram, trigram].
    pub baseline_accuracy: Option<[f64; 3]>,
    pub baseline_bits: Option<[f64; 3]>,
    /// Mean |e|^2 of hidden fires in the window.
    pub tension: Option<f64>,
    /// Mean, over inter-byte windows, of the furthest hop the current byte's wave reached
    /// (shortest-path hops from its input cell) before the next byte arrived.
    pub hop_depth: Option<f64>,
    /// Fraction of hidden fires driven by an older byte (or none): self-sustained activity.
    pub reverb_fraction: Option<f64>,
    /// Mean number of input cells that received any message in the window before a byte.
    pub hot_inputs_per_byte: Option<f64>,
    /// Fraction of bytes whose own input cell received any message before it arrived.
    pub truth_hot_fraction: Option<f64>,
    pub events_total: u64,
    pub pending_events: usize,
    /// Hidden cells that fired in the window (exact).
    pub fired: PopStats,
    /// Fixed deterministic sample of hidden cells (population view, O(RESERVOIR)).
    pub reservoir: PopStats,
}

/// FNV-1a 64: tiny, dependency-free, stable across Rust versions.
pub struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv {
    pub fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    pub fn u64(&mut self, x: u64) {
        self.bytes(&x.to_le_bytes());
    }
    pub fn f32s(&mut self, xs: &[f32]) {
        for x in xs {
            self.bytes(&x.to_bits().to_le_bytes());
        }
    }
    pub fn finish(&self) -> u64 {
        self.0
    }
}

pub struct Observatory {
    out: Option<Box<dyn Write>>,
    every: u64,
    next: u64,
    last_tick: u64,
    last_events: u64,
    reservoir: Vec<u32>,
    hash: Fnv,
    last_wall: Instant,
    start: Instant,
    pub reports: u64,
    pub last: Option<Metrics>,
}

impl Observatory {
    pub fn new(eng: &Engine, every: u64, seed: u64, out: Option<Box<dyn Write>>) -> Self {
        assert!(every > 0, "metrics interval must be > 0");
        let cells = &eng.g.cells;
        let mut pool: Vec<u32> = (0..cells.slots.capacity() as u32)
            .filter(|&c| cells.slots.is_alive(c) && cells.kind[c as usize] == Kind::Hidden)
            .collect();
        let mut rng = stream(seed, STREAM_RESERVOIR);
        let k = RESERVOIR.min(pool.len());
        for i in 0..k {
            let j = rng.random_range(i..pool.len());
            pool.swap(i, j);
        }
        pool.truncate(k);
        pool.sort_unstable();
        let now = Instant::now();
        Self {
            out,
            every,
            next: eng.now + every,
            last_tick: eng.now,
            last_events: eng.stats.events,
            reservoir: pool,
            hash: Fnv::default(),
            last_wall: now,
            start: now,
            reports: 0,
            last: None,
        }
    }

    /// Write a non-metrics line (header); hashed, so config differences show up in the hash.
    pub fn header(&mut self, v: &serde_json::Value) -> io::Result<()> {
        let line = v.to_string();
        self.hash.bytes(line.as_bytes());
        match &mut self.out {
            Some(w) => writeln!(w, "{line}"),
            None => Ok(()),
        }
    }

    /// Emit a report for every interval boundary <= t (called before processing events at t).
    pub fn observe_until(&mut self, eng: &mut Engine, t: u64) -> io::Result<()> {
        while self.next <= t {
            self.report(eng, self.next)?;
            self.next += self.every;
        }
        Ok(())
    }

    /// Final partial report covering everything processed since the last boundary.
    pub fn finish(&mut self, eng: &mut Engine) -> io::Result<()> {
        if eng.stats.events > self.last_events {
            self.report(eng, eng.now + 1)?;
        }
        match &mut self.out {
            Some(w) => w.flush(),
            None => Ok(()),
        }
    }

    pub fn metrics_hash(&self) -> u64 {
        self.hash.finish()
    }

    fn report(&mut self, eng: &mut Engine, tick: u64) -> io::Result<()> {
        let s = &eng.stats;
        let ratio = |a: f64, b: u64| (b > 0).then(|| a / b as f64);
        let hidden_fired: Vec<usize> = s
            .fired
            .iter()
            .map(|&c| c as usize)
            .filter(|&c| eng.g.cells.kind[c] == Kind::Hidden)
            .collect();
        let m = Metrics {
            kind: "metrics",
            tick,
            bytes_seen: s.bytes_total,
            cells_total: eng.g.cells.slots.live(),
            cells_fired_in_window: s.fired.len(),
            fires_in_window: s.fires,
            fires_per_tick: ratio(s.fires as f64, tick - self.last_tick),
            max_fires_per_tick: s.max_tick_fires,
            hidden_fires_in_window: s.hidden_fires,
            accuracy: ratio(s.correct as f64, s.bytes),
            surprise_bits: ratio(s.surprise_sum, s.bytes),
            surprise_temperature: (s.calibrator.beta > 0.0).then(|| 1.0 / s.calibrator.beta),
            baseline_accuracy: (s.bytes > 0)
                .then(|| s.baselines.correct.map(|c| c as f64 / s.bytes as f64)),
            baseline_bits: (s.bytes > 0).then(|| s.baselines.bits.map(|b| b / s.bytes as f64)),
            tension: ratio(s.e2_sum, s.hidden_fires),
            hop_depth: ratio(s.depth_sum as f64, s.depth_windows),
            reverb_fraction: ratio(s.reverb_fires as f64, s.hidden_fires),
            hot_inputs_per_byte: ratio(s.hot_inputs as f64, s.bytes),
            truth_hot_fraction: ratio(s.truth_hot as f64, s.bytes),
            events_total: s.events,
            pending_events: eng.pending_events(),
            fired: pop_stats(eng, hidden_fired.into_iter(), tick),
            reservoir: pop_stats(eng, self.reservoir.iter().map(|&c| c as usize), tick),
        };
        let line = serde_json::to_value(&m).map_err(io::Error::other)?;
        self.hash.bytes(line.to_string().as_bytes());
        let now = Instant::now();
        if let Some(w) = &mut self.out {
            let dt = now.duration_since(self.last_wall).as_secs_f64();
            let mut v = line;
            v["wall"] = json!({
                "events_per_sec": (eng.stats.events - self.last_events) as f64 / dt.max(1e-9),
                "elapsed_s": now.duration_since(self.start).as_secs_f64(),
            });
            writeln!(w, "{v}")?;
        }
        self.last_wall = now;
        self.last_events = eng.stats.events;
        self.last_tick = tick;
        self.reports += 1;
        self.last = Some(m);
        eng.stats.reset_window();
        Ok(())
    }
}

/// Hash of every bit of dynamical state: cells, edges, ring, pending events, clock.
pub fn state_hash(eng: &Engine) -> u64 {
    let mut h = Fnv::default();
    h.u64(eng.now);
    let cells = &eng.g.cells;
    for c in 0..cells.slots.capacity() {
        h.u64(cells.slots.id(c as u32).generation as u64);
        h.u64(cells.slots.is_alive(c as u32) as u64);
        h.u64(cells.kind[c] as u64);
        for v in [&cells.h[c], &cells.h_prev[c], &cells.u[c]] {
            h.f32s(v);
        }
        h.f32s(cells.w[c].as_flattened());
        h.f32s(cells.q[c].as_flattened());
        let s = cells.scalars(c);
        h.f32s(&[
            s.heat,
            s.log_theta,
            s.adapt,
            s.rate,
            s.v,
            cells.tau[c],
            cells.gain_norm0[c],
        ]);
        h.u64(cells.last_t[c]);
        h.u64(cells.wave[c].origin);
        h.u64(cells.wave[c].depth as u64);
        h.u64(cells.fire_pending[c] as u64);
        for cr in &cells.credit[c] {
            h.u64(((cr.edge.idx as u64) << 32) | cr.edge.generation as u64);
            h.f32s(&cr.m);
        }
        for &e in cells.outs[c].iter().chain(&cells.ins[c]) {
            h.u64(e as u64);
        }
    }
    let e = &eng.g.edges;
    for i in 0..e.slots.capacity() {
        h.u64(e.slots.id(i as u32).generation as u64);
        h.u64(e.slots.is_alive(i as u32) as u64);
        h.u64(((e.src[i] as u64) << 32) | e.dst[i] as u64);
        h.f32s(&[e.gain[i], e.vitality[i]]);
        h.u64(e.last_t[i]);
    }
    h.f32s(&eng.senses.bias);
    for &n in &eng.senses.bias_seen {
        h.u64(n);
    }
    h.u64(eng.ring.total());
    for ev in eng.ring.iter() {
        h.u64(((ev.cell.idx as u64) << 32) | ev.cell.generation as u64);
        h.u64(ev.t);
        h.bytes(&ev.h.map(|x| x as u8));
    }
    for ev in eng.scheduler().sorted() {
        h.u64(ev.t);
        h.u64(ev.phase as u64);
        h.u64(ev.cell as u64);
        match &ev.payload {
            Payload::Arrive => h.u64(0),
            Payload::Deliver { edge, m, wave } => {
                h.u64(((edge.idx as u64) << 32) | edge.generation as u64);
                h.f32s(m);
                h.u64(wave.origin);
                h.u64(wave.depth as u64);
            }
            Payload::Fire { generation } => h.u64(*generation as u64),
        }
    }
    h.finish()
}

/// Per-cell stats, one JSON line per live cell, decayed (read-only) to `eng.now`.
pub fn write_snapshot(eng: &Engine, w: &mut impl Write) -> io::Result<()> {
    let cells = &eng.g.cells;
    for c in 0..cells.slots.capacity() {
        if !cells.slots.is_alive(c as u32) {
            continue;
        }
        let s = eng.peek(c, eng.now);
        let row = json!({
            "idx": c,
            "generation": cells.slots.id(c as u32).generation,
            "kind": cells.kind[c],
            "theta": rules::theta_eff(&s),
            "log_theta": s.log_theta,
            "adapt": s.adapt,
            "rate": s.rate,
            "v": s.v,
            "tau": cells.tau[c],
            "heat": s.heat,
            "h_norm": norm(&cells.h[c]),
            "in_degree": cells.ins[c].len(),
            "out_degree": cells.outs[c].len(),
            "in_gain_norm": eng.g.in_gain_norm(c),
            "gain_norm0": cells.gain_norm0[c],
            "last_t": cells.last_t[c],
        });
        writeln!(w, "{row}")?;
    }
    w.flush()
}

/// Order-0 entropy of the corpus in bits per byte: the surprise of a unigram predictor.
pub fn unigram_entropy_bits(bytes: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let n = bytes.len() as f64;
    counts
        .iter()
        .filter(|&&k| k > 0)
        .map(|&k| k as f64 / n)
        .map(|p| -p * p.log2())
        .sum()
}

/// Accuracy of always guessing the most frequent byte: the accuracy baseline.
pub fn unigram_top1_accuracy(bytes: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    counts.iter().max().copied().unwrap_or(0) as f64 / bytes.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unigram_baselines_hand_cases() {
        assert_eq!(unigram_entropy_bits(b"aaaa"), 0.0);
        assert_eq!(unigram_entropy_bits(b"aabb"), 1.0);
        assert_eq!(unigram_entropy_bits(b"abcd"), 2.0);
        assert_eq!(unigram_top1_accuracy(b"aaab"), 0.75);
    }

    #[test]
    fn count_baselines_hand_cases() {
        // "abab...": unigram can be right half the time at best; bigram learns a->b, b->a.
        let mut m = Baselines::default();
        for &b in b"abababababababababab" {
            m.observe(b);
        }
        // Bigram misses the first 3 bytes (contexts 0, a, b are each new once): 17 right.
        assert_eq!(m.correct[1], 17);
        // Trigram misses the first 4 (contexts 00, 0a, ab, ba are each new once): 16 right.
        assert_eq!(m.correct[2], 16);
        assert!(m.correct[0] <= 10);
        // KT code length of the very first byte: -log2(0.5 / 128) = 8 bits.
        let mut u = CountModel::<0>::default();
        assert_eq!(u.observe(b'x'), (false, 8.0));
        let (ok, bits) = u.observe(b'x');
        assert!(ok && (bits - -(1.5f64 / 129.0).log2()).abs() < 1e-12);
    }

    #[test]
    fn calibrator_uniform_and_fit() {
        let mut c = Calibrator::default();
        assert_eq!(c.observe(&[0.0; 256], 7), 8.0);
        // 4 classes; the readout marks one with score 1. It is right half the time, otherwise the
        // truth is one of the other 3. Log-loss optimum: e^beta / (e^beta + 3) = 1/2 -> beta = ln 3.
        let mut c = Calibrator::default();
        for i in 0..40_000usize {
            let marked = i % 4;
            let truth = if i % 2 == 0 {
                marked
            } else {
                (marked + 1 + (i / 2) % 3) % 4
            };
            let mut s = [0.0; 4];
            s[marked] = 1.0;
            c.observe(&s, truth);
        }
        assert!((c.beta - 3f64.ln()).abs() < 0.1, "{}", c.beta);
    }

    #[test]
    fn dist_percentiles() {
        let mut xs: Vec<f64> = (1..=101).rev().map(f64::from).collect();
        let d = dist(&mut xs).unwrap();
        assert_eq!((d.mean, d.p50, d.p90, d.p99), (51.0, 51.0, 91.0, 100.0));
        assert!(dist(&mut []).is_none());
    }

    #[test]
    fn stats_dedup_and_windows() {
        let mut s = Stats::default();
        s.on_fire(3, 1);
        s.on_fire(3, 1);
        s.on_fire(4, 2);
        assert_eq!((s.fired.clone(), s.max_tick_fires), (vec![3, 4], 2));
        s.on_byte(false, &[0.0; 4], 0);
        s.on_hidden_fire(
            1.0,
            Wave {
                origin: 1,
                depth: 3,
            },
        );
        s.on_hidden_fire(
            1.0,
            Wave {
                origin: 0,
                depth: 9,
            },
        ); // no ancestry: reverberation
        s.on_byte(true, &[0.0; 4], 0);
        assert_eq!(
            (s.depth_sum, s.depth_windows, s.correct, s.reverb_fires),
            (3, 1, 1, 1)
        );
        s.reset_window();
        s.on_fire(3, 5);
        assert_eq!(s.fired, vec![3]);
    }
}
