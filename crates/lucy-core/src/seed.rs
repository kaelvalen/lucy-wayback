//! The initial graph. Fully determined by (seed, config); every random draw comes from its
//! own ChaCha stream, so changing one part (e.g. adding dormant cells) never perturbs another.

use std::ops::Range;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::kernel::Engine;
use crate::num::{D, M, S, ZERO, ZERO_M};
use crate::rules::Params;
use crate::senses::{N_BYTES, Senses};
use crate::store::{CellInit, Graph, Kind};

pub const STREAM_GRAPH: u64 = 0;
pub const STREAM_CODES: u64 = 1;
pub const STREAM_RESERVOIR: u64 = 2;
pub const STREAM_DORMANT: u64 = 3;
/// Observatory probe shuffling (never touches dynamics).
pub const STREAM_PROBE: u64 = 4;
/// Wayback dreams (seed cells of spontaneous runs during consolidation).
pub const STREAM_DREAM: u64 = 5;

pub fn stream(seed: u64, id: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed);
    r.set_stream(id);
    r
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeedConfig {
    pub seed: u64,
    pub hidden: usize,
    /// Hidden -> hidden out-edges per hidden cell.
    pub fanout: usize,
    /// Input -> hidden out-edges per input cell.
    pub input_fanout: usize,
    /// Hidden -> input in-edges per input cell (the prediction pathway).
    pub predict_fanin: usize,
    /// Gains ~ U(-init_gain, init_gain).
    pub init_gain: S,
    /// W entries ~ U(-a, a), a = init_w / sqrt(D) (spectral radius ~ init_w).
    pub init_w: S,
    /// Q entries ~ U(-a, a), a = init_q / sqrt(D).
    pub init_q: S,
    /// Starting threshold; homeostasis moves it from here.
    pub init_theta: S,
    pub ticks_per_byte: u64,
    pub ring_cap: usize,
}

impl Default for SeedConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            hidden: 4096,
            fanout: 8,
            input_fanout: 16,
            predict_fanin: 16,
            init_gain: 0.5,
            init_w: 0.5,
            init_q: 0.1,
            init_theta: 1.0,
            ticks_per_byte: 8,
            ring_cap: 1 << 18,
        }
    }
}

impl SeedConfig {
    pub fn validate(&self) -> Result<(), String> {
        let no_reservoir = self.hidden == 0
            && self.fanout == 0
            && self.input_fanout == 0
            && self.predict_fanin == 0;
        if self.hidden <= self.fanout && !no_reservoir {
            return Err(format!(
                "hidden ({}) must exceed fanout ({})",
                self.hidden, self.fanout
            ));
        }
        if self.hidden < self.input_fanout.max(self.predict_fanin) {
            return Err("hidden must be >= input_fanout and predict_fanin".into());
        }
        if self.ticks_per_byte == 0 || self.ring_cap == 0 {
            return Err("ticks_per_byte and ring_cap must be >= 1".into());
        }
        let finite = [self.init_gain, self.init_w, self.init_q, self.init_theta];
        if finite.iter().any(|x| !x.is_finite() || *x < 0.0) || self.init_theta <= 0.0 {
            return Err("init_* must be finite and >= 0, init_theta > 0".into());
        }
        Ok(())
    }
}

fn uniform(rng: &mut ChaCha8Rng, a: S) -> S {
    (rng.random::<S>() * 2.0 - 1.0) * a
}

fn rand_mat(rng: &mut ChaCha8Rng, scale: S) -> M {
    let a = scale / (D as S).sqrt();
    std::array::from_fn(|_| std::array::from_fn(|_| uniform(rng, a)))
}

/// `k` distinct indices from `range`, excluding `exclude`.
fn sample_distinct(
    rng: &mut ChaCha8Rng,
    range: Range<u32>,
    k: usize,
    exclude: Option<u32>,
) -> Vec<u32> {
    let avail = range.len() - usize::from(exclude.is_some_and(|x| range.contains(&x)));
    assert!(k <= avail, "cannot sample {k} distinct from {avail}");
    let mut out = Vec::with_capacity(k);
    while out.len() < k {
        let x = rng.random_range(range.clone());
        if Some(x) != exclude && !out.contains(&x) {
            out.push(x);
        }
    }
    out
}

fn add_hidden(
    g: &mut Graph,
    rng: &mut ChaCha8Rng,
    cfg: &SeedConfig,
    p: &Params,
    n: usize,
    t: u64,
) -> Range<u32> {
    let start = g.cells.slots.capacity() as u32;
    for _ in 0..n {
        let (w, q) = (rand_mat(rng, cfg.init_w), rand_mat(rng, cfg.init_q));
        let init = CellInit {
            kind: Kind::Hidden,
            h: ZERO,
            w,
            q,
            log_theta: cfg.init_theta.ln(),
            rate: p.target_rate,
            t,
        };
        let id = g.add_cell(init);
        assert_eq!(id.generation, 0, "seed expects fresh, contiguous slots");
    }
    start..start + n as u32
}

fn wire_recurrent(
    g: &mut Graph,
    rng: &mut ChaCha8Rng,
    cells: Range<u32>,
    fanout: usize,
    init_gain: S,
    t: u64,
) {
    for c in cells.clone() {
        for d in sample_distinct(rng, cells.clone(), fanout, Some(c)) {
            let gain = uniform(rng, init_gain);
            g.add_edge(c, d, gain, t);
        }
    }
}

/// Build the seed graph: 256 input cells (idx == byte), `hidden` hidden cells, and
/// input -> hidden, hidden -> hidden, hidden -> input edges.
pub fn build(cfg: &SeedConfig, p: Params) -> Engine {
    let mut rng = stream(cfg.seed, STREAM_GRAPH);
    let mut g = Graph::default();
    for _ in 0..N_BYTES {
        let init = CellInit {
            kind: Kind::Input,
            h: ZERO,
            w: ZERO_M,
            q: ZERO_M,
            log_theta: cfg.init_theta.ln(),
            rate: p.target_rate,
            t: 0,
        };
        g.add_cell(init);
    }
    let hidden = add_hidden(&mut g, &mut rng, cfg, &p, cfg.hidden, 0);
    wire_recurrent(
        &mut g,
        &mut rng,
        hidden.clone(),
        cfg.fanout,
        cfg.init_gain,
        0,
    );
    for b in 0..N_BYTES as u32 {
        for d in sample_distinct(&mut rng, hidden.clone(), cfg.input_fanout, None) {
            let gain = uniform(&mut rng, cfg.init_gain);
            g.add_edge(b, d, gain, 0);
        }
        for s in sample_distinct(&mut rng, hidden.clone(), cfg.predict_fanin, None) {
            let gain = uniform(&mut rng, cfg.init_gain);
            g.add_edge(s, b, gain, 0);
        }
    }
    g.fix_gain_norms(0..g.cells.slots.capacity());
    let mut eng = Engine::new(
        p,
        g,
        Senses::new(cfg.seed, cfg.ticks_per_byte),
        cfg.ring_cap,
    );
    eng.plasticity.dream_seed = cfg.seed;
    eng
}

/// Add `n` hidden cells wired only among themselves: never reached by input, so they must cost
/// nothing. Used by the scaling benchmark.
pub fn add_dormant(eng: &mut Engine, cfg: &SeedConfig, n: usize) {
    let mut rng = stream(cfg.seed, STREAM_DORMANT);
    let t = eng.now;
    let cells = add_hidden(&mut eng.g, &mut rng, cfg, &eng.p, n, t);
    wire_recurrent(
        &mut eng.g,
        &mut rng,
        cells.clone(),
        cfg.fanout,
        cfg.init_gain,
        t,
    );
    eng.g
        .fix_gain_norms(cells.start as usize..cells.end as usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> SeedConfig {
        SeedConfig {
            hidden: 300,
            ..SeedConfig::default()
        }
    }

    #[test]
    fn seed_graph_shape() {
        let cfg = small();
        let eng = build(&cfg, Params::default());
        let c = &eng.g.cells;
        assert_eq!(c.slots.live(), 256 + 300);
        assert_eq!(eng.g.edges.slots.live(), 300 * 8 + 256 * 16 * 2);
        for b in 0..256 {
            assert_eq!(c.kind[b], Kind::Input);
            assert_eq!(c.outs[b].len(), 16);
            assert_eq!(c.ins[b].len(), 16);
        }
        for h in 256..556 {
            // Out-edges = 8 recurrent (hidden) + some prediction edges to inputs.
            let mut dsts: Vec<u32> = c.outs[h]
                .iter()
                .map(|&e| eng.g.edges.dst[e as usize])
                .filter(|&d| d >= 256)
                .collect();
            assert!(dsts.iter().all(|&d| d as usize != h), "no self-loops");
            dsts.sort_unstable();
            dsts.dedup();
            assert_eq!(dsts.len(), 8, "distinct targets");
        }
        assert!((eng.g.in_gain_norm(300) - c.gain_norm0[300]).abs() < 1e-6);
    }

    #[test]
    fn seed_is_deterministic() {
        let a = build(&small(), Params::default());
        let b = build(&small(), Params::default());
        assert_eq!(a.g.edges.gain, b.g.edges.gain);
        assert_eq!(a.g.cells.w, b.g.cells.w);
        let c = build(&SeedConfig { seed: 1, ..small() }, Params::default());
        assert_ne!(a.g.edges.gain, c.g.edges.gain);
    }

    #[test]
    fn dormant_cells_do_not_perturb_the_seed() {
        let a = build(&small(), Params::default());
        let mut b = build(&small(), Params::default());
        add_dormant(&mut b, &small(), 500);
        assert_eq!(b.g.cells.slots.live(), a.g.cells.slots.live() + 500);
        let n = a.g.edges.gain.len();
        assert_eq!(a.g.edges.gain[..], b.g.edges.gain[..n]);
    }
}
