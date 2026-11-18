//! Structural plasticity, first slice: co-activation edge growth into input cells.
//!
//! Every `consolidate_every` bytes the engine replays its fire recording (the ring buffer) since
//! the last pass, an offline, sleep-like consolidation. A byte window is the set of hidden cells
//! that fired between two byte arrivals (the K ticks before the second). Hidden cells that fire
//! before byte `b` more often than chance get a new edge `h -> b` with a small gain. How many a
//! cell may grow is its tension budget (`rules::growth_budget`): proportional to how long its
//! prediction error stayed above its own running floor since the last pass, so a cell whose error
//! keeps falling grows nothing. In-edges of
//! input cells live on *earned* vitality (credited when they fed the cell whose byte then really
//! arrived, see `kernel::arrive`): below `prune_vitality` they die, and above `input_in_cap`
//! the lowest-vitality edge dies first.
//!
//! Cost per pass is proportional to fires recorded since the last pass plus the 256 input cells'
//! in-edges: it scales with activity, never with total graph size. Deterministic: every map is
//! ordered and every tie is broken by index.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};

use crate::kernel::Engine;
use crate::num::S;
use crate::rules;
use crate::senses::N_BYTES;
use crate::store::Kind;

/// Consolidation state: where to replay from, and each input cell's tension (growth law).
#[derive(Clone, Debug)]
pub struct Plasticity {
    /// First ring sequence number not yet replayed.
    pub next_seq: u64,
    pub passes: u64,
    /// Per input cell: error EMA, its running floor, excess accumulated since the last pass,
    /// fractional growth budget carried over.
    pub err: Vec<S>,
    pub floor: Vec<S>,
    pub excess: Vec<S>,
    pub carry: Vec<S>,
}

impl Default for Plasticity {
    fn default() -> Self {
        // Error 1 = "predicts nothing" (pred = 0, |code|^2 / D = 1).
        Self {
            next_seq: 0,
            passes: 0,
            err: vec![1.0; N_BYTES],
            floor: vec![1.0; N_BYTES],
            excess: vec![0.0; N_BYTES],
            carry: vec![0.0; N_BYTES],
        }
    }
}

impl Plasticity {
    /// Byte `b` arrived and its cell's readout missed by `x = |code - pred|^2 / D`.
    pub fn observe_error(&mut self, b: usize, x: S, p: &rules::Params) {
        let (err, floor, excess) = rules::tension_step(self.err[b], self.floor[b], x, p);
        self.err[b] = err;
        self.floor[b] = floor;
        self.excess[b] += excess;
    }
}

/// What one consolidation pass did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassReport {
    pub windows: u64,
    pub grown: u64,
    pub pruned: u64,
    /// Sum over input cells of their growth budgets this pass.
    pub budget: u64,
    /// Hidden -> hidden edges grown / pruned this pass.
    pub grown_hidden: u64,
    pub pruned_hidden: u64,
    /// Fire events overwritten in the ring before they could be replayed.
    pub lost_events: u64,
}

/// Replay, grow, prune. Called by the kernel at a byte arrival (`t`).
pub fn consolidate(eng: &mut Engine, t: u64) -> PassReport {
    let p = &eng.p.clone();
    let mut rep = PassReport::default();
    let total = eng.ring.total();
    let from = eng.plasticity.next_seq.max(total - eng.ring.len() as u64);
    rep.lost_events = from - eng.plasticity.next_seq;
    eng.plasticity.next_seq = total;
    eng.plasticity.passes += 1;

    // 1. Replay: co-occurrence counts N(h, b), hidden window counts f_h, byte counts n_b.
    let mut pairs: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut f_h: BTreeMap<u32, u32> = BTreeMap::new();
    let mut n_b = [0u32; N_BYTES];
    let mut window: Vec<u32> = Vec::new();
    let mut started = false; // the first window may be cut off by the pass boundary
    let mut seq_pairs = SeqPairs::new(
        p.hidden_grow_per_pass > 0,
        u64::from(p.hidden_grow_lag),
        p.hidden_grow_stride,
        (eng.plasticity.passes % u64::from(p.hidden_grow_stride.max(1))) as u32,
    );
    for seq in from..total {
        let ev = *eng.ring.get(seq).expect("retained");
        if !eng.g.cells.slots.is_live(ev.cell) {
            continue;
        }
        match eng.g.cells.kind[ev.cell.idx as usize] {
            Kind::Hidden => {
                window.push(ev.cell.idx);
                seq_pairs.fire(ev.cell.idx, ev.t);
            }
            Kind::Input => {
                if started {
                    window.sort_unstable();
                    window.dedup();
                    let b = ev.cell.idx;
                    n_b[b as usize] += 1;
                    for &h in &window {
                        *pairs.entry((b, h)).or_default() += 1;
                        *f_h.entry(h).or_default() += 1;
                    }
                    rep.windows += 1;
                }
                window.clear();
                started = true;
            }
        }
    }

    // 2. Grow: per byte cell, up to its tension budget, the hidden cells with the largest excess
    //    co-occurrence over chance (N(h,b) - f_h n_b / W) that are not already connected.
    let mut budget = [0u32; N_BYTES];
    for (b, n) in budget.iter_mut().enumerate() {
        let pl = &mut eng.plasticity;
        (*n, pl.carry[b]) = rules::growth_budget(pl.excess[b], pl.carry[b], p);
        pl.excess[b] = 0.0;
        rep.budget += u64::from(*n);
    }
    if rep.windows > 0 {
        let w = rep.windows as f64;
        let mut cand: Vec<(u32, f64, u32, u32)> = pairs
            .iter()
            .filter_map(|(&(b, h), &n)| {
                let excess = n as f64 - f_h[&h] as f64 * n_b[b as usize] as f64 / w;
                (excess > 0.0).then_some((b, excess, h, n))
            })
            .collect();
        cand.sort_by(|x, y| x.0.cmp(&y.0).then(y.1.total_cmp(&x.1)).then(x.2.cmp(&y.2)));
        let mut taken = 0;
        let mut cur = u32::MAX;
        for (b, _, h, n) in cand {
            if b != cur {
                cur = b;
                taken = 0;
            }
            if taken >= budget[b as usize] {
                continue;
            }
            let ins = &eng.g.cells.ins[b as usize];
            if ins.iter().any(|&e| eng.g.edges.src[e as usize] == h) {
                continue;
            }
            let e = eng.g.add_edge(h, b, p.grow_gain, t);
            eng.g.edges.vitality[e.idx as usize] = n as S; // it already earned n co-occurrences
            taken += 1;
            rep.grown += 1;
        }
    }

    // 2b. Hidden -> hidden: the pairs with the largest excess sequential co-firing.
    if p.hidden_grow_per_pass > 0 {
        let (g, pr) = grow_hidden(eng, &seq_pairs, t);
        rep.grown_hidden = g;
        rep.pruned_hidden = pr;
    }

    // 3. Prune input cells' in-edges: below the vitality floor, then lowest-first over the cap.
    for b in 0..N_BYTES {
        let mut scored: Vec<(S, u32)> = eng.g.cells.ins[b]
            .iter()
            .map(|&e| {
                let i = e as usize;
                let v = eng.g.edges.vitality[i] * rules::decay(t - eng.g.edges.last_t[i], p.t_v);
                (v, e)
            })
            .collect();
        scored.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        let over = scored.len().saturating_sub(p.input_in_cap as usize);
        for (k, &(v, e)) in scored.iter().enumerate() {
            if k < over || v < p.prune_vitality {
                let id = eng.g.edges.slots.id(e);
                eng.g.remove_edge(id);
                rep.pruned += 1;
            }
        }
    }
    rep
}

/// Multiply-shift hasher for u64 keys: deterministic and fast (iteration order is never used:
/// candidates are fully sorted before any decision).
#[derive(Default)]
struct Mix(u64);

impl Hasher for Mix {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(self.0 ^ b as u64);
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = (x ^ (x >> 31)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    fn write_u32(&mut self, x: u32) {
        self.write_u64(x as u64);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<Mix>>;

/// Sequential co-firing counts during replay: N(A -> B) = times B fired 1..=lag ticks after A.
struct SeqPairs {
    on: bool,
    lag: u64,
    stride: u32,
    phase: u32,
    recent: VecDeque<(u64, u32)>,
    pairs: FastMap<u64, u32>,
    fires: FastMap<u32, u32>,
    span: Option<(u64, u64)>,
}

impl SeqPairs {
    fn new(on: bool, lag: u64, stride: u32, phase: u32) -> Self {
        Self {
            on,
            lag,
            stride,
            phase,
            recent: VecDeque::new(),
            pairs: FastMap::default(),
            fires: FastMap::default(),
            span: None,
        }
    }

    fn fire(&mut self, c: u32, t: u64) {
        if !self.on {
            return;
        }
        while self
            .recent
            .front()
            .is_some_and(|&(t0, _)| t0 + self.lag < t)
        {
            self.recent.pop_front();
        }
        if c % self.stride == self.phase {
            for &(t0, a) in &self.recent {
                if t0 < t && a != c {
                    *self.pairs.entry(((a as u64) << 32) | c as u64).or_default() += 1;
                }
            }
        }
        self.recent.push_back((t, c));
        *self.fires.entry(c).or_default() += 1;
        let (lo, _) = self.span.unwrap_or((t, t));
        self.span = Some((lo, t));
    }
}

/// Grow the `hidden_grow_per_pass` unconnected pairs with the largest excess
/// `N(A->B) - f_A f_B lag / T`, then enforce the hidden in-degree cap on their targets
/// (lowest decayed usage vitality dies first; edges from input cells never compete).
fn grow_hidden(eng: &mut Engine, sp: &SeqPairs, t: u64) -> (u64, u64) {
    let p = eng.p.clone();
    let Some((lo, hi)) = sp.span else {
        return (0, 0);
    };
    let ticks = (hi - lo + 1) as f64;
    let mut cand: Vec<(f64, u64)> = sp
        .pairs
        .iter()
        .filter_map(|(&k, &n)| {
            let (a, b) = ((k >> 32) as u32, k as u32);
            let expect = sp.fires[&a] as f64 * sp.fires[&b] as f64 * sp.lag as f64 / ticks;
            let excess = n as f64 - expect;
            (excess > 0.0).then_some((excess, k))
        })
        .collect();
    cand.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    let (mut grown, mut pruned) = (0, 0);
    let mut targets = Vec::new();
    for (_, k) in cand {
        if grown >= u64::from(p.hidden_grow_per_pass) {
            break;
        }
        let (a, b) = ((k >> 32) as u32, k as u32);
        let outs = &eng.g.cells.outs[a as usize];
        if outs.iter().any(|&e| eng.g.edges.dst[e as usize] == b) {
            continue;
        }
        let e = eng.g.add_edge(a, b, p.grow_gain, t);
        eng.g.edges.vitality[e.idx as usize] = sp.pairs[&k] as S;
        targets.push(b);
        grown += 1;
    }
    targets.sort_unstable();
    targets.dedup();
    for b in targets {
        let mut scored: Vec<(S, u32)> = eng.g.cells.ins[b as usize]
            .iter()
            .filter(|&&e| eng.g.cells.kind[eng.g.edges.src[e as usize] as usize] == Kind::Hidden)
            .map(|&e| {
                let i = e as usize;
                let v = eng.g.edges.vitality[i] * rules::decay(t - eng.g.edges.last_t[i], p.t_v);
                (v, e)
            })
            .collect();
        scored.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        let over = scored.len().saturating_sub(p.hidden_in_cap as usize);
        for (k, &(v, e)) in scored.iter().enumerate() {
            if k < over || v < p.prune_vitality {
                let id = eng.g.edges.slots.id(e);
                eng.g.remove_edge(id);
                pruned += 1;
            }
        }
    }
    (grown, pruned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::Params;
    use crate::seed::{self, SeedConfig};

    fn engine(p: Params) -> Engine {
        seed::build(
            &SeedConfig {
                hidden: 300,
                ..SeedConfig::default()
            },
            p,
        )
    }

    fn run_bytes(eng: &mut Engine, text: &[u8], ticks: u64) {
        let mut src = text.iter().copied();
        while eng.next_time().is_some_and(|t| t < ticks) {
            eng.step(&mut src);
        }
    }

    #[test]
    fn disabled_by_zero_period() {
        let mut eng = engine(Params {
            consolidate_every: 0,
            ..Params::default()
        });
        let edges = eng.g.edges.slots.live();
        run_bytes(&mut eng, &b"abcab".repeat(400), 12_000);
        assert_eq!(
            (eng.plasticity.passes, eng.g.edges.slots.live()),
            (0, edges)
        );
    }

    #[test]
    fn grows_predictive_edges_and_respects_cap() {
        let p = Params {
            consolidate_every: 200,
            grow_per_tension: 10.0,
            input_in_cap: 18,
            ..Params::default()
        };
        let cap = p.input_in_cap as usize;
        let mut eng = engine(p);
        run_bytes(&mut eng, &b"the cat sat on the mat. ".repeat(100), 8 * 2000);
        assert!(eng.plasticity.passes >= 9);
        assert!(eng.stats.edges_grown > 0, "co-activation must create edges");
        for b in 0..N_BYTES {
            assert!(eng.g.cells.ins[b].len() <= cap, "byte {b} over cap");
            let mut srcs: Vec<u32> = eng.g.cells.ins[b]
                .iter()
                .map(|&e| eng.g.edges.src[e as usize])
                .collect();
            assert!(
                srcs.iter()
                    .all(|&s| eng.g.cells.kind[s as usize] == Kind::Hidden)
            );
            let n = srcs.len();
            srcs.sort_unstable();
            srcs.dedup();
            assert_eq!(srcs.len(), n, "byte {b}: duplicate source");
        }
    }

    #[test]
    fn hidden_growth_follows_sequences_and_respects_cap() {
        let p = Params {
            consolidate_every: 200,
            hidden_grow_per_pass: 50,
            hidden_in_cap: 12,
            ..Params::default()
        };
        let cap = p.hidden_in_cap as usize;
        let mut eng = engine(p);
        let before = eng.g.edges.slots.live();
        let hidden_in = |eng: &Engine, c: usize| {
            eng.g.cells.ins[c]
                .iter()
                .filter(|&&e| {
                    eng.g.cells.kind[eng.g.edges.src[e as usize] as usize] == Kind::Hidden
                })
                .count()
        };
        let seed_in: Vec<usize> = (0..eng.g.cells.slots.capacity())
            .map(|c| hidden_in(&eng, c))
            .collect();
        run_bytes(&mut eng, &b"the cat sat on the mat. ".repeat(100), 8 * 2000);
        assert!(
            eng.g.edges.slots.live() > before,
            "hidden growth must add edges"
        );
        for (c, &seed) in seed_in.iter().enumerate().skip(N_BYTES) {
            let ins = &eng.g.cells.ins[c];
            let hidden_ins = hidden_in(&eng, c);
            // Growth never pushes a cell over the cap (a seed fan-in above it is left alone).
            assert!(hidden_ins <= cap.max(seed), "cell {c}: {hidden_ins}");
            let mut srcs: Vec<u32> = ins.iter().map(|&e| eng.g.edges.src[e as usize]).collect();
            assert!(!srcs.contains(&(c as u32)), "self-loop on {c}");
            let n = srcs.len();
            srcs.sort_unstable();
            srcs.dedup();
            assert_eq!(srcs.len(), n, "cell {c}: duplicate source");
        }
    }

    #[test]
    fn deterministic() {
        let p = Params {
            consolidate_every: 150,
            hidden_grow_per_pass: 20,
            ..Params::default()
        };
        let run = || {
            let mut eng = engine(p.clone());
            run_bytes(&mut eng, &b"abracadabra ".repeat(100), 8 * 1000);
            crate::observe::state_hash(&eng)
        };
        assert_eq!(run(), run());
    }
}
