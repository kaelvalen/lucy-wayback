//! Structural plasticity, first slice: co-activation edge growth into input cells.
//!
//! Every `consolidate_every` bytes the engine replays its fire recording (the ring buffer) since
//! the last pass, an offline, sleep-like consolidation. A byte window is the set of hidden cells
//! that fired between two byte arrivals (the K ticks before the second). Hidden cells that fire
//! before byte `b` more often than chance get a new edge `h -> b` with a small gain. In-edges of
//! input cells live on *earned* vitality (credited when they fed the cell whose byte then really
//! arrived, see `kernel::arrive`): below `prune_vitality` they die, and above `input_in_cap`
//! the lowest-vitality edge dies first.
//!
//! Cost per pass is proportional to fires recorded since the last pass plus the 256 input cells'
//! in-edges: it scales with activity, never with total graph size. Deterministic: every map is
//! ordered and every tie is broken by index.

use std::collections::BTreeMap;

use crate::kernel::Engine;
use crate::num::S;
use crate::rules;
use crate::senses::N_BYTES;
use crate::store::Kind;

/// Consolidation bookkeeping (not dynamics state: it only decides when and from where to replay).
#[derive(Clone, Debug, Default)]
pub struct Plasticity {
    /// First ring sequence number not yet replayed.
    pub next_seq: u64,
    pub passes: u64,
}

/// What one consolidation pass did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassReport {
    pub windows: u64,
    pub grown: u64,
    pub pruned: u64,
    /// Fire events overwritten in the ring before they could be replayed.
    pub lost_events: u64,
}

/// Replay, grow, prune. Called by the kernel at a byte arrival (`t`).
pub fn consolidate(eng: &mut Engine, t: u64) -> PassReport {
    let p = &eng.p;
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
    for seq in from..total {
        let ev = *eng.ring.get(seq).expect("retained");
        if !eng.g.cells.slots.is_live(ev.cell) {
            continue;
        }
        match eng.g.cells.kind[ev.cell.idx as usize] {
            Kind::Hidden => window.push(ev.cell.idx),
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

    // 2. Grow: per byte cell, the hidden cells with the largest excess co-occurrence over chance
    //    (N(h,b) - f_h n_b / W) that are not already connected.
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
            if taken >= p.grow_per_cell {
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
            grow_per_cell: 3,
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
    fn deterministic() {
        let p = Params {
            consolidate_every: 150,
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
