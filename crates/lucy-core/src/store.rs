//! Slab storage: struct-of-arrays cell and edge columns with free lists and generation
//! counters, so ids stay unambiguous across future growth, splitting and death.

use serde::{Deserialize, Serialize};

use crate::num::{M, S, V, ZERO};
use crate::rules::Scalars;

/// Slab handle. `generation` changes every time a slot is released, so stale ids are detectable.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct Id {
    pub idx: u32,
    pub generation: u32,
}
pub type CellId = Id;
pub type EdgeId = Id;

/// Slot allocator shared by cells and edges: LIFO free list (deterministic) + generations.
#[derive(Clone, Debug, Default)]
pub struct Slots {
    generation: Vec<u32>,
    alive: Vec<bool>,
    free: Vec<u32>,
    live: usize,
}

impl Slots {
    /// Returns the id and whether the slot is fresh (columns must be pushed) or reused (overwritten).
    pub fn alloc(&mut self) -> (Id, bool) {
        self.live += 1;
        match self.free.pop() {
            Some(idx) => {
                self.alive[idx as usize] = true;
                (
                    Id {
                        idx,
                        generation: self.generation[idx as usize],
                    },
                    false,
                )
            }
            None => {
                let idx = u32::try_from(self.generation.len()).expect("slab exceeds u32 ids");
                self.generation.push(0);
                self.alive.push(true);
                (Id { idx, generation: 0 }, true)
            }
        }
    }

    pub fn release(&mut self, id: Id) -> bool {
        if !self.is_live(id) {
            return false;
        }
        let i = id.idx as usize;
        self.alive[i] = false;
        self.generation[i] = self.generation[i].wrapping_add(1);
        self.free.push(id.idx);
        self.live -= 1;
        true
    }

    pub fn is_live(&self, id: Id) -> bool {
        self.is_alive(id.idx) && self.generation[id.idx as usize] == id.generation
    }

    pub fn is_alive(&self, idx: u32) -> bool {
        self.alive.get(idx as usize).copied().unwrap_or(false)
    }

    pub fn id(&self, idx: u32) -> Id {
        Id {
            idx,
            generation: self.generation[idx as usize],
        }
    }

    /// High-water mark of slots (live + free).
    pub fn capacity(&self) -> usize {
        self.generation.len()
    }

    pub fn live(&self) -> usize {
        self.live
    }
}

fn put<T>(col: &mut Vec<T>, i: usize, val: T, fresh: bool) {
    if fresh { col.push(val) } else { col[i] = val }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    /// Byte sensor and next-byte predictor (its sub-threshold heat is the prediction).
    Input,
    Hidden,
}

/// Ancestry of activity: which byte (1-based arrival number, 0 = none) started the wave and how
/// many hops it has travelled. Merge: the newer byte wins; same byte keeps the shortest path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Wave {
    pub origin: u64,
    pub depth: u32,
}

impl Wave {
    pub fn merge(self, o: Wave) -> Wave {
        match o.origin.cmp(&self.origin) {
            std::cmp::Ordering::Greater => o,
            std::cmp::Ordering::Equal => Wave {
                origin: self.origin,
                depth: self.depth.min(o.depth),
            },
            std::cmp::Ordering::Less => self,
        }
    }

    pub fn next_hop(self) -> Wave {
        Wave {
            origin: self.origin,
            depth: self.depth + 1,
        }
    }
}

/// Decayed sum of the messages one in-edge contributed since the cell last fired.
#[derive(Clone, Copy, Debug)]
pub struct Credit {
    pub edge: EdgeId,
    pub m: V,
}

pub struct CellInit {
    pub kind: Kind,
    pub h: V,
    pub w: M,
    pub q: M,
    pub log_theta: S,
    pub rate: S,
    pub t: u64,
}

/// Cell columns (struct of arrays), indexed by `Id::idx`.
#[derive(Default, Clone)]
pub struct CellStore {
    pub slots: Slots,
    pub kind: Vec<Kind>,
    pub h: Vec<V>,
    pub h_prev: Vec<V>,
    pub u: Vec<V>,
    pub w: Vec<M>,
    pub q: Vec<M>,
    pub heat: Vec<S>,
    pub log_theta: Vec<S>,
    pub adapt: Vec<S>,
    pub rate: Vec<S>,
    pub v: Vec<S>,
    pub tau: Vec<S>,
    /// Birth L2 norm of in-gains: the synaptic-scaling target (fix 4).
    pub gain_norm0: Vec<S>,
    pub last_t: Vec<u64>,
    /// Freshest wave integrated since the last fire.
    pub wave: Vec<Wave>,
    pub fire_pending: Vec<bool>,
    /// Ring sequence number of the cell's latest fire (`buffer::NO_PREV` if none).
    pub last_seq: Vec<u64>,
    /// Birth Frobenius norm of W: the cap of W-norm homeostasis (wayback replay).
    pub w_norm0: Vec<S>,
    pub outs: Vec<Vec<u32>>,
    pub ins: Vec<Vec<u32>>,
    pub credit: Vec<Vec<Credit>>,
}

impl CellStore {
    fn add(&mut self, init: CellInit) -> CellId {
        let (id, fresh) = self.slots.alloc();
        let i = id.idx as usize;
        put(&mut self.kind, i, init.kind, fresh);
        put(&mut self.h, i, init.h, fresh);
        put(&mut self.h_prev, i, ZERO, fresh);
        put(&mut self.u, i, ZERO, fresh);
        put(&mut self.w, i, init.w, fresh);
        put(&mut self.q, i, init.q, fresh);
        put(&mut self.heat, i, 0.0, fresh);
        put(&mut self.log_theta, i, init.log_theta, fresh);
        put(&mut self.adapt, i, 0.0, fresh);
        put(&mut self.rate, i, init.rate, fresh);
        put(&mut self.v, i, 1.0, fresh);
        put(&mut self.tau, i, 0.0, fresh);
        put(&mut self.gain_norm0, i, 0.0, fresh);
        put(&mut self.last_t, i, init.t, fresh);
        put(&mut self.wave, i, Wave::default(), fresh);
        put(&mut self.fire_pending, i, false, fresh);
        put(&mut self.last_seq, i, crate::buffer::NO_PREV, fresh);
        let w_norm = init
            .w
            .as_flattened()
            .iter()
            .map(|x| x * x)
            .sum::<S>()
            .sqrt();
        put(&mut self.w_norm0, i, w_norm, fresh);
        put(&mut self.outs, i, Vec::new(), fresh);
        put(&mut self.ins, i, Vec::new(), fresh);
        put(&mut self.credit, i, Vec::new(), fresh);
        id
    }

    pub fn scalars(&self, c: usize) -> Scalars {
        Scalars {
            heat: self.heat[c],
            log_theta: self.log_theta[c],
            adapt: self.adapt[c],
            rate: self.rate[c],
            v: self.v[c],
        }
    }

    pub fn set_scalars(&mut self, c: usize, s: Scalars) {
        self.heat[c] = s.heat;
        self.log_theta[c] = s.log_theta;
        self.adapt[c] = s.adapt;
        self.rate[c] = s.rate;
        self.v[c] = s.v;
    }

    /// Clear the integration window (u, heat, credit, wave).
    pub fn clear_input(&mut self, c: usize) {
        self.u[c] = ZERO;
        self.heat[c] = 0.0;
        self.credit[c].clear();
        self.wave[c] = Wave::default();
    }
}

/// Edge columns, indexed by `Id::idx`.
#[derive(Default, Clone)]
pub struct EdgeStore {
    pub slots: Slots,
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
    pub gain: Vec<S>,
    pub vitality: Vec<S>,
    pub last_t: Vec<u64>,
}

#[derive(Default, Clone)]
pub struct Graph {
    pub cells: CellStore,
    pub edges: EdgeStore,
}

impl Graph {
    pub fn add_cell(&mut self, init: CellInit) -> CellId {
        self.cells.add(init)
    }

    pub fn add_edge(&mut self, src: u32, dst: u32, gain: S, t: u64) -> EdgeId {
        assert!(
            self.cells.slots.is_alive(src) && self.cells.slots.is_alive(dst),
            "edge to dead cell"
        );
        let e = &mut self.edges;
        let (id, fresh) = e.slots.alloc();
        let i = id.idx as usize;
        put(&mut e.src, i, src, fresh);
        put(&mut e.dst, i, dst, fresh);
        put(&mut e.gain, i, gain, fresh);
        put(&mut e.vitality, i, 1.0, fresh);
        put(&mut e.last_t, i, t, fresh);
        self.cells.outs[src as usize].push(id.idx);
        self.cells.ins[dst as usize].push(id.idx);
        id
    }

    /// Removes the edge from both adjacency lists. In-flight messages and credit entries
    /// carry the edge generation and are dropped when they no longer match.
    pub fn remove_edge(&mut self, id: EdgeId) -> bool {
        if !self.edges.slots.release(id) {
            return false;
        }
        let i = id.idx as usize;
        let (s, d) = (self.edges.src[i] as usize, self.edges.dst[i] as usize);
        for list in [&mut self.cells.outs[s], &mut self.cells.ins[d]] {
            if let Some(p) = list.iter().position(|&x| x == id.idx) {
                list.swap_remove(p);
            }
        }
        true
    }

    /// Removes the cell and all incident edges. Pending events carry the generation and are dropped.
    pub fn remove_cell(&mut self, id: CellId) -> bool {
        if !self.cells.slots.is_live(id) {
            return false;
        }
        let c = id.idx as usize;
        let incident: Vec<u32> = self.cells.outs[c]
            .iter()
            .chain(&self.cells.ins[c])
            .copied()
            .collect();
        for e in incident {
            let eid = self.edges.slots.id(e);
            self.remove_edge(eid);
        }
        self.cells.clear_input(c);
        self.cells.slots.release(id)
    }

    pub fn in_gain_norm(&self, c: usize) -> S {
        self.cells.ins[c]
            .iter()
            .map(|&e| self.edges.gain[e as usize].powi(2))
            .sum::<S>()
            .sqrt()
    }

    /// Record the current in-gain norm as each cell's synaptic-scaling target.
    pub fn fix_gain_norms(&mut self, cells: std::ops::Range<usize>) {
        for c in cells {
            if self.cells.slots.is_alive(c as u32) {
                self.cells.gain_norm0[c] = self.in_gain_norm(c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::ZERO_M;

    fn hidden() -> CellInit {
        CellInit {
            kind: Kind::Hidden,
            h: ZERO,
            w: ZERO_M,
            q: ZERO_M,
            log_theta: 0.0,
            rate: 0.0,
            t: 0,
        }
    }

    #[test]
    fn slot_reuse_bumps_generation() {
        let mut s = Slots::default();
        let (a, fresh_a) = s.alloc();
        let (b, _) = s.alloc();
        assert!(fresh_a && a.idx == 0 && b.idx == 1);
        assert!(s.release(a));
        assert!(!s.release(a), "double free must be rejected");
        assert!(!s.is_live(a));
        let (c, fresh_c) = s.alloc();
        assert!(!fresh_c);
        assert_eq!((c.idx, c.generation), (0, 1));
        assert!(
            s.is_live(c) && !s.is_live(a),
            "stale id must not alias the new occupant"
        );
        assert_eq!((s.live(), s.capacity()), (2, 2));
    }

    #[test]
    fn cell_reuse_resets_columns_and_removes_edges() {
        let mut g = Graph::default();
        let a = g.add_cell(hidden());
        let b = g.add_cell(hidden());
        let e = g.add_edge(a.idx, b.idx, 0.7, 0);
        g.cells.heat[a.idx as usize] = 9.0;
        assert!(g.remove_cell(a));
        assert!(!g.edges.slots.is_live(e));
        assert!(g.cells.ins[b.idx as usize].is_empty());
        let a2 = g.add_cell(hidden());
        assert_eq!((a2.idx, a2.generation), (a.idx, a.generation + 1));
        assert_eq!(g.cells.heat[a2.idx as usize], 0.0);
        assert!(g.cells.outs[a2.idx as usize].is_empty());
        let e2 = g.add_edge(b.idx, a2.idx, 0.1, 5);
        assert_eq!((e2.idx, e2.generation), (e.idx, e.generation + 1));
    }

    #[test]
    fn gain_norm_target() {
        let mut g = Graph::default();
        let a = g.add_cell(hidden()).idx;
        let b = g.add_cell(hidden()).idx;
        let c = g.add_cell(hidden()).idx;
        g.add_edge(a, c, 3.0, 0);
        g.add_edge(b, c, 4.0, 0);
        g.fix_gain_norms(0..3);
        assert_eq!(g.cells.gain_norm0[c as usize], 5.0);
        assert_eq!(g.cells.gain_norm0[a as usize], 0.0);
    }
}
