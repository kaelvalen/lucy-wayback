//! Context cells (docs/PROPOSAL-v2.md): a cell's identity is its context, a suffix of the byte
//! history. Cells form a suffix trie: the root is the empty context, and the child of a cell for
//! byte y is the same context extended one byte further into the past.
//!
//! - SPREAD: at each byte the active chain is walked from the root through the children, keyed by
//!   the newest byte, then the next older one, and so on. It holds every cell whose context is a
//!   suffix of the last `ctx_depth` bytes: ≤ K + 1 cells, neighbour to neighbour, O(K) per byte
//!   whatever the cell count.
//! - TRACE: each cell counts what followed it; the prediction is the confidence-weighted vote
//!   of the chain, `w = T/(T+2) · 2^depth / T`.
//! - TENSION/GROWTH: when the deepest active cell mispredicts and has ≥ `ctx_grow_min` evidence,
//!   it grows one child one byte deeper. Cells only grow where Lucy is wrong, and only while the
//!   tissue is under its cell budget (metabolism).
//! - DECAY: counts are earned vitality and decay lazily (`exp(-dt / ctx_t_v)`, dt in bytes, applied
//!   when a cell is touched). Sleep sweeps a fixed number of cells per pass and prunes leaves whose
//!   vitality fell below `ctx_prune_floor`, so pruning costs O(1) per byte too.
//!
//! With `ctx_t_v = 0` (no decay) this is `scripts/context_cells_reference.py` up to tie-breaks:
//! successor and child lists are sorted by byte, and ties go to the lowest byte.

use serde::{Deserialize, Serialize};

use crate::num::S;
use crate::rules::Params;
use crate::senses::N_BYTES;

/// Byte guessed when nothing has been learned yet (the reference's default).
const COLD_GUESS: u8 = b' ';
/// Share of probability mass spread uniformly, so no byte ever costs infinite bits.
const EPS: f64 = 1e-3;
/// Marks a free slot.
const DEAD: u32 = u32::MAX;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Cell {
    /// Shorter context (one byte less in the past); unused for the root.
    parent: u32,
    depth: u32,
    /// Byte count at the last decay.
    last_t: u64,
    /// Sum of `succ` counts.
    total: f64,
    /// Times this context was seen (undecayed evidence: the growth gate).
    visits: u32,
    /// Byte one step further into the past -> child cell, sorted by byte.
    children: Vec<(u8, u32)>,
    /// What followed this context and how often (decayed), sorted by byte.
    succ: Vec<(u8, f64)>,
}

impl Cell {
    /// Most frequent successor; the lowest byte wins ties.
    fn best(&self) -> Option<u8> {
        let mut best: Option<(u8, f64)> = None;
        for &(b, v) in &self.succ {
            if best.is_none_or(|(_, bv)| v > bv) {
                best = Some((b, v));
            }
        }
        best.map(|(b, _)| b)
    }

    fn child(&self, y: u8) -> Option<u32> {
        self.children
            .binary_search_by_key(&y, |&(b, _)| b)
            .ok()
            .map(|i| self.children[i].1)
    }

    fn count(&mut self, x: u8) {
        match self.succ.binary_search_by_key(&x, |&(b, _)| b) {
            Ok(i) => self.succ[i].1 += 1.0,
            Err(i) => self.succ.insert(i, (x, 1.0)),
        }
        self.total += 1.0;
        self.visits = self.visits.saturating_add(1);
    }

    /// Bring the counts to byte time `now`.
    fn decay(&mut self, now: u64, t_v: f64) {
        let dt = now - self.last_t;
        self.last_t = now;
        if t_v > 0.0 && dt > 0 {
            let f = (-(dt as f64) / t_v).exp();
            for (_, v) in &mut self.succ {
                *v *= f;
            }
            self.total *= f;
        }
    }

    /// Vitality at `now` without touching the cell.
    fn vitality(&self, now: u64, t_v: f64) -> f64 {
        match t_v > 0.0 {
            true => self.total * (-((now - self.last_t) as f64) / t_v).exp(),
            false => self.total,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Context {
    cells: Vec<Cell>,
    free: Vec<u32>,
    live: usize,
    /// The last `ctx_depth` bytes, oldest first.
    hist: Vec<u8>,
    /// Bytes seen (the decay clock).
    now: u64,
    /// Next slot the sleep sweep visits.
    cursor: u32,
    /// Cells grown and pruned since the start; growths refused by the budget.
    pub grown: u64,
    pub pruned: u64,
    pub folded: u64,
    pub starved: u64,
    #[serde(skip)]
    chain: Vec<u32>,
    #[serde(skip)]
    score: Vec<f64>,
    #[serde(skip)]
    order: Vec<u8>,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            cells: vec![Cell::default()],
            free: Vec::new(),
            live: 1,
            hist: Vec::new(),
            now: 0,
            cursor: 1,
            grown: 0,
            pruned: 0,
            folded: 0,
            starved: 0,
            chain: Vec::new(),
            score: vec![0.0; N_BYTES],
            order: Vec::new(),
        }
    }
}

impl Context {
    /// Live cells, root included.
    pub fn cells(&self) -> usize {
        self.live
    }

    /// Byte `x` arrives: fill `probs` with the prediction made before seeing it, learn, grow.
    /// Returns the top-1 guess.
    pub fn arrive(&mut self, x: u8, p: &Params, probs: &mut [S]) -> u8 {
        self.score.resize(N_BYTES, 0.0);
        let t_v = p.ctx_t_v;
        // SPREAD: the chain of existing suffixes of the history, shortest first.
        self.chain.clear();
        self.chain.push(0);
        let mut c = 0usize;
        for &y in self.hist.iter().rev() {
            match self.cells[c].child(y) {
                Some(ch) => {
                    self.chain.push(ch);
                    c = ch as usize;
                }
                None => break,
            }
        }
        // TRACE: confidence-weighted vote of the (decayed) chain.
        for &b in &self.order {
            self.score[b as usize] = 0.0;
        }
        self.order.clear();
        for &ci in &self.chain {
            let cell = &mut self.cells[ci as usize];
            cell.decay(self.now, t_v);
            let t = cell.total;
            if t <= 0.0 {
                continue;
            }
            let w = (t / (t + 2.0)) * 2f64.powi(cell.depth as i32) / t;
            for &(b, v) in &cell.succ {
                // w * v > 0, so a zero score means first touch.
                if self.score[b as usize] == 0.0 {
                    self.order.push(b);
                }
                self.score[b as usize] += w * v;
            }
        }
        let mut guess = COLD_GUESS;
        let mut z = 0.0;
        let mut top = f64::NEG_INFINITY;
        for &b in &self.order {
            let s = self.score[b as usize];
            if s > top || (s == top && b < guess) {
                (top, guess) = (s, b);
            }
            z += s;
        }
        for (b, q) in probs.iter_mut().enumerate() {
            let share = if z != 0.0 { self.score[b] / z } else { 0.0 };
            *q = (share * (1.0 - EPS) + EPS / N_BYTES as f64) as S;
        }
        // Learn: every active cell earns; the deepest grows a child if it mispredicted.
        let deep = *self.chain.last().unwrap() as usize;
        let deep_guess = self.cells[deep].best();
        for &ci in &self.chain {
            self.cells[ci as usize].count(x);
        }
        let d = self.cells[deep].depth as usize;
        if deep_guess != Some(x)
            && d < self.hist.len()
            && f64::from(self.cells[deep].visits) >= p.ctx_grow_min
        {
            let y = self.hist[self.hist.len() - 1 - d];
            self.grow(deep, y, x, p);
        }
        self.hist.push(x);
        if self.hist.len() > p.ctx_depth as usize {
            self.hist.remove(0);
        }
        self.now += 1;
        guess
    }

    fn grow(&mut self, parent: usize, y: u8, x: u8, p: &Params) {
        if p.ctx_budget > 0 && self.live >= p.ctx_budget as usize {
            self.starved += 1;
            return;
        }
        let mut cell = Cell {
            parent: parent as u32,
            depth: self.cells[parent].depth + 1,
            last_t: self.now,
            ..Cell::default()
        };
        cell.count(x);
        let id = match self.free.pop() {
            Some(id) => {
                self.cells[id as usize] = cell;
                id
            }
            None => {
                let id = u32::try_from(self.cells.len()).expect("context cells exceed u32 ids");
                self.cells.push(cell);
                id
            }
        };
        let ch = &mut self.cells[parent].children;
        let i = ch.binary_search_by_key(&y, |&(b, _)| b).unwrap_err();
        ch.insert(i, (y, id));
        self.live += 1;
        self.grown += 1;
    }

    /// Sleep (structural only, it never adds evidence): visit the next `ctx_sweep` slots; a leaf
    /// dies if its vitality is below the floor, or folds into its parent if it saves fewer than
    /// `ctx_fold_bits` (a non-leaf waits until its children are gone). Returns cells removed.
    pub fn sleep(&mut self, p: &Params) -> u64 {
        let n = self.cells.len() as u32;
        if n <= 1 {
            return 0;
        }
        let mut removed = 0;
        for _ in 0..p.ctx_sweep.min(n - 1) {
            let c = self.cursor as usize;
            self.cursor = if self.cursor + 1 >= n {
                1
            } else {
                self.cursor + 1
            };
            let cell = &self.cells[c];
            if cell.depth == DEAD || !cell.children.is_empty() {
                continue;
            }
            let vitality = cell.vitality(self.now, p.ctx_t_v);
            if vitality < p.ctx_prune_floor {
                self.pruned += 1;
            } else if p.ctx_fold_bits > 0.0 && self.fold_gain(c, vitality) < p.ctx_fold_bits {
                self.folded += 1;
            } else {
                continue;
            }
            let parent = cell.parent as usize;
            let ch = &mut self.cells[parent].children;
            ch.retain(|&(_, id)| id as usize != c);
            self.cells[c] = Cell {
                depth: DEAD,
                ..Cell::default()
            };
            self.free.push(c as u32);
            self.live -= 1;
            removed += 1;
        }
        removed
    }

    /// Bits a leaf saves over its parent on the evidence it holds (MDL): vitality times
    /// KL(p_cell || p_parent). Its counts are already in the parent (every chain cell counts
    /// every byte), so folding it loses only this. Ratios n/T do not depend on decay.
    fn fold_gain(&self, c: usize, vitality: f64) -> f64 {
        let (cell, parent) = (&self.cells[c], &self.cells[self.cells[c].parent as usize]);
        let kl: f64 = cell
            .succ
            .iter()
            .filter(|&&(_, v)| v > 0.0)
            .map(|&(b, v)| {
                // The parent holds every count its child holds; if not, never fold.
                let q = parent
                    .succ
                    .binary_search_by_key(&b, |&(x, _)| x)
                    .map_or(0.0, |i| parent.succ[i].1);
                let pc = v / cell.total;
                if q > 0.0 {
                    pc * (pc / (q / parent.total)).log2()
                } else {
                    f64::INFINITY
                }
            })
            .sum();
        vitality * kl
    }

    /// Feed the state hash (determinism check).
    pub fn feed_hash(&self, h: &mut crate::observe::Fnv) {
        h.u64(self.cells.len() as u64);
        for c in &self.cells {
            h.u64(((c.parent as u64) << 32) | c.depth as u64);
            h.u64(c.last_t);
            h.u64(c.total.to_bits());
            h.u64(c.visits as u64);
            for &(b, ch) in &c.children {
                h.u64(((b as u64) << 32) | ch as u64);
            }
            for &(b, v) in &c.succ {
                h.u64(b as u64);
                h.u64(v.to_bits());
            }
        }
        for &f in &self.free {
            h.u64(f as u64);
        }
        h.bytes(&self.hist);
        h.u64(self.now);
        h.u64(self.cursor as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(ctx: &mut Context, p: &Params, s: &[u8]) -> usize {
        let mut probs = vec![0.0; N_BYTES];
        s.iter()
            .filter(|&&x| ctx.arrive(x, p, &mut probs) == x)
            .count()
    }

    #[test]
    fn learns_a_cycle_and_grows_only_on_misses() {
        let p = Params::default();
        let mut ctx = Context::default();
        let mut probs = vec![0.0; N_BYTES];
        let stream = b"abcabcabcabcabcabcabcabc";
        let mut right = 0;
        for (i, &x) in stream.iter().enumerate() {
            let g = ctx.arrive(x, &p, &mut probs);
            right += usize::from(g == x);
            // Before anything is learned the reference gives every byte EPS / 256 (no renormalization).
            let total: f64 = probs.iter().map(|&q| q as f64).sum();
            let want = if i == 0 { EPS } else { 1.0 };
            assert!((total - want).abs() < 1e-4, "byte {i}: probs sum {total}");
        }
        // After one cycle, order-1 contexts predict perfectly; growth stops when nothing misses.
        assert!(right >= stream.len() - 5, "right {right}");
        let cells = ctx.cells();
        assert_eq!(feed(&mut ctx, &p, b"abcabc"), 6);
        assert_eq!(ctx.cells(), cells);
    }

    #[test]
    fn unvisited_contexts_die_and_slots_are_reused() {
        let p = Params {
            ctx_t_v: 50.0,
            ctx_sweep: 1 << 20,
            ..Params::default()
        };
        let mut ctx = Context::default();
        feed(&mut ctx, &p, &b"xyzw".repeat(20));
        let grown = ctx.grown;
        assert!(grown > 0);
        // A long stretch of another pattern: the old contexts fade below the floor.
        feed(&mut ctx, &p, &b"ab".repeat(300));
        ctx.sleep(&p);
        assert!(ctx.pruned > 0);
        assert_eq!(ctx.live, 1 + ctx.grown as usize - ctx.pruned as usize);
        // Every child link points at a live cell whose parent is the linking cell.
        for (i, c) in ctx.cells.iter().enumerate() {
            for &(_, ch) in &c.children {
                assert_ne!(ctx.cells[ch as usize].depth, DEAD);
                assert_eq!(ctx.cells[ch as usize].parent as usize, i);
            }
        }
        let slots = ctx.cells.len();
        feed(&mut ctx, &p, &b"xyzw".repeat(20));
        assert!(ctx.cells.len() <= slots, "freed slots are reused first");
    }

    #[test]
    fn fold_removes_only_leaves_that_add_nothing() {
        let leaf = |parent, succ: Vec<(u8, f64)>| Cell {
            parent,
            depth: 1,
            total: succ.iter().map(|x| x.1).sum(),
            succ,
            ..Cell::default()
        };
        let mut ctx = Context::default();
        ctx.cells[0] = leaf(0, vec![(b'a', 10.0), (b'b', 10.0)]);
        ctx.cells[0].depth = 0;
        ctx.cells[0].children = vec![(b'x', 1), (b'y', 2)];
        ctx.cells.push(leaf(0, vec![(b'a', 5.0), (b'b', 5.0)])); // same as the parent: KL 0
        ctx.cells.push(leaf(0, vec![(b'a', 5.0)])); // 5 x log2(1 / 0.5) = 5 bits
        ctx.live = 3;
        let p = Params {
            ctx_fold_bits: 1.0,
            ctx_t_v: 0.0,
            ..Params::default()
        };
        assert_eq!(ctx.sleep(&p), 1);
        assert_eq!(ctx.folded, 1);
        assert_eq!(ctx.cells[0].children, vec![(b'y', 2)]);
        assert_eq!(ctx.cells[1].depth, DEAD);
    }

    #[test]
    fn budget_stops_growth() {
        let p = Params {
            ctx_budget: 5,
            ..Params::default()
        };
        let mut ctx = Context::default();
        feed(
            &mut ctx,
            &p,
            b"the cat sat on the mat and the dog sat on the log",
        );
        assert_eq!(ctx.cells(), 5);
        assert!(ctx.starved > 0);
    }
}
