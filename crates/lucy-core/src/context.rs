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
//!   it grows one child one byte deeper. Cells only grow where Lucy is wrong.
//!
//! This is an exact port of `scripts/context_cells_reference.py`: successor lists keep first-seen
//! order, so ties break exactly as Python's dict / `Counter.most_common` do.

use serde::{Deserialize, Serialize};

use crate::num::S;
use crate::rules::Params;
use crate::senses::N_BYTES;

/// Byte guessed when nothing has been learned yet (the reference's default).
const COLD_GUESS: u8 = b' ';
/// Share of probability mass spread uniformly, so no byte ever costs infinite bits.
const EPS: f64 = 1e-3;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Cell {
    depth: u32,
    /// Byte one step further into the past -> child cell.
    children: Vec<(u8, u32)>,
    /// What followed this context (first-seen order) and how often.
    succ: Vec<(u8, f64)>,
    total: f64,
}

impl Cell {
    /// Most frequent successor; the first seen wins ties (`Counter.most_common(1)`).
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
            .iter()
            .find(|&&(b, _)| b == y)
            .map(|&(_, c)| c)
    }

    fn count(&mut self, x: u8) {
        match self.succ.iter_mut().find(|(b, _)| *b == x) {
            Some((_, v)) => *v += 1.0,
            None => self.succ.push((x, 1.0)),
        }
        self.total += 1.0;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Context {
    cells: Vec<Cell>,
    /// The last `ctx_depth` bytes, oldest first.
    hist: Vec<u8>,
    /// Cells grown since the start.
    pub grown: u64,
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
            hist: Vec::new(),
            grown: 0,
            chain: Vec::new(),
            score: vec![0.0; N_BYTES],
            order: Vec::new(),
        }
    }
}

impl Context {
    pub fn cells(&self) -> usize {
        self.cells.len()
    }

    /// Byte `x` arrives: fill `probs` with the prediction made before seeing it, learn, grow.
    /// Returns the top-1 guess.
    pub fn arrive(&mut self, x: u8, p: &Params, probs: &mut [S]) -> u8 {
        self.score.resize(N_BYTES, 0.0);
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
        // TRACE: confidence-weighted vote, accumulated in first-touch order.
        for &b in &self.order {
            self.score[b as usize] = 0.0;
        }
        self.order.clear();
        for &ci in &self.chain {
            let cell = &self.cells[ci as usize];
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
            if s > top {
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
        if deep_guess != Some(x) && d < self.hist.len() && self.cells[deep].total >= p.ctx_grow_min
        {
            let y = self.hist[self.hist.len() - 1 - d];
            let id = u32::try_from(self.cells.len()).expect("context cells exceed u32 ids");
            let mut child = Cell {
                depth: d as u32 + 1,
                ..Cell::default()
            };
            child.count(x);
            self.cells.push(child);
            self.cells[deep].children.push((y, id));
            self.grown += 1;
        }
        self.hist.push(x);
        if self.hist.len() > p.ctx_depth as usize {
            self.hist.remove(0);
        }
        guess
    }

    /// Feed the state hash (determinism check).
    pub fn hash(&self, h: &mut crate::observe::Fnv) {
        h.u64(self.cells.len() as u64);
        for c in &self.cells {
            h.u64(c.depth as u64);
            for &(b, ch) in &c.children {
                h.u64(((b as u64) << 32) | ch as u64);
            }
            for &(b, v) in &c.succ {
                h.u64(b as u64);
                h.u64(v.to_bits());
            }
        }
        h.bytes(&self.hist);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference script's numbers on a hand-checkable stream.
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
        for &x in b"abcabc" {
            assert_eq!(ctx.arrive(x, &p, &mut probs), x);
        }
        assert_eq!(ctx.cells(), cells);
    }
}
