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

mod classes;

pub use classes::ClassMap;
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
    /// Class cell this context belongs to (step 5).
    class: Option<u32>,
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

    /// Count `x` with weight `w` (1 for corpus bytes); visits always count 1.
    fn count(&mut self, x: u8, w: f64) {
        match self.succ.binary_search_by_key(&x, |&(b, _)| b) {
            Ok(i) => self.succ[i].1 += w,
            Err(i) => self.succ.insert(i, (x, w)),
        }
        self.total += w;
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
    /// The last `max(ctx_depth, ctx_chunk_cap)` bytes, oldest first.
    hist: Vec<u8>,
    /// Chunks (hierarchy): length of the previous chunk and of the current partial one.
    chunk: [u32; 2],
    /// Bytes seen (the decay clock).
    now: u64,
    /// Next slot the sleep sweep visits.
    cursor: u32,
    /// Cells grown and pruned since the start; growths refused by the budget.
    pub grown: u64,
    pub pruned: u64,
    pub folded: u64,
    pub starved: u64,
    classes: classes::Classes,
    /// Set by `predict`: entropy of the prediction (bits), visits of the deepest chain cell.
    #[serde(skip)]
    pub entropy: f64,
    #[serde(skip)]
    pub deep_visits: u32,
    #[serde(skip)]
    pub deep_depth: u32,
    /// Weight of each learned byte (1; `lucy talk --kael-weight/--recency` raise it). Not saved.
    #[serde(skip, default = "one")]
    pub weight: f64,
    #[serde(skip)]
    reach_now: usize,
    #[serde(skip)]
    vote_class: Option<u32>,
    #[serde(skip)]
    chain: Vec<u32>,
    #[serde(skip)]
    score: Vec<f64>,
    #[serde(skip)]
    order: Vec<u8>,
}

fn one() -> f64 {
    1.0
}

/// A saved context position (`Context::mark`).
pub struct Mark(Vec<u8>, [u32; 2]);

impl Default for Context {
    fn default() -> Self {
        Self {
            cells: vec![Cell::default()],
            free: Vec::new(),
            live: 1,
            hist: Vec::new(),
            chunk: [0; 2],
            now: 0,
            cursor: 1,
            grown: 0,
            pruned: 0,
            folded: 0,
            starved: 0,
            classes: Default::default(),
            entropy: 0.0,
            deep_visits: 0,
            deep_depth: 0,
            weight: 1.0,
            reach_now: 0,
            vote_class: None,
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
        let guess = self.predict(p, probs);
        self.learn(x, p, probs);
        guess
    }

    /// The next-byte distribution from the current history (no learning). Also sets `entropy`
    /// (bits) and `deep_visits` (visits of the deepest chain cell).
    pub fn predict(&mut self, p: &Params, probs: &mut [S]) -> u8 {
        self.score.resize(N_BYTES, 0.0);
        let t_v = p.ctx_t_v;
        // SPREAD: the chain of existing suffixes of the history, shortest first.
        self.chain.clear();
        self.chain.push(0);
        self.reach_now = self.reach(p);
        let mut c = 0usize;
        for &y in self.hist.iter().rev().take(self.reach_now) {
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
            let w = 2f64.powi(cell.depth.min(p.vote_depth()) as i32);
            Self::vote(&mut self.score, &mut self.order, cell, w);
        }
        // Class back-off: the deepest classed cell's class votes just below that cell.
        self.vote_class = None;
        if p.cls_mode > 0 {
            let found = self.chain.iter().rev().find_map(|&ci| {
                let c = &self.cells[ci as usize];
                c.class.map(|k| (c.depth, k))
            });
            if let Some((d, k)) = found {
                let cl = self.classes.get(k);
                cl.decay(self.now, t_v);
                let w = p.cls_weight * 2f64.powi(d.min(p.vote_depth()) as i32 - 1);
                Self::vote(&mut self.score, &mut self.order, cl, w);
                self.vote_class = Some(k);
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
        let floor = EPS / N_BYTES as f64;
        for (b, q) in probs.iter_mut().enumerate() {
            let share = if z != 0.0 { self.score[b] / z } else { 0.0 };
            *q = (share * (1.0 - EPS) + floor) as S;
        }
        // Entropy over the touched bytes plus the uniform floor of the others: O(touched).
        let h_touched: f64 = self
            .order
            .iter()
            .map(|&b| {
                let q = self.score[b as usize] / z * (1.0 - EPS) + floor;
                -q * q.log2()
            })
            .sum();
        self.entropy = h_touched - (N_BYTES - self.order.len()) as f64 * floor * floor.log2();
        let deep = *self.chain.last().unwrap() as usize;
        self.deep_visits = self.cells[deep].visits;
        self.deep_depth = self.cells[deep].depth;
        guess
    }

    /// Add `cell`'s vote, `w · T/(T+2) / T` per count.
    fn vote(score: &mut [f64], order: &mut Vec<u8>, cell: &Cell, w: f64) {
        let t = cell.total;
        if t <= 0.0 {
            return;
        }
        let w = (t / (t + 2.0)) * w / t;
        for &(b, v) in &cell.succ {
            // w * v > 0, so a zero score means first touch.
            if score[b as usize] == 0.0 {
                order.push(b);
            }
            score[b as usize] += w * v;
        }
    }

    /// Learn `x` after `predict`: every active cell earns; the deepest grows a child if it
    /// mispredicted.
    fn learn(&mut self, x: u8, p: &Params, probs: &[S]) {
        let deep = *self.chain.last().unwrap() as usize;
        let deep_guess = self.cells[deep].best();
        for &ci in &self.chain {
            self.cells[ci as usize].count(x, self.weight);
        }
        if let Some(k) = self.vote_class {
            self.classes.count(k, x, self.weight);
        }
        let d = self.cells[deep].depth as usize;
        if deep_guess != Some(x)
            && d < self.reach_now
            && f64::from(self.cells[deep].visits) >= p.ctx_grow_min
        {
            let y = self.hist[self.hist.len() - 1 - d];
            self.grow(deep, y, x, p);
        }
        self.pass(x, p, probs);
        self.now += 1;
    }

    /// `x` becomes context only: history and chunks move, no cell counts or grows. Lucy's own
    /// output goes through here, never through `learn` (no self-reinforcement).
    pub fn pass(&mut self, x: u8, p: &Params, probs: &[S]) {
        // Chunk boundary: x starts a new chunk after a non-alphanumeric byte (mode 1), or when
        // it was surprising, a tension peak (mode 2).
        let boundary = match p.ctx_chunk {
            1 => {
                x.is_ascii_alphanumeric()
                    && self.hist.last().is_some_and(|b| !b.is_ascii_alphanumeric())
            }
            2 => -f64::from(probs[x as usize]).log2() > p.ctx_chunk_bits,
            _ => false,
        };
        self.chunk = match boundary {
            true => [self.chunk[1], 1],
            false => [self.chunk[0], self.chunk[1].saturating_add(1)],
        };
        self.hist.push(x);
        let keep = match p.ctx_chunk {
            0 => p.ctx_depth,
            _ => p.ctx_depth.max(p.ctx_chunk_cap),
        };
        if self.hist.len() > keep as usize {
            self.hist.remove(0);
        }
    }

    /// The newest history byte.
    pub fn last(&self) -> Option<u8> {
        self.hist.last().copied()
    }

    /// Continue the text: sample each byte from the blended prediction at `temperature`
    /// (0 = top-1), up to `max` bytes or through `stop`. Every byte becomes context only
    /// (`pass`), never evidence. `out` gets each byte with the entropy (bits) it was drawn from.
    pub fn generate(
        &mut self,
        p: &Params,
        temperature: f64,
        max: usize,
        stop: u8,
        rng: &mut impl rand::Rng,
        mut out: impl FnMut(u8, f64),
    ) {
        let mut probs = vec![0.0; N_BYTES];
        for _ in 0..max {
            let x = self.sample(p, temperature, rng, &mut probs);
            out(x, self.entropy);
            self.pass(x, p, &probs);
            if x == stop {
                break;
            }
        }
    }

    /// Draw the next byte from the blended prediction at `temperature` (0 = top-1). Fills
    /// `probs` and `entropy`; learns nothing and does not move the history (`pass` does).
    pub fn sample(
        &mut self,
        p: &Params,
        temperature: f64,
        rng: &mut impl rand::Rng,
        probs: &mut [S],
    ) -> u8 {
        let top = self.predict(p, probs);
        if temperature <= 0.0 {
            return top;
        }
        let w: Vec<f64> = probs
            .iter()
            .map(|&q| f64::from(q).powf(1.0 / temperature))
            .collect();
        let mut r = rng.random::<f64>() * w.iter().sum::<f64>();
        w.iter()
            .position(|&v| {
                r -= v;
                r < 0.0
            })
            .unwrap_or(N_BYTES - 1) as u8
    }

    /// The context position (history and chunk state), to undo `pass`es with `rewind`.
    pub fn mark(&self) -> Mark {
        Mark(self.hist.clone(), self.chunk)
    }

    pub fn rewind(&mut self, m: Mark) {
        (self.hist, self.chunk) = (m.0, m.1);
    }

    /// How far back the chain may reach: K bytes, or with chunks on, back to the start of the
    /// previous chunk (capped). Deeper contexts exist only where they are chunk-anchored.
    fn reach(&self, p: &Params) -> usize {
        let k = p.ctx_depth as usize;
        let r = match p.ctx_chunk {
            0 => k,
            _ => k.max((self.chunk[0] + self.chunk[1]).min(p.ctx_chunk_cap) as usize),
        };
        r.min(self.hist.len())
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
        cell.count(x, self.weight);
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
            if self.cells[c].depth == DEAD {
                continue;
            }
            if p.cls_mode > 0 {
                self.classify(c, p);
            }
            let cell = &self.cells[c];
            if !cell.children.is_empty() {
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
            self.leave(c);
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

    /// A full sleep: enough passes to visit every slot once.
    pub fn sleep_all(&mut self, p: &Params) -> u64 {
        let passes = self.cells.len().div_ceil(p.ctx_sweep.max(1) as usize);
        (0..passes).map(|_| self.sleep(p)).sum()
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
            h.u64(c.class.map_or(u64::MAX, u64::from));
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
        h.u64(((self.chunk[0] as u64) << 32) | self.chunk[1] as u64);
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
    fn chunks_let_the_chain_reach_back_to_the_previous_chunk() {
        let base = Params {
            ctx_depth: 2,
            ..Params::default()
        };
        let mut ctx = Context::default();
        feed(&mut ctx, &base, b"kitchen table");
        assert_eq!(ctx.reach(&base), 2);
        let p = Params {
            ctx_chunk: 1,
            ..base
        };
        let mut ctx = Context::default();
        feed(&mut ctx, &p, b"kitchen table");
        // Previous chunk "kitchen " (8) + partial "table" (5).
        assert_eq!(ctx.chunk, [8, 5]);
        assert_eq!(ctx.reach(&p), 13);
        assert_eq!(
            ctx.reach(&Params {
                ctx_chunk_cap: 6,
                ..p.clone()
            }),
            6
        );
    }

    /// Lucy's own output is context, never evidence: no count, visit or cell is added.
    #[test]
    fn generation_adds_no_evidence() {
        let p = Params::default();
        let mut ctx = Context::default();
        feed(
            &mut ctx,
            &p,
            &b"*MOT: where is the ball ? *CHI: there .\n".repeat(30),
        );
        let evidence = |c: &Context| {
            let v: u64 = c.cells.iter().map(|x| u64::from(x.visits)).sum();
            let n: usize = c.cells.iter().map(|x| x.succ.len()).sum();
            (c.cells(), v, n, c.now)
        };
        let before = evidence(&ctx);
        let mut text = Vec::new();
        let mut rng = crate::seed::stream(1, 2);
        ctx.generate(&p, 0.0, 100, b'\n', &mut rng, |x, _| text.push(x));
        assert_eq!(evidence(&ctx), before);
        assert_eq!(text, b"*MOT: where is the ball ? *CHI: there .\n");
        ctx.generate(&p, 1.0, 50, b'\n', &mut rng, |_, h| assert!(h.is_finite()));
        assert_eq!(evidence(&ctx), before);
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
