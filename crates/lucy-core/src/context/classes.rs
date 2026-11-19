//! Class cells (step 5, docs/DESIGN.md): contexts whose successor distributions are alike share a
//! class that pools evidence. Assignment happens only in the sleep sweep (structural); a class
//! counts bytes online, when it stands in for the deepest classed cell of the chain.
//!
//! Modes (`cls_mode`): 1 = similarity (LSH bucket on the top-2 successors, then the nearest class
//! by Hellinger distance within `cls_radius`), 2 = random control (fixed hash of the cell id into
//! `cls_random_classes` classes), 3 = oracle (a fixed map from context bytes to classes, built
//! offline by `batch_classes`).

use std::collections::BTreeMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{Cell, Context, DEAD};
use crate::rules::Params;

pub type ClassMap = BTreeMap<Vec<u8>, u32>;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Classes {
    /// Pooled counts per class (`depth == DEAD` marks a free slot).
    cells: Vec<Cell>,
    members: Vec<u32>,
    keys: Vec<u16>,
    free: Vec<u32>,
    /// LSH: sorted top-2 successors -> classes founded with that key (similarity mode).
    buckets: BTreeMap<u16, Vec<u32>>,
    pub live: usize,
    /// Oracle mode: context (newest byte first) -> class.
    #[serde(skip)]
    pub oracle: Option<ClassMap>,
}

/// Sorted pair of the two most frequent successors (ties: lowest byte).
fn key(c: &Cell) -> u16 {
    let mut top = [(0u8, f64::NEG_INFINITY); 2];
    for &(b, v) in &c.succ {
        if v > top[0].1 {
            top = [(b, v), top[0]];
        } else if v > top[1].1 {
            top[1] = (b, v);
        }
    }
    let (a, b) = (
        top[0].0,
        if top[1].1.is_finite() {
            top[1].0
        } else {
            top[0].0
        },
    );
    ((a.min(b) as u16) << 8) | a.max(b) as u16
}

/// Hellinger distance between two successor distributions (sparse, sorted by byte).
fn hellinger(a: &Cell, b: &Cell) -> f64 {
    if a.total <= 0.0 || b.total <= 0.0 {
        return 1.0;
    }
    let (mut i, mut j, mut bc) = (0, 0, 0.0);
    while i < a.succ.len() && j < b.succ.len() {
        let ((x, u), (y, v)) = (a.succ[i], b.succ[j]);
        match x.cmp(&y) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                bc += (u / a.total * v / b.total).sqrt();
                (i, j) = (i + 1, j + 1);
            }
        }
    }
    (1.0 - bc).max(0.0).sqrt()
}

/// Fixed scramble of a cell id (random control).
fn mix(x: u64) -> u64 {
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

impl Classes {
    pub fn count(&mut self, k: u32, x: u8) {
        self.cells[k as usize].count(x);
    }

    pub(super) fn get(&mut self, k: u32) -> &mut Cell {
        &mut self.cells[k as usize]
    }

    /// A class slot seeded with the founder's counts (they are the same observations, pooled).
    fn found(&mut self, id: Option<u32>, seed: &Cell, key: u16) -> u32 {
        let k = match id {
            Some(k) => k,
            None => self.free.pop().unwrap_or(self.cells.len() as u32),
        };
        let i = k as usize;
        if i >= self.cells.len() {
            self.cells.resize(
                i + 1,
                Cell {
                    depth: DEAD,
                    ..Cell::default()
                },
            );
            self.members.resize(i + 1, 0);
            self.keys.resize(i + 1, 0);
        }
        self.cells[i] = Cell {
            depth: 0,
            last_t: seed.last_t,
            total: seed.total,
            visits: seed.visits,
            succ: seed.succ.clone(),
            ..Cell::default()
        };
        self.keys[i] = key;
        self.live += 1;
        k
    }

    fn alive(&self, k: u32) -> bool {
        self.cells.get(k as usize).is_some_and(|c| c.depth != DEAD)
    }
}

impl Context {
    /// Live classes.
    pub fn classes(&self) -> usize {
        self.classes.live
    }

    pub fn set_oracle(&mut self, map: ClassMap) {
        self.classes.oracle = Some(map);
    }

    fn eligible(&self, c: usize, p: &Params) -> bool {
        let cell = &self.cells[c];
        cell.depth >= 1 && cell.depth <= p.ctx_depth && cell.visits >= p.cls_min_visits
    }

    /// Sleep-time assignment of one visited cell.
    pub(super) fn classify(&mut self, c: usize, p: &Params) {
        if !self.eligible(c, p) {
            return;
        }
        let cl = &mut self.classes;
        let cell = &self.cells[c];
        match p.cls_mode {
            1 => {
                if let Some(k) = cell.class {
                    if hellinger(cell, &cl.cells[k as usize]) <= p.cls_radius {
                        return;
                    }
                    self.leave(c);
                }
                let (cell, cl) = (&self.cells[c], &mut self.classes);
                let key = key(cell);
                let best = cl.buckets.get(&key).and_then(|ks| {
                    ks.iter()
                        .map(|&k| (hellinger(cell, &cl.cells[k as usize]), k))
                        .filter(|&(d, _)| d <= p.cls_radius)
                        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                        .map(|(_, k)| k)
                });
                let k = match best {
                    Some(k) => k,
                    None if cl.live < p.cls_budget as usize => {
                        let k = cl.found(None, cell, key);
                        cl.buckets.entry(key).or_default().push(k);
                        k
                    }
                    None => return,
                };
                self.join(c, k);
            }
            2 | 3 if cell.class.is_none() => {
                let id = match p.cls_mode {
                    2 => (mix(c as u64) % u64::from(p.cls_random_classes.max(1))) as u32,
                    _ => {
                        let ctx = self.context_of(c);
                        match self.classes.oracle.as_ref().and_then(|m| m.get(&ctx)) {
                            Some(&k) => k,
                            None => return,
                        }
                    }
                };
                if !self.classes.alive(id) {
                    let cell = &self.cells[c];
                    self.classes.found(Some(id), cell, key(cell));
                }
                self.join(c, id);
            }
            _ => {}
        }
    }

    fn join(&mut self, c: usize, k: u32) {
        self.cells[c].class = Some(k);
        self.classes.members[k as usize] += 1;
    }

    /// Drop a cell's class membership; a class without members dies.
    pub(super) fn leave(&mut self, c: usize) {
        let Some(k) = self.cells[c].class.take() else {
            return;
        };
        let cl = &mut self.classes;
        let i = k as usize;
        cl.members[i] -= 1;
        if cl.members[i] > 0 {
            return;
        }
        if let Some(ks) = cl.buckets.get_mut(&cl.keys[i]) {
            ks.retain(|&x| x != k);
            if ks.is_empty() {
                cl.buckets.remove(&cl.keys[i]);
            }
        }
        cl.cells[i] = Cell {
            depth: DEAD,
            ..Cell::default()
        };
        if cl.oracle.is_none() {
            cl.free.push(k);
        }
        cl.live -= 1;
    }

    /// The context of cell `c`, newest byte first (the path of child keys from the root).
    pub fn context_of(&self, mut c: usize) -> Vec<u8> {
        let mut out = Vec::new();
        while c != 0 {
            let parent = self.cells[c].parent as usize;
            let &(b, _) = self.cells[parent]
                .children
                .iter()
                .find(|&&(_, ch)| ch as usize == c)
                .expect("child link");
            out.push(b);
            c = parent;
        }
        out.reverse();
        out
    }

    /// Offline upper bound: cluster every eligible context of this tissue into `k` classes with
    /// hindsight counts, Brown-style: exchange moves that maximize the class-conditional
    /// log-likelihood of successors, `Σ_class Σ_x N(class, x) ln(N(class, x) / N(class))`.
    /// Seeds: the k most visited contexts; first assignment by Hellinger distance to the seeds.
    pub fn batch_classes(&self, p: &Params, k: usize, sweeps: usize) -> ClassMap {
        let items: Vec<usize> = (1..self.cells.len())
            .filter(|&c| self.cells[c].depth != DEAD && self.eligible(c, p))
            .collect();
        let k = k.min(items.len()).max(1);
        if items.is_empty() {
            return ClassMap::new();
        }
        let mut by_visits = items.clone();
        by_visits.sort_by_key(|&c| (std::cmp::Reverse(self.cells[c].visits), c));
        let seeds = &by_visits[..k];
        let mut class: Vec<usize> = items
            .par_iter()
            .map(|&c| {
                let d = |s: &usize| hellinger(&self.cells[c], &self.cells[*s]);
                (0..k)
                    .min_by(|&a, &b| d(&seeds[a]).total_cmp(&d(&seeds[b])).then(a.cmp(&b)))
                    .unwrap()
            })
            .collect();
        let xlnx = |x: f64| if x > 0.0 { x * x.ln() } else { 0.0 };
        let mut n = vec![[0.0f64; 256]; k];
        let mut t = vec![0.0f64; k];
        let add = |n: &mut [[f64; 256]], t: &mut [f64], j: usize, c: &Cell, s: f64| {
            for &(b, v) in &c.succ {
                n[j][b as usize] += s * v;
            }
            t[j] += s * c.total;
        };
        for (i, &c) in items.iter().enumerate() {
            add(&mut n, &mut t, class[i], &self.cells[c], 1.0);
        }
        for _ in 0..sweeps {
            let mut moved = 0;
            for (i, &c) in items.iter().enumerate() {
                let cell = &self.cells[c];
                add(&mut n, &mut t, class[i], cell, -1.0);
                // Gain of adding the item to class j (the parts of the objective it changes).
                let gain = |j: usize| -> f64 {
                    cell.succ
                        .iter()
                        .map(|&(b, v)| xlnx(n[j][b as usize] + v) - xlnx(n[j][b as usize]))
                        .sum::<f64>()
                        - (xlnx(t[j] + cell.total) - xlnx(t[j]))
                };
                // Pure per class, total order with index tie-break: same result on any thread count.
                let best = (0..k)
                    .into_par_iter()
                    .map(|j| (gain(j), j))
                    .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)))
                    .unwrap()
                    .1;
                moved += usize::from(best != class[i]);
                class[i] = best;
                add(&mut n, &mut t, best, cell, 1.0);
            }
            if moved == 0 {
                break;
            }
        }
        items
            .iter()
            .zip(class)
            .map(|(&c, j)| (self.context_of(c), j as u32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(succ: &[(u8, f64)]) -> Cell {
        Cell {
            total: succ.iter().map(|x| x.1).sum(),
            succ: succ.to_vec(),
            ..Cell::default()
        }
    }

    #[test]
    fn key_and_distance() {
        let a = cell(&[(b'a', 6.0), (b'b', 3.0), (b'c', 1.0)]);
        assert_eq!(key(&a), (b'a' as u16) << 8 | b'b' as u16);
        assert_eq!(key(&cell(&[(b'z', 1.0)])), (b'z' as u16) << 8 | b'z' as u16);
        assert!(hellinger(&a, &a) < 1e-6);
        assert!((hellinger(&a, &cell(&[(b'q', 1.0)])) - 1.0).abs() < 1e-12);
    }

    /// Contexts followed by the same bytes end up in one class; a different one does not.
    #[test]
    fn exchange_groups_alike_successors() {
        let p = Params {
            cls_min_visits: 1,
            ..Params::default()
        };
        let mut ctx = Context::default();
        for (y, succ) in [
            (b'x', vec![(b'a', 9.0), (b'b', 1.0)]),
            (b'y', vec![(b'a', 8.0), (b'b', 2.0)]),
            (b'z', vec![(b'q', 10.0)]),
        ] {
            let mut c = cell(&succ);
            (c.depth, c.visits) = (1, 10);
            ctx.cells.push(c);
            let id = ctx.cells.len() as u32 - 1;
            ctx.cells[0].children.push((y, id));
        }
        let m = ctx.batch_classes(&p, 2, 5);
        assert_eq!(m[&b"x".to_vec()], m[&b"y".to_vec()]);
        assert_ne!(m[&b"x".to_vec()], m[&b"z".to_vec()]);
    }
}
