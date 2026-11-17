//! Byte-level sensory surface. 256 input cells (cell idx == byte value) both sense and predict:
//! between arrivals they integrate hidden messages sub-threshold (TRACE); each cell reads its
//! trace through its own Q into a score for "my byte comes next", plus a learned prior. An
//! arriving byte forces its cell to fire, and reality teaches every cell that was hot.

use rand::Rng;

use crate::num::{S, V};
use crate::rules::{self, Params};
use crate::seed::{STREAM_CODES, stream};

pub const N_BYTES: usize = 256;

pub struct Senses {
    /// Seed-derived code per byte, entries in {-1, +1} (inside the tanh range of cell states).
    pub codes: Vec<V>,
    pub ticks_per_byte: u64,
    /// Per-input-cell prior expectation (score bias), valid as of byte number `bias_seen`.
    pub bias: Vec<S>,
    pub bias_seen: Vec<u64>,
    /// Reused readout buffer: one score per input cell.
    pub scores: Vec<S>,
    hot: Vec<u32>,
    is_hot: Vec<bool>,
}

impl Senses {
    pub fn new(seed: u64, ticks_per_byte: u64) -> Self {
        let mut rng = stream(seed, STREAM_CODES);
        let codes = (0..N_BYTES)
            .map(|_| std::array::from_fn(|_| if rng.random::<bool>() { 1.0 } else { -1.0 }))
            .collect();
        Self {
            codes,
            ticks_per_byte,
            bias: vec![0.0; N_BYTES],
            bias_seen: vec![0; N_BYTES],
            scores: vec![0.0; N_BYTES],
            hot: Vec::new(),
            is_hot: vec![false; N_BYTES],
        }
    }

    /// Input cell `c` received a message in the current inter-byte window.
    pub fn mark_hot(&mut self, c: usize) {
        if !self.is_hot[c] {
            self.is_hot[c] = true;
            self.hot.push(c as u32);
        }
    }

    /// Prior of input cell `c` just before byte number `n` (lazy cold-byte decay applied).
    pub fn prior(&self, c: usize, n: u64, p: &Params) -> S {
        rules::bias_idle(self.bias[c], n - self.bias_seen[c], p)
    }

    /// Teach the prior of a touched cell at byte number `n` from its score error.
    pub fn learn_prior(&mut self, c: usize, n: u64, truth: bool, score: S, p: &Params) {
        self.bias[c] = rules::bias_step(self.prior(c, n, p), truth, score, p);
        self.bias_seen[c] = n + 1;
    }

    /// Hot input cells of the window just closed (insertion order); clears the set. O(hot).
    pub fn take_hot(&mut self) -> Vec<u32> {
        for &c in &self.hot {
            self.is_hot[c as usize] = false;
        }
        std::mem::take(&mut self.hot)
    }
}

/// Guess = highest-scoring input cell (ties -> lowest byte).
pub fn guess(scores: &[S]) -> u32 {
    let mut best = 0;
    for (c, &x) in scores.iter().enumerate() {
        if x > scores[best] {
            best = c;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_deterministic_and_signed() {
        let a = Senses::new(7, 8);
        let b = Senses::new(7, 8);
        assert_eq!(a.codes, b.codes);
        assert_ne!(a.codes, Senses::new(8, 8).codes);
        assert!(a.codes.iter().flatten().all(|&x| x == 1.0 || x == -1.0));
    }

    #[test]
    fn hot_set_dedups_and_clears() {
        let mut s = Senses::new(0, 8);
        s.mark_hot(5);
        s.mark_hot(3);
        s.mark_hot(5);
        assert_eq!(s.take_hot(), vec![5, 3]);
        assert!(s.take_hot().is_empty());
    }

    #[test]
    fn guess_picks_highest_lowest_on_tie() {
        assert_eq!(guess(&[0.0; 4]), 0);
        assert_eq!(guess(&[0.1, 0.5, 0.5, -1.0]), 1);
    }

    #[test]
    fn prior_is_lazy_and_learns() {
        let p = Params::default();
        let mut s = Senses::new(0, 8);
        s.learn_prior(3, 0, true, 0.0, &p);
        assert!((s.bias[3] - 0.005).abs() < 1e-9);
        assert!((s.prior(3, 11, &p) - 0.005 * 0.995f32.powi(10)).abs() < 1e-9);
    }
}
