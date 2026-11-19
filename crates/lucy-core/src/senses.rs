//! Byte-level sensory surface. 256 input cells (cell idx == byte value) both sense and predict:
//! between arrivals they integrate hidden messages sub-threshold (TRACE); each cell reads its
//! trace through its own Q into a score for "my byte comes next", plus a learned prior. An
//! arriving byte forces its cell to fire, and reality teaches every cell that was hot.

use rand::Rng;

use crate::num::{S, V};
use crate::seed::{STREAM_CODES, stream};

pub const N_BYTES: usize = 256;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Senses {
    /// Seed-derived code per byte, entries in {-1, +1} (inside the tanh range of cell states).
    pub codes: Vec<V>,
    pub ticks_per_byte: u64,
    /// Per-input-cell bias logit (a cold cell's whole score).
    pub bias: Vec<S>,
    /// Reused readout buffers: one score (logit) and one softmax probability per input cell.
    pub scores: Vec<S>,
    pub probs: Vec<S>,
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
            scores: vec![0.0; N_BYTES],
            probs: vec![0.0; N_BYTES],
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
}
