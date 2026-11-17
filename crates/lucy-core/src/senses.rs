//! Byte-level sensory surface. 256 input cells (cell idx == byte value) both sense and predict:
//! between arrivals they integrate hidden messages sub-threshold (TRACE = the prediction); an
//! arriving byte forces its cell to fire, and reality teaches every cell that was hot.

use rand::Rng;

use crate::num::{S, V};
use crate::seed::{STREAM_CODES, stream};

pub const N_BYTES: usize = 256;

pub struct Senses {
    /// Seed-derived code per byte, entries in {-1, +1} (inside the tanh range of cell states).
    pub codes: Vec<V>,
    pub ticks_per_byte: u64,
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

/// Guess = hottest input cell (ties -> lowest byte); `None` if nothing is hot.
pub fn guess(heats: &[(u32, S)]) -> Option<u32> {
    heats
        .iter()
        .filter(|&&(_, h)| h > 0.0)
        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|&(c, _)| c)
}

/// `-log2 softmax(heat / sigma_heat)[truth]` over `n` cells, of which only `heats` are nonzero
/// (the rest are implicitly 0). Scale-free: no temperature constant. O(len(heats)).
pub fn surprise_bits(heats: &[(u32, S)], n: usize, truth: u32) -> f64 {
    let nf = n as f64;
    let n_cold = (n - heats.len()) as f64;
    let sum: f64 = heats.iter().map(|&(_, h)| h as f64).sum();
    let sum2: f64 = heats.iter().map(|&(_, h)| (h as f64).powi(2)).sum();
    let mean = sum / nf;
    let sigma = (sum2 / nf - mean * mean).max(0.0).sqrt();
    if sigma == 0.0 {
        return nf.log2();
    }
    let logit = |h: S| h as f64 / sigma;
    let floor = if n_cold > 0.0 { 0.0 } else { f64::NEG_INFINITY };
    let mx = heats.iter().map(|&(_, h)| logit(h)).fold(floor, f64::max);
    let z: f64 = heats
        .iter()
        .map(|&(_, h)| (logit(h) - mx).exp())
        .sum::<f64>()
        + n_cold * (-mx).exp();
    let l_true = heats
        .iter()
        .find(|&&(c, _)| c == truth)
        .map_or(0.0, |&(_, h)| logit(h));
    (mx + z.ln() - l_true) / std::f64::consts::LN_2
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
    fn guess_picks_hottest_lowest_on_tie() {
        assert_eq!(guess(&[]), None);
        assert_eq!(guess(&[(4, 0.0)]), None);
        assert_eq!(guess(&[(9, 1.0), (4, 2.0), (2, 2.0)]), Some(2));
    }

    #[test]
    fn surprise_hand_cases() {
        // Nothing hot: uniform over 256 -> 8 bits.
        assert_eq!(surprise_bits(&[], 256, 65), 8.0);
        // Only the truth is hot: logit = 256/sqrt(255) ~ 16.03, so p ~ 1 - 255 e^-16.03.
        let s = surprise_bits(&[(65, 3.0)], 256, 65);
        assert!(s > 0.0 && s < 1e-4, "{s}");
        // Two equally hot cells, truth among them: ~1 bit (1.0021 by hand).
        let s = surprise_bits(&[(1, 2.0), (2, 2.0)], 256, 2);
        assert!((s - 1.0021).abs() < 1e-3, "{s}");
        // Truth cold while another is hot: worse than uniform.
        assert!(surprise_bits(&[(1, 2.0)], 256, 2) > 8.0);
    }
}
