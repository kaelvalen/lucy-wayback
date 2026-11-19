//! Learned mixing of the chain's votes (`ctx_mix`): the prediction is the linear mixture
//! `M(x) = Σ_c a_c p̂_c(x) / Σ_c a_c` of the chain cells' successor distributions, with
//! `a_c = exp(θ[depth][evidence])`. One log-reliability θ per (depth × evidence) bucket,
//! evidence = ⌊log2 T⌋, learned online: after each byte, one gradient step on that byte's
//! log-loss `−ln p(x)`, `p = (1 − ε) M + ε/256`:
//!   `∂ ln p(x) / ∂θ_b = (1 − ε)/p(x) · Σ_{c ∈ b} a_c (p̂_c(x) − M(x)) / Σ_c a_c`.
//! Only the chain's buckets move: O(chain) per byte, local, no global pass.
//! θ starts at the fixed weighting `2^min(depth, cap) · T/(T+2)`, so learning begins from the
//! best fixed variant.

use serde::{Deserialize, Serialize};

pub const DEPTHS: usize = 65;
pub const EVIDENCE: usize = 24;
/// Kinds of extra voters from outside the chain (e.g. the word level, `words.rs`).
pub const EXTRA_KINDS: usize = 8;
/// Largest step one byte may take on one θ (stability against rare huge gradients).
const MAX_STEP: f64 = 0.5;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Mixer {
    theta: Vec<f64>,
}

fn evidence(total: f64) -> usize {
    let e = if total >= 1.0 {
        total.log2() as usize
    } else {
        0
    };
    e.min(EVIDENCE - 1)
}

pub fn bucket(depth: u32, total: f64) -> usize {
    (depth as usize).min(DEPTHS - 1) * EVIDENCE + evidence(total)
}

/// Buckets of extra voters sit after the chain's: one row of evidence buckets per kind.
pub fn extra_bucket(kind: usize, total: f64) -> usize {
    (DEPTHS + kind.min(EXTRA_KINDS - 1)) * EVIDENCE + evidence(total)
}

impl Mixer {
    /// Fill θ with the fixed weighting on first use (`cap` = depth whose weight stops doubling).
    /// Extra voters start like a cell at depth `cap`. A shorter θ (an older save) is extended.
    pub fn ensure(&mut self, cap: u32) {
        let n = (DEPTHS + EXTRA_KINDS) * EVIDENCE;
        if self.theta.len() >= n {
            return;
        }
        let start = self.theta.len();
        self.theta.extend((start..n).map(|i| {
            let (d, e) = (((i / EVIDENCE) as u32).min(DEPTHS as u32), i % EVIDENCE);
            let t = (1u64 << e) as f64;
            f64::from(d.min(cap)) * std::f64::consts::LN_2 + (t / (t + 2.0)).ln()
        }));
    }

    pub fn weight(&self, b: usize) -> f64 {
        self.theta[b].exp()
    }

    /// One step for the byte that arrived. `members`: (bucket, a, p̂(x)) of each voter, `z` = Σ a,
    /// `m` = M(x), `eps` as in the prediction.
    pub fn learn(&mut self, members: &[(usize, f64, f64)], z: f64, m: f64, eps: f64, rate: f64) {
        if z <= 0.0 {
            return;
        }
        let p = (1.0 - eps) * m + eps / 256.0;
        for &(b, a, ph) in members {
            let g = (1.0 - eps) / p * a * (ph - m) / z;
            self.theta[b] += (rate * g).clamp(-MAX_STEP, MAX_STEP);
        }
    }

    pub fn feed_hash(&self, h: &mut crate::observe::Fnv) {
        for t in &self.theta {
            h.u64(t.to_bits());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two voters, one always right: its bucket's weight rises, the other's falls.
    #[test]
    fn reliable_buckets_gain_weight() {
        let mut m = Mixer::default();
        m.ensure(8);
        let (good, bad) = (bucket(3, 10.0), bucket(5, 10.0));
        let (g0, b0) = (m.weight(good), m.weight(bad));
        for _ in 0..200 {
            let (a, b) = (m.weight(good), m.weight(bad));
            let z = a + b;
            let mx = (a * 1.0 + b * 0.0) / z;
            m.learn(&[(good, a, 1.0), (bad, b, 0.0)], z, mx, 1e-3, 0.05);
        }
        assert!(m.weight(good) > g0 && m.weight(bad) < b0);
        assert_eq!(bucket(0, 0.0), 0);
        assert_eq!(bucket(100, 1e30), DEPTHS * EVIDENCE - 1);
    }
}
