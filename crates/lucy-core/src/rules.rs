//! Every dynamics and learning rule, and every constant, lives here, so the physics can be
//! swapped without touching storage or scheduling. Rationale for each: docs/DESIGN.md.

use serde::{Deserialize, Serialize};

use crate::num::{D, M, S, V, axpy, dot, matvec, norm2, outer_add, scale, sub};

/// All tunable constants. Timescales are in ticks; step sizes are dimensionless.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Params {
    /// Decay timescale of u, heat and credit (the coincidence / trace window).
    pub t_heat: S,
    /// Rate EMA timescale; also the evidence horizon of threshold homeostasis.
    pub t_rate: S,
    /// Vitality timescale (cells and edges).
    pub t_v: S,
    /// Decay timescale of fast threshold adaptation (fix 5).
    pub t_adapt: S,
    /// Target fires per tick per cell. Sets the compute budget: fires/tick ~ N * target_rate.
    pub target_rate: S,
    /// Slow log-threshold homeostasis step per tick (fixes 1, 2).
    pub eta_theta: S,
    /// Fast adaptation added per fire: theta_eff = theta * (1 + adapt) (fix 5).
    pub adapt_step: S,
    /// Normalized-LMS step for the predictor Q, in (0, 2) (fix 3).
    pub eta_q: S,
    /// NLMS regularizer so h_prev = 0 is a no-op instead of a division by zero.
    pub nlms_eps: S,
    /// Credit step for in-edge gains (followed by synaptic scaling, fix 4).
    pub eta_g: S,
    /// Per-fire EMA weight of tau (tension).
    pub alpha_tau: S,
    /// Vitality added per use (cell fire / edge transmit).
    pub alpha_v: S,
    /// Cross-entropy step of each input cell's bias logit (competitive readout).
    pub eta_bias: S,
    /// Competitive readout step on input cells' Q, normalized by `readout_eps + |u|^2`: plain
    /// logistic SGD for weak inputs, normalized for strong ones (a tiny regularizer lets a
    /// weak input take a full step that strong inputs then amplify into huge logits).
    pub eta_readout: S,
    pub readout_eps: S,
    /// Structural plasticity: consolidate (replay, grow, prune) every this many bytes; 0 = off.
    pub consolidate_every: u32,
    /// Growth law: edges an input cell may grow per unit of accumulated tension excess
    /// (sum over its arrivals of err - floor since the last pass).
    pub grow_per_tension: S,
    /// EMA weight of an input cell's prediction error, per arrival of its own byte.
    pub alpha_err: S,
    /// Per-arrival decay of the error floor toward 0 (it also follows the error down at once).
    pub alpha_floor: S,
    /// Gain of a newly grown edge (synaptic scaling then folds it into the cell's budget).
    pub grow_gain: S,
    /// In-edges of input cells whose earned vitality decays below this die.
    pub prune_vitality: S,
    /// In-degree cap of input cells; over it, the lowest-vitality edge dies.
    pub input_in_cap: u32,
    /// Hidden -> hidden growth: new A -> B edges per consolidation pass (0 = off), for pairs where
    /// B fires 1..=hidden_grow_lag ticks after A more often than chance.
    pub hidden_grow_per_pass: u32,
    pub hidden_grow_lag: u32,
    /// Each pass considers targets B with B % stride == pass % stride (cost / stride; every
    /// cell is a candidate target once per `stride` passes).
    pub hidden_grow_stride: u32,
    /// Cap on a hidden cell's hidden -> hidden in-degree (edges from input cells never compete).
    pub hidden_in_cap: u32,
    /// Wayback (reverse replay into W): segments per consolidation pass (0 = off), each the
    /// `replay_bytes` bytes before one of the pass's most surprising arrivals.
    pub replay_per_pass: u32,
    pub replay_bytes: u32,
    /// W step (NLMS through tanh), << eta_q so Q can track the drift.
    pub eta_w: S,
    /// Per-hop decay of the feedback signal, and the RMS below which the chain stops.
    pub replay_gamma: S,
    pub replay_min_delta: S,
    /// Dreams per pass (pre-update W, for dream consistency) and seed cells per dream.
    pub dreams_per_pass: u32,
    pub dream_seeds: u32,
    /// Readout consolidation: re-run replayed segments forward under the updated W and give
    /// the competitive readout its step on the recomputed input traces.
    pub replay_readout: bool,
    /// Readout consolidation segments: a uniform sample of the pass's byte arrivals (true) or
    /// the surprise-ranked W-replay segments (false). Same count as `replay_per_pass`.
    pub readout_uniform: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            t_heat: 4.0,
            t_rate: 2000.0,
            t_v: 100_000.0,
            t_adapt: 50.0,
            target_rate: 0.005,
            eta_theta: 1e-3,
            adapt_step: 1.0,
            eta_q: 0.1,
            nlms_eps: 1e-3,
            eta_g: 0.05,
            alpha_tau: 0.05,
            alpha_v: 1.0,
            eta_bias: 0.01,
            eta_readout: 0.3,
            readout_eps: 1.0,
            consolidate_every: 1000,
            grow_per_tension: 1.0,
            alpha_err: 0.03,
            alpha_floor: 0.001,
            grow_gain: 0.1,
            prune_vitality: 0.5,
            input_in_cap: 32,
            hidden_grow_per_pass: 0,
            hidden_grow_lag: 3,
            hidden_grow_stride: 8,
            hidden_in_cap: 16,
            replay_per_pass: 0,
            replay_bytes: 4,
            eta_w: 0.01,
            replay_gamma: 0.5,
            replay_min_delta: 1e-3,
            dreams_per_pass: 32,
            dream_seeds: 8,
            replay_readout: true,
            readout_uniform: true,
        }
    }
}

impl Params {
    pub fn validate(&self) -> Result<(), String> {
        let positive = [
            ("t_heat", self.t_heat),
            ("t_rate", self.t_rate),
            ("t_v", self.t_v),
            ("t_adapt", self.t_adapt),
            ("target_rate", self.target_rate),
            ("nlms_eps", self.nlms_eps),
            ("readout_eps", self.readout_eps),
        ];
        let non_negative = [
            ("eta_theta", self.eta_theta),
            ("adapt_step", self.adapt_step),
            ("eta_g", self.eta_g),
            ("eta_readout", self.eta_readout),
            ("eta_w", self.eta_w),
            ("replay_gamma", self.replay_gamma),
            ("replay_min_delta", self.replay_min_delta),
            ("alpha_v", self.alpha_v),
        ];
        for (name, x) in positive {
            if !(x.is_finite() && x > 0.0) {
                return Err(format!("{name} must be finite and > 0, got {x}"));
            }
        }
        for (name, x) in non_negative {
            if !(x.is_finite() && x >= 0.0) {
                return Err(format!("{name} must be finite and >= 0, got {x}"));
            }
        }
        if !(self.eta_q >= 0.0 && self.eta_q < 2.0) {
            return Err(format!(
                "eta_q must be in [0, 2) for NLMS stability, got {}",
                self.eta_q
            ));
        }
        for (name, x) in [
            ("alpha_err", self.alpha_err),
            ("alpha_floor", self.alpha_floor),
        ] {
            if !(x > 0.0 && x <= 1.0) {
                return Err(format!("{name} must be in (0, 1], got {x}"));
            }
        }
        if !(self.grow_per_tension.is_finite() && self.grow_per_tension >= 0.0) {
            return Err("grow_per_tension must be finite and >= 0".into());
        }
        if !(self.grow_gain.is_finite()
            && self.prune_vitality.is_finite()
            && self.prune_vitality >= 0.0)
        {
            return Err("grow_gain must be finite and prune_vitality finite and >= 0".into());
        }
        if self.hidden_grow_per_pass > 0
            && (self.hidden_grow_lag == 0
                || self.hidden_in_cap == 0
                || self.hidden_grow_stride == 0)
        {
            return Err("hidden growth needs lag, in-cap and stride >= 1".into());
        }
        if self.consolidate_every > 0 && self.input_in_cap == 0 {
            return Err("input_in_cap must be >= 1 when consolidation is on".into());
        }
        if !(self.eta_bias >= 0.0 && self.eta_bias < 1.0) {
            return Err(format!("eta_bias must be in [0, 1), got {}", self.eta_bias));
        }
        if !(self.alpha_tau > 0.0 && self.alpha_tau <= 1.0) {
            return Err(format!(
                "alpha_tau must be in (0, 1], got {}",
                self.alpha_tau
            ));
        }
        Ok(())
    }
}

/// The lazily-decaying scalar state of one cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scalars {
    pub heat: S,
    pub log_theta: S,
    pub adapt: S,
    pub rate: S,
    pub v: S,
}

/// Exponential decay factor over `dt` ticks with timescale `tau`.
#[inline]
pub fn decay(dt: u64, tau: S) -> S {
    if dt == 0 {
        1.0
    } else {
        (-(dt as S) / tau).exp()
    }
}

/// Integral of an exponentially decaying rate r0 over `dt` ticks.
#[inline]
pub fn rate_integral(r0: S, dt: S, t_rate: S) -> S {
    r0 * t_rate * (1.0 - (-dt / t_rate).exp())
}

/// Lazy decay of a cell's scalars from its last touch to now. Pure, so the observatory can
/// read state at any time without perturbing it.
///
/// Threshold homeostasis (fixes 1+2) is integrated in closed form over the interval in the log
/// domain, so silent cells lower their threshold when next touched. The evidence horizon is
/// capped at `t_rate`: the rate EMA has forgotten anything older, so silence beyond it would
/// otherwise be counted without bound and drive theta to zero.
pub fn decay_scalars(s: Scalars, dt: u64, p: &Params) -> Scalars {
    if dt == 0 {
        return s;
    }
    let horizon = (dt as S).min(p.t_rate);
    let fired = rate_integral(s.rate, horizon, p.t_rate);
    Scalars {
        heat: s.heat * decay(dt, p.t_heat),
        log_theta: s.log_theta + p.eta_theta * (fired / p.target_rate - horizon),
        adapt: s.adapt * decay(dt, p.t_adapt),
        rate: s.rate * decay(dt, p.t_rate),
        v: s.v * decay(dt, p.t_v),
    }
}

/// Fast path for [`decay_scalars`]: the same factors, read from per-timescale tables for short
/// intervals (filled by the same `decay` function, so bit-identical) and computed once each.
/// Built from `Params` at engine construction.
#[derive(Clone, Debug)]
pub struct Decay {
    heat: Vec<S>,
    adapt: Vec<S>,
    rate: Vec<S>,
    v: Vec<S>,
    /// `exp(-horizon / t_rate)` once the horizon is capped at `t_rate`.
    rate_capped: S,
}

const DECAY_TABLE: u64 = 2048;

impl Decay {
    pub fn new(p: &Params) -> Self {
        let table = |tau: S| (0..DECAY_TABLE).map(|dt| decay(dt, tau)).collect();
        Self {
            heat: table(p.t_heat),
            adapt: table(p.t_adapt),
            rate: table(p.t_rate),
            v: table(p.t_v),
            rate_capped: (-p.t_rate / p.t_rate).exp(),
        }
    }

    #[inline]
    fn factor(table: &[S], dt: u64, tau: S) -> S {
        if dt < DECAY_TABLE {
            table[dt as usize]
        } else {
            decay(dt, tau)
        }
    }

    /// The trace (u, heat, credit) decay factor over `dt`.
    #[inline]
    pub fn heat(&self, dt: u64, p: &Params) -> S {
        Self::factor(&self.heat, dt, p.t_heat)
    }

    /// Exactly [`decay_scalars`], with each exponential evaluated at most once.
    pub fn scalars(&self, s: Scalars, dt: u64, p: &Params) -> Scalars {
        if dt == 0 {
            return s;
        }
        let f_rate = Self::factor(&self.rate, dt, p.t_rate);
        let horizon = (dt as S).min(p.t_rate);
        // rate_integral's exp(-horizon / t_rate) is f_rate itself while the horizon is uncapped.
        let e = if (dt as S) <= p.t_rate {
            f_rate
        } else {
            self.rate_capped
        };
        let fired = s.rate * p.t_rate * (1.0 - e);
        Scalars {
            heat: s.heat * self.heat(dt, p),
            log_theta: s.log_theta + p.eta_theta * (fired / p.target_rate - horizon),
            adapt: s.adapt * Self::factor(&self.adapt, dt, p.t_adapt),
            rate: s.rate * f_rate,
            v: s.v * Self::factor(&self.v, dt, p.t_v),
        }
    }
}

/// Effective threshold: slow homeostatic level times fast adaptation.
#[inline]
pub fn theta_eff(s: &Scalars) -> S {
    s.log_theta.exp() * (1.0 + s.adapt)
}

/// Scalar bookkeeping of one fire (state must already be decayed to now).
pub fn on_fire(s: &mut Scalars, p: &Params) {
    s.heat = 0.0;
    s.rate += 1.0 / p.t_rate;
    s.adapt += p.adapt_step;
    s.v += p.alpha_v;
}

/// Edge vitality after a transmit `dt` ticks after its last use.
#[inline]
pub fn edge_vitality(v: S, dt: u64, p: &Params) -> S {
    v * decay(dt, p.t_v) + p.alpha_v
}

/// TENSION of a hidden cell: `e = u - Q h`, `h` = the state its last fire produced.
#[inline]
pub fn prediction_error(u: &V, q: &M, h: &V) -> V {
    sub(u, &matvec(q, h))
}

/// Normalized LMS: `Q += eta_q / (eps + |h_prev|^2) * e h_prev^T` (fix 3, stable for eta_q < 2).
pub fn nlms(q: &mut M, e: &V, h_prev: &V, p: &Params) {
    let s = p.eta_q / (p.nlms_eps + norm2(h_prev));
    outer_add(q, e, h_prev, s);
}

/// Local credit for one in-edge: `delta gain = eta_g * dot(toward, m) / D`, where `toward` is
/// the direction the cell's input should move. Hidden cells pass `-e` (inputs move toward the
/// prediction); input cells pass `code - u` (inputs move toward reality).
#[inline]
pub fn credit_step(toward: &V, m: &V, eta_g: S) -> S {
    eta_g * dot(toward, m) / D as S
}

/// Synaptic scaling (fix 4): multiplier that restores the in-gain L2 norm to its birth value.
#[inline]
pub fn scale_factor(norm2: S, target: S) -> S {
    if norm2 > 0.0 {
        target / norm2.sqrt()
    } else {
        1.0
    }
}

/// RMS clip (fix 6): `u * sqrt(D) / max(|u|, sqrt(D))`, so tanh never saturates on input scale.
pub fn clip_rms(u: &V) -> V {
    let cap = (D as S).sqrt();
    let n = norm2(u).sqrt();
    if n <= cap {
        return *u;
    }
    let mut out = [0.0; D];
    axpy(&mut out, cap / n, u);
    out
}

/// New state: `h = tanh(W h + clip(u))`.
#[inline]
pub fn activate(w: &M, h: &V, u: &V) -> V {
    let mut x = matvec(w, h);
    axpy(&mut x, 1.0, &clip_rms(u));
    x.map(S::tanh)
}

/// Input-cell readout: the cell predicts its own byte code from its input, `pred = Q u`, and
/// scores `prior + dot(pred, code) / D` (1 = "my byte comes next", 0 = "not mine").
pub fn readout(q: &M, u: &V, code: &V, prior: S) -> (V, S) {
    let pred = matvec(q, u);
    let score = prior + dot(&pred, code) / D as S;
    (pred, score)
}

/// Competitive readout: `p = softmax(scores)` over the whole sensory sheet (max-subtracted,
/// fixed summation order).
pub fn softmax(scores: &[S], out: &mut [S]) {
    let mx = scores.iter().copied().fold(S::NEG_INFINITY, S::max);
    let mut z = 0.0;
    for (o, &s) in out.iter_mut().zip(scores) {
        *o = (s - mx).exp();
        z += *o;
    }
    for o in out.iter_mut() {
        *o /= z;
    }
}

/// Softmax error of an input cell as a vector error for its readout: `e = (y - p) code`.
/// NLMS on Q with this e moves the cell's evidence `dot(Q u, code) / D` by ~`eta_q (y - p)`,
/// the cross-entropy gradient; `Q^T e` is the direction its input should move (in-gain credit).
#[inline]
pub fn readout_error(code: &V, truth: bool, p: S) -> V {
    let mut e = *code;
    scale(&mut e, S::from(u8::from(truth)) - p);
    e
}

/// Readout step on an input cell's Q: `Q += eta_readout / (readout_eps + |u|^2) * e u^T`.
pub fn readout_step(q: &mut M, e: &V, u: &V, p: &Params) {
    let s = p.eta_readout / (p.readout_eps + norm2(u));
    outer_add(q, e, u, s);
}

/// Desired change of a tanh cell's pre-activation for a desired change `delta` of its output
/// `h`: `(1 - h^2) * delta` (elementwise).
#[inline]
pub fn pre_activation_delta(h: &V, delta: &V) -> V {
    std::array::from_fn(|i| (1.0 - h[i] * h[i]) * delta[i])
}

/// Local W step (wayback): NLMS toward a pre-activation change `g` for the transition from
/// `h_prev`: `W += eta_w / (nlms_eps + |h_prev|^2) * g h_prev^T`; then W-norm homeostasis caps
/// the Frobenius norm at its birth value `norm0` (the cell's spectral share never grows).
pub fn w_step(w: &mut M, g: &V, h_prev: &V, norm0: S, p: &Params) {
    let s = p.eta_w / (p.nlms_eps + norm2(h_prev));
    outer_add(w, g, h_prev, s);
    let n = w.as_flattened().iter().map(|x| x * x).sum::<S>().sqrt();
    if n > norm0 && n > 0.0 {
        let k = norm0 / n;
        for x in w.as_flattened_mut() {
            *x *= k;
        }
    }
}

/// Bias logit, cross-entropy step for every cell of the sheet: `bias += eta_bias (y - p)`.
#[inline]
pub fn bias_step(bias: S, truth: bool, p: S, params: &Params) -> S {
    bias + params.eta_bias * (S::from(u8::from(truth)) - p)
}

#[inline]
pub fn ema(old: S, x: S, alpha: S) -> S {
    old + alpha * (x - old)
}

/// TENSION drives structure. At each arrival of its own byte, an input cell updates the EMA of
/// its prediction error `x = |code - pred|^2 / D` and a running floor of that EMA: the floor
/// follows the error down at once and otherwise decays toward 0 by `alpha_floor` per arrival
/// (an aspiration level). Returns (err, floor, excess = err - floor): zero while the error keeps
/// falling faster than the floor decays, growing while it is stuck, including stuck at "predicts
/// nothing" (err = 1), which a floor that only tracked the error's own minimum would never see.
pub fn tension_step(err: S, floor: S, x: S, p: &Params) -> (S, S, S) {
    let err = ema(err, x, p.alpha_err);
    let floor = err.min(floor * (1.0 - p.alpha_floor));
    (err, floor, err - floor)
}

/// Edges a cell may grow this pass: `grow_per_tension * excess + carry`, at most `input_in_cap`;
/// the fractional part carries to the next pass. Returns (budget, carry).
pub fn growth_budget(excess: S, carry: S, p: &Params) -> (u32, S) {
    let x = p.grow_per_tension * excess + carry;
    let n = x.floor();
    (n.min(p.input_in_cap as S) as u32, x - n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::{ZERO, ZERO_M, basis};

    fn close(a: S, b: S, tol: S) -> bool {
        (a - b).abs() <= tol
    }

    fn sc(rate: S) -> Scalars {
        Scalars {
            heat: 2.0,
            log_theta: 0.3,
            adapt: 1.5,
            rate,
            v: 4.0,
        }
    }

    #[test]
    fn lazy_decay_composes() {
        // Decaying in two steps must equal decaying once over the sum (within the horizon).
        let p = Params::default();
        let s = sc(0.02);
        let two = decay_scalars(decay_scalars(s, 300, &p), 700, &p);
        let one = decay_scalars(s, 1000, &p);
        assert!(close(two.heat, one.heat, 1e-6));
        assert!(close(two.rate, one.rate, 1e-7));
        assert!(close(two.v, one.v, 1e-5));
        assert!(close(two.adapt, one.adapt, 1e-6));
        assert!(close(two.log_theta, one.log_theta, 1e-4));
    }

    #[test]
    fn lazy_decay_values() {
        let p = Params::default();
        let s = decay_scalars(sc(0.0), 4, &p);
        assert!(close(s.heat, 2.0 * (-1.0f32).exp(), 1e-6));
        assert!(close(s.adapt, 1.5 * (-4.0f32 / 50.0).exp(), 1e-6));
        assert_eq!(decay_scalars(sc(0.3), 0, &p), sc(0.3));
    }

    #[test]
    fn silent_cell_lowers_theta_with_capped_horizon() {
        let p = Params::default();
        let s = decay_scalars(sc(0.0), 100, &p);
        assert!(close(s.log_theta, 0.3 - p.eta_theta * 100.0, 1e-6));
        let far = decay_scalars(sc(0.0), 10_000_000, &p);
        assert!(close(far.log_theta, 0.3 - p.eta_theta * p.t_rate, 1e-5));
    }

    /// The fast path must be bit-identical to the reference for every interval length.
    #[test]
    fn decay_table_is_bit_identical() {
        let p = Params::default();
        let d = Decay::new(&p);
        let s = Scalars {
            heat: 1.7,
            log_theta: -0.4,
            adapt: 2.3,
            rate: 0.0071,
            v: 412.0,
        };
        let dts = (0..3000u64).chain([1999, 2000, 2001, 2047, 2048, 5000, 99_999, 10_000_000]);
        for dt in dts {
            let (a, b) = (decay_scalars(s, dt, &p), d.scalars(s, dt, &p));
            let bits = |x: Scalars| [x.heat, x.log_theta, x.adapt, x.rate, x.v].map(S::to_bits);
            assert_eq!(bits(a), bits(b), "dt = {dt}");
            assert_eq!(d.heat(dt, &p).to_bits(), decay(dt, p.t_heat).to_bits());
        }
    }

    #[test]
    fn rate_integral_matches_numeric() {
        let (r0, dt, tr) = (0.04, 500.0, 2000.0);
        let steps = 100_000;
        let h = dt / steps as S;
        let numeric: f64 = (0..steps)
            .map(|i| (r0 * (-((i as S + 0.5) * h) / tr).exp() * h) as f64)
            .sum();
        assert!((rate_integral(r0, dt, tr) as f64 - numeric).abs() < 1e-3);
    }

    #[test]
    fn theta_rises_when_firing_above_target() {
        let p = Params::default();
        let s = decay_scalars(sc(10.0 * p.target_rate), 10, &p);
        assert!(s.log_theta > 0.3);
    }

    #[test]
    fn prediction_error_hand_case() {
        let mut q = ZERO_M;
        q[0][0] = 0.5;
        q[1][1] = 0.5;
        let mut h_prev = ZERO;
        h_prev[0] = 1.0;
        h_prev[1] = 2.0;
        let mut u = ZERO;
        u[0] = 1.0;
        u[1] = 1.0;
        let e = prediction_error(&u, &q, &h_prev);
        assert_eq!((e[0], e[1], e[2]), (0.5, 0.0, 0.0));
    }

    #[test]
    fn nlms_hand_case() {
        let p = Params {
            eta_q: 0.1,
            nlms_eps: 1e-3,
            ..Params::default()
        };
        let mut q = ZERO_M;
        let mut h_prev = ZERO;
        h_prev[0] = 1.0;
        h_prev[1] = 1.0;
        nlms(&mut q, &basis(0), &h_prev, &p);
        let want = 0.1 / 2.001;
        assert!(close(q[0][0], want, 1e-7) && close(q[0][1], want, 1e-7));
        assert_eq!(q[1][0], 0.0);
        // NLMS fully corrects in one step when eta_q = 1: the new prediction matches u.
        let p1 = Params {
            eta_q: 1.0,
            nlms_eps: 1e-9,
            ..Params::default()
        };
        let mut q = ZERO_M;
        let e = prediction_error(&basis(3), &q, &h_prev);
        nlms(&mut q, &e, &h_prev, &p1);
        assert!(norm2(&prediction_error(&basis(3), &q, &h_prev)) < 1e-6);
    }

    #[test]
    fn credit_and_scaling_hand_case() {
        let mut m = ZERO;
        m[0] = 2.0;
        assert!(close(credit_step(&basis(0), &m, 0.05), 0.003125, 1e-9));
        assert_eq!(scale_factor(25.0, 10.0), 2.0); // gains [3,4] -> [6,8]
        assert_eq!(scale_factor(0.0, 10.0), 1.0);
    }

    #[test]
    fn clip_rms_caps_norm() {
        let big = [2.0; D]; // norm = 2 sqrt(D)
        let c = clip_rms(&big);
        assert!(close(norm2(&c).sqrt(), (D as S).sqrt(), 1e-4));
        let small = [0.5; D];
        assert_eq!(clip_rms(&small), small);
    }

    #[test]
    fn activate_hand_case() {
        let mut u = ZERO;
        u[0] = 0.5;
        let h = activate(&ZERO_M, &ZERO, &u);
        assert!(close(h[0], 0.5f32.tanh(), 1e-7));
        assert_eq!(h[1], 0.0);
    }

    #[test]
    fn readout_and_prior_hand_cases() {
        let p = Params::default();
        let mut q = ZERO_M;
        q[1][0] = 2.0;
        let mut u = ZERO;
        u[0] = 0.5;
        let mut code = ZERO;
        code[1] = 1.0;
        code[2] = -1.0;
        let (pred, s) = readout(&q, &u, &code, 0.25);
        assert_eq!(pred, basis(1));
        assert!(close(s, 0.25 + 1.0 / 32.0, 1e-7));
        assert!(close(
            bias_step(0.2, true, 0.6, &p),
            0.2 + p.eta_bias * 0.4,
            1e-7
        ));
        let e = readout_error(&[1.0; D], false, 0.25);
        assert!(e.iter().all(|&x| x == -0.25));
    }

    /// A stream of i.i.d. bytes: priors converge to byte frequencies (frequency is learnable).
    #[test]
    fn priors_learn_frequencies() {
        let p = Params {
            eta_bias: 0.01,
            ..Params::default()
        };
        // Softmax biases converge to log-frequencies (up to a shared offset).
        let (mut bias, mut pr) = ([0.0; 3], [0.0; 3]);
        for i in 0..20_000 {
            let y = [i % 4 != 0, i % 4 == 0, false]; // 75 %, 25 %, never
            softmax(&bias, &mut pr);
            for k in 0..3 {
                bias[k] = bias_step(bias[k], y[k], pr[k], &p);
            }
        }
        softmax(&bias, &mut pr);
        assert!(
            close(pr[0], 0.75, 0.03) && close(pr[1], 0.25, 0.03),
            "{pr:?}"
        );
        assert!(pr[2] < 0.01, "{pr:?}");
    }

    #[test]
    fn falling_error_grows_nothing_stuck_error_does() {
        let p = Params::default();
        let (mut err, mut floor, mut total) = (1.0, 1.0, 0.0);
        for i in 0..200 {
            let x = 0.5 - i as S / 500.0; // steadily improving, faster than the floor decays
            let (e, f, ex) = tension_step(err, floor, x, &p);
            (err, floor, total) = (e, f, total + ex);
        }
        assert_eq!(total, 0.0);
        let mut total = 0.0;
        for i in 0..500 {
            let x = if i % 2 == 0 { 0.9 } else { 0.3 }; // stuck, noisy
            let (e, f, ex) = tension_step(err, floor, x, &p);
            (err, floor, total) = (e, f, total + ex);
        }
        assert!(total > 1.0, "{total}");
        // Stuck at "predicts nothing": excess still accumulates.
        let (mut err, mut floor, mut total) = (1.0, 1.0, 0.0);
        for _ in 0..500 {
            let (e, f, ex) = tension_step(err, floor, 1.0, &p);
            (err, floor, total) = (e, f, total + ex);
        }
        assert!(total > 50.0, "{total}");
        // Budget: integer part now, fraction carried; capped at input_in_cap.
        let p = Params {
            grow_per_tension: 2.0,
            ..Params::default()
        };
        assert_eq!(growth_budget(1.3, 0.0, &p), (2, 0.5999999));
        assert_eq!(growth_budget(0.1, 0.6, &p).0, 0);
        assert_eq!(growth_budget(1e6, 0.0, &p).0, p.input_in_cap);
    }

    #[test]
    fn validate_rejects_bad_params() {
        assert!(Params::default().validate().is_ok());
        assert!(
            Params {
                eta_q: 2.5,
                ..Params::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Params {
                t_heat: 0.0,
                ..Params::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Params {
                nlms_eps: S::NAN,
                ..Params::default()
            }
            .validate()
            .is_err()
        );
    }
}
