//! Save format 4 (`LUCYSV04`, commits 857a46f..7007e93^): frozen layouts of the parts that format 5
//! changed, and their upgrade. Format 5 added learned mixing: `Params::ctx_mix`/`ctx_mix_rate`
//! and `Context::mixer`. The upgrade sets `ctx_mix = false`, so a format-4 tissue predicts
//! exactly as before; if mixing is turned on later, θ starts from its fixed weighting.
//! Never edit these structs: they describe bytes already on disk.

use serde::Deserialize;

use crate::buffer::Ring;
use crate::context::ContextV04;
use crate::kernel::Engine;
use crate::num::S;
use crate::observe::Stats;
use crate::plasticity::Plasticity;
use crate::rules::{Params, Tissue};
use crate::scheduler::Scheduler;
use crate::senses::Senses;
use crate::store::Graph;

#[derive(Deserialize)]
pub(super) struct ParamsV04 {
    t_heat: S,
    t_heat_input: S,
    t_rate: S,
    t_v: S,
    t_adapt: S,
    target_rate: S,
    eta_theta: S,
    adapt_step: S,
    eta_q: S,
    nlms_eps: S,
    eta_g: S,
    alpha_tau: S,
    alpha_v: S,
    eta_bias: S,
    eta_readout: S,
    readout_eps: S,
    consolidate_every: u32,
    grow_per_tension: S,
    alpha_err: S,
    alpha_floor: S,
    grow_gain: S,
    prune_vitality: S,
    input_in_cap: u32,
    hidden_grow_per_pass: u32,
    hidden_grow_lag: u32,
    hidden_grow_stride: u32,
    hidden_in_cap: u32,
    replay_per_pass: u32,
    replay_bytes: u32,
    eta_w: S,
    replay_gamma: S,
    replay_min_delta: S,
    dreams_per_pass: u32,
    dream_seeds: u32,
    replay_readout: bool,
    readout_uniform: bool,
    tissue: Tissue,
    ctx_depth: u32,
    ctx_vote_depth: u32,
    ctx_grow_min: f64,
    ctx_t_v: f64,
    ctx_prune_floor: f64,
    ctx_fold_bits: f64,
    ctx_chunk: u32,
    ctx_chunk_bits: f64,
    ctx_chunk_cap: u32,
    cls_mode: u32,
    cls_min_visits: u32,
    cls_radius: f64,
    cls_budget: u32,
    cls_weight: f64,
    cls_random_classes: u32,
    ctx_sweep: u32,
    ctx_budget: u32,
}

impl From<ParamsV04> for Params {
    fn from(o: ParamsV04) -> Params {
        Params {
            t_heat: o.t_heat,
            t_heat_input: o.t_heat_input,
            t_rate: o.t_rate,
            t_v: o.t_v,
            t_adapt: o.t_adapt,
            target_rate: o.target_rate,
            eta_theta: o.eta_theta,
            adapt_step: o.adapt_step,
            eta_q: o.eta_q,
            nlms_eps: o.nlms_eps,
            eta_g: o.eta_g,
            alpha_tau: o.alpha_tau,
            alpha_v: o.alpha_v,
            eta_bias: o.eta_bias,
            eta_readout: o.eta_readout,
            readout_eps: o.readout_eps,
            consolidate_every: o.consolidate_every,
            grow_per_tension: o.grow_per_tension,
            alpha_err: o.alpha_err,
            alpha_floor: o.alpha_floor,
            grow_gain: o.grow_gain,
            prune_vitality: o.prune_vitality,
            input_in_cap: o.input_in_cap,
            hidden_grow_per_pass: o.hidden_grow_per_pass,
            hidden_grow_lag: o.hidden_grow_lag,
            hidden_grow_stride: o.hidden_grow_stride,
            hidden_in_cap: o.hidden_in_cap,
            replay_per_pass: o.replay_per_pass,
            replay_bytes: o.replay_bytes,
            eta_w: o.eta_w,
            replay_gamma: o.replay_gamma,
            replay_min_delta: o.replay_min_delta,
            dreams_per_pass: o.dreams_per_pass,
            dream_seeds: o.dream_seeds,
            replay_readout: o.replay_readout,
            readout_uniform: o.readout_uniform,
            tissue: o.tissue,
            ctx_depth: o.ctx_depth,
            ctx_vote_depth: o.ctx_vote_depth,
            ctx_grow_min: o.ctx_grow_min,
            ctx_t_v: o.ctx_t_v,
            ctx_prune_floor: o.ctx_prune_floor,
            ctx_fold_bits: o.ctx_fold_bits,
            ctx_chunk: o.ctx_chunk,
            ctx_chunk_bits: o.ctx_chunk_bits,
            ctx_chunk_cap: o.ctx_chunk_cap,
            cls_mode: o.cls_mode,
            cls_min_visits: o.cls_min_visits,
            cls_radius: o.cls_radius,
            cls_budget: o.cls_budget,
            cls_weight: o.cls_weight,
            cls_random_classes: o.cls_random_classes,
            ctx_sweep: o.ctx_sweep,
            ctx_budget: o.ctx_budget,
            ctx_mix: false,
            ctx_mix_rate: Params::default().ctx_mix_rate,
        }
    }
}

/// The engine as saved in format 4 (same field order; `parallel` and `decay` are not saved).
#[derive(Deserialize)]
pub(super) struct EngineV04 {
    p: ParamsV04,
    g: Graph,
    senses: Senses,
    ring: Ring,
    stats: Stats,
    sched: Scheduler,
    plasticity: Plasticity,
    now: u64,
    ctx: ContextV04,
}

impl From<EngineV04> for Engine {
    fn from(e: EngineV04) -> Engine {
        Engine::from_parts(
            e.p.into(),
            e.g,
            e.senses,
            e.ring,
            e.stats,
            e.sched,
            e.plasticity,
            e.now,
            e.ctx.into(),
        )
    }
}
