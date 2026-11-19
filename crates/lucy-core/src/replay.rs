//! Wayback: reverse replay with one-hop feedback into W, protected by pre-update dreams.
//! Runs inside consolidation (see docs/DESIGN.md, "reverse replay").
//!
//! 1. Dreams first, with the pre-update W: seeded spontaneous runs in a sandbox (scratch input
//!    integration, frozen thresholds, no learning, no writes to live state). Every dream fire
//!    records `(h_prev, u, h)`.
//! 2. Real replays: ring segments ending at the most surprising byte arrivals of the last pass,
//!    walked backward in time. Each replayed fire carries `delta`, the desired change of the
//!    state it produced, sent one hop back per step using only the connecting edge's gain.
//!    W of the fired cell takes a local NLMS step through tanh.
//! 3. Dream consistency: each cell whose W changed is pulled back toward reproducing its
//!    recorded pre-update dream transitions.
//! 4. Readout consolidation: every replayed segment is re-run *forward* under the updated W,
//!    with the recorded spike timing (who fired when, including the byte arrivals, which are
//!    the input cells' forced fires in the ring) and recomputed states, no tissue learning.
//!    At each recorded byte the competitive readout takes its usual step on the recomputed
//!    input-cell traces against the true byte, so Q follows the states W now produces.
//!
//! Cost is bounded by `replay_per_pass` segments and `dreams_per_pass` dreams of
//! `replay_bytes * K` ticks each: it scales with activity, never with total graph size.

use std::collections::{BTreeMap, BTreeSet};

use rand::Rng;

use crate::buffer::{FireEvent, NO_PREV};
use crate::kernel::Engine;
use crate::num::{D, S, V, ZERO, axpy, norm};
use crate::rules;
use crate::seed::{STREAM_DREAM, stream};
use crate::store::Kind;

/// A surprising byte arrival kept for replay.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Anchor {
    /// Competitive-readout surprise `-ln p(true byte)`.
    pub surprise: S,
    pub t: u64,
    /// Ring length at the arrival: every fire before the byte has a smaller sequence number.
    pub seq: u64,
    /// Desired change of the input cell's input, `Q_b^T e_b`, with `e_b = (1 - p_b) code_b`.
    pub toward: V,
    /// Hidden cells whose messages were in the cell's trace window, with the edge gains.
    pub sources: Vec<(u32, S)>,
}

/// Keep the `cap` most surprising anchors (ties keep the earlier one).
pub fn offer(anchors: &mut Vec<Anchor>, a: Anchor, cap: usize) {
    if anchors.len() < cap {
        anchors.push(a);
        return;
    }
    let (i, min) = anchors
        .iter()
        .enumerate()
        .fold((0, S::INFINITY), |(bi, bm), (i, x)| {
            if x.surprise < bm {
                (i, x.surprise)
            } else {
                (bi, bm)
            }
        });
    if a.surprise > min {
        anchors[i] = a;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReplayReport {
    pub segments: u64,
    /// Anchors whose segment was no longer in the ring.
    pub lost: u64,
    pub w_updates: u64,
    pub delta_rms_sum: f64,
    pub dream_fires: u64,
    pub dream_updates: u64,
    /// Forward readout consolidation: bytes re-scored (and guessed right) under the new W.
    pub readout_updates: u64,
    pub readout_correct: u64,
    /// Mean |recomputed - recorded| state over forward-replayed fires of cells whose W did not
    /// change (a check of the forward re-run itself: quantization + window truncation only).
    pub state_err_sum: f64,
    pub state_err_n: u64,
}

/// One recorded dream transition (full precision).
struct DreamFire {
    cell: u32,
    h_prev: V,
    u: V,
    h: V,
}

fn dequant(h: &[i8; D]) -> V {
    std::array::from_fn(|i| h[i] as S / 127.0)
}

/// Scale `delta` so its RMS is at most 1 (no explosion along the chain); returns the RMS kept.
fn rms_clip(d: &mut V) -> S {
    let rms = norm(d) / (D as S).sqrt();
    if rms > 1.0 {
        for x in d.iter_mut() {
            *x /= rms;
        }
        1.0
    } else {
        rms
    }
}

/// Whole pass: dreams (pre-update W), real reverse replays, dream consistency, readout
/// consolidation. `from`: first ring sequence number of this pass.
pub fn wayback(eng: &mut Engine, t: u64, from: u64) -> ReplayReport {
    let p = eng.p.clone();
    let mut rep = ReplayReport::default();
    let mut anchors = std::mem::take(&mut eng.plasticity.anchors);
    if p.replay_per_pass == 0 || anchors.is_empty() {
        return rep;
    }
    let mut rng = stream(
        eng.plasticity
            .dream_seed
            .wrapping_add(eng.plasticity.passes.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
        STREAM_DREAM,
    );
    let span = u64::from(p.replay_bytes) * eng.senses.ticks_per_byte;

    // 1. Dreams with the pre-update W, recorded before any W changes.
    let mut dreams: Vec<DreamFire> = Vec::new();
    for _ in 0..p.dreams_per_pass {
        dream(eng, t, span, &mut rng, &mut dreams);
    }
    rep.dream_fires = dreams.len() as u64;

    // 2. Real replays, most surprising first (ties: earlier arrival first).
    anchors.sort_by(|a, b| b.surprise.total_cmp(&a.surprise).then(a.t.cmp(&b.t)));
    let mut touched: BTreeSet<u32> = BTreeSet::new();
    for a in &anchors {
        replay_segment(eng, a, span, &mut touched, &mut rep);
    }

    // 3. Dream consistency: pull changed cells back toward their pre-update transitions.
    for d in &dreams {
        if !touched.contains(&d.cell) {
            continue; // W unchanged: the transition is reproduced exactly
        }
        let c = d.cell as usize;
        let cells = &mut eng.g.cells;
        let now = rules::activate(&cells.w[c], &d.h_prev, &d.u);
        let delta: V = std::array::from_fn(|i| d.h[i] - now[i]);
        let g = rules::pre_activation_delta(&now, &delta);
        rules::w_step(&mut cells.w[c], &g, &d.h_prev, cells.w_norm0[c], &p);
        rep.dream_updates += 1;
    }

    // 4. Readout consolidation, forward, under the updated W: on a uniform sample of the
    //    pass's byte arrivals (`readout_uniform`), or on the surprise-ranked segments.
    if p.replay_readout {
        let ends = if p.readout_uniform {
            uniform_arrivals(eng, from, anchors.len(), &mut rng)
        } else {
            anchors
        };
        for a in &ends {
            readout_forward(eng, a, span, &touched, &mut rep);
        }
    }
    rep
}

/// `k` byte arrivals (input-cell forced fires) of this pass, uniformly without replacement,
/// as segment ends in ring order.
fn uniform_arrivals(eng: &Engine, from: u64, k: usize, rng: &mut impl Rng) -> Vec<Anchor> {
    let total = eng.ring.total();
    let from = from.max(total - eng.ring.len() as u64);
    let mut arrivals: Vec<(u64, u64)> = (from..total)
        .filter_map(|s| {
            let ev = eng.ring.get(s)?;
            (eng.g.cells.kind[ev.cell.idx as usize] == Kind::Input).then_some((s, ev.t))
        })
        .collect();
    let k = k.min(arrivals.len());
    for i in 0..k {
        let j = rng.random_range(i..arrivals.len());
        arrivals.swap(i, j);
    }
    arrivals.truncate(k);
    arrivals.sort_unstable();
    arrivals
        .into_iter()
        .map(|(seq, t)| Anchor {
            surprise: 0.0,
            t,
            seq,
            toward: ZERO,
            sources: Vec::new(),
        })
        .collect()
}

/// Ring events (all kinds) in `[t_a - span, t_a]` up to and including the anchor's own byte
/// arrival (`a.seq`), ascending; None if the segment is no longer whole in the ring.
fn segment_with_arrival(eng: &Engine, a: &Anchor, span: u64) -> Option<Vec<FireEvent>> {
    let lo = a.t.saturating_sub(span);
    let total = eng.ring.total();
    let oldest = total - eng.ring.len() as u64;
    if a.seq < oldest || a.seq >= total {
        return None;
    }
    let mut events = Vec::new();
    let mut seq = a.seq + 1;
    let mut complete = oldest == 0;
    while seq > oldest {
        seq -= 1;
        let ev = *eng.ring.get(seq).expect("retained");
        if ev.t < lo {
            complete = true;
            break;
        }
        if eng.g.cells.slots.is_live(ev.cell) {
            events.push(ev);
        }
    }
    complete.then(|| {
        events.reverse();
        events
    })
}

/// Forward re-run of one segment with recorded spike timing and recomputed states; the
/// competitive readout learns (Q only) at each recorded byte after the first.
fn readout_forward(
    eng: &mut Engine,
    a: &Anchor,
    span: u64,
    touched: &BTreeSet<u32>,
    rep: &mut ReplayReport,
) {
    let Some(events) = segment_with_arrival(eng, a, span) else {
        return;
    };
    let p = eng.p.clone();
    let window = 2 * p.t_heat.ceil() as u64;
    let oldest = eng.ring.total() - eng.ring.len() as u64;
    let mut state: Vec<V> = Vec::with_capacity(events.len());
    let mut last: BTreeMap<u32, (u64, usize)> = BTreeMap::new();
    let mut prev_arrival: Option<(u64, usize)> = None;
    let mut probs = vec![0.0; crate::senses::N_BYTES];
    for (k, ev) in events.iter().enumerate() {
        let c = ev.cell.idx;
        let cells = &eng.g.cells;
        let h = match cells.kind[c as usize] {
            Kind::Hidden => {
                let mut u = ZERO;
                for &e in &cells.ins[c as usize] {
                    let i = e as usize;
                    if let Some(&(ft, j)) = last.get(&eng.g.edges.src[i])
                        && ft < ev.t
                        && ft + window >= ev.t
                    {
                        let f = eng.g.edges.gain[i] * rules::decay(ev.t - ft - 1, p.t_heat);
                        axpy(&mut u, f, &state[j]);
                    }
                }
                let h_prev = match last.get(&c) {
                    Some(&(_, j)) => Some(state[j]),
                    None if ev.prev != NO_PREV && ev.prev >= oldest => {
                        eng.ring.get(ev.prev).map(|x| dequant(&x.h))
                    }
                    None => None,
                };
                match h_prev {
                    Some(hp) => {
                        let h = rules::activate(&cells.w[c as usize], &hp, &u);
                        if !touched.contains(&c) {
                            let rec = dequant(&ev.h);
                            rep.state_err_sum +=
                                (0..D).map(|i| (h[i] - rec[i]).abs() as f64).sum::<f64>()
                                    / D as f64;
                            rep.state_err_n += 1;
                        }
                        h
                    }
                    None => dequant(&ev.h), // no recorded start state: keep what happened
                }
            }
            Kind::Input => {
                if let Some((t0, start)) = prev_arrival {
                    readout_at_byte(
                        eng, &events, &state, start, k, t0, ev.t, c as usize, &mut probs, rep,
                    );
                }
                prev_arrival = Some((ev.t, k + 1));
                eng.senses.codes[c as usize]
            }
        };
        state.push(h);
        last.insert(c, (ev.t, k));
    }
}

/// One recorded byte `b` at `t`: input-cell traces from the recomputed hidden fires in
/// `events[start..end]` (fired in `[t0, t)`), competitive scores, softmax, Q step.
#[allow(clippy::too_many_arguments)]
fn readout_at_byte(
    eng: &mut Engine,
    events: &[FireEvent],
    state: &[V],
    start: usize,
    end: usize,
    t0: u64,
    t: u64,
    b: usize,
    probs: &mut [S],
    rep: &mut ReplayReport,
) {
    let p = &eng.p;
    let mut hot: BTreeMap<u32, V> = BTreeMap::new();
    for j in start..end {
        let ev = &events[j];
        let c = ev.cell.idx as usize;
        if eng.g.cells.kind[c] != Kind::Hidden || ev.t < t0 || ev.t >= t {
            continue;
        }
        for &e in &eng.g.cells.outs[c] {
            let i = e as usize;
            let d = eng.g.edges.dst[i];
            if eng.g.cells.kind[d as usize] == Kind::Input {
                let f = eng.g.edges.gain[i] * rules::decay(t - ev.t - 1, p.t_heat_input);
                axpy(hot.entry(d).or_insert(ZERO), f, &state[j]);
            }
        }
    }
    let mut scores = eng.senses.bias.clone();
    for (&d, u) in &hot {
        let d = d as usize;
        let cells = &eng.g.cells;
        scores[d] = rules::readout(&cells.q[d], u, &eng.senses.codes[d], scores[d]).1;
    }
    rules::softmax(&scores, probs);
    rep.readout_updates += 1;
    rep.readout_correct += u64::from(crate::senses::guess(&scores) as usize == b);
    let p = eng.p.clone();
    for (&d, u) in &hot {
        let d = d as usize;
        let e = rules::readout_error(&eng.senses.codes[d], d == b, probs[d]);
        rules::readout_step(&mut eng.g.cells.q[d], &e, u, &p);
    }
}

/// One dream: `dream_seeds` random hidden cells fire with their current state; messages spread
/// along live out-edges into hidden cells with the live rules (`activate`, heat vs frozen
/// `theta_eff`, trace decay `t_heat`), each cell at most once per dream, at most the target
/// fire budget per tick, for `span` ticks. Live state is only read.
fn dream(eng: &Engine, t: u64, span: u64, rng: &mut impl Rng, out: &mut Vec<DreamFire>) {
    let p = &eng.p;
    let cells = &eng.g.cells;
    let cap = cells.slots.capacity() as u32;
    let first = crate::senses::N_BYTES as u32;
    if cap <= first {
        return;
    }
    let budget = ((cap - first) as S * p.target_rate).round().max(1.0) as usize;
    let mut fired: BTreeSet<u32> = BTreeSet::new();
    let mut front: Vec<(u32, V)> = Vec::new();
    for _ in 0..p.dream_seeds {
        let c = rng.random_range(first..cap);
        if cells.slots.is_alive(c) && cells.kind[c as usize] == Kind::Hidden && fired.insert(c) {
            front.push((c, cells.h[c as usize]));
        }
    }
    front.sort_unstable_by_key(|x| x.0);
    // Scratch trace per receiving cell: (u, heat, tick of last update).
    let mut trace: BTreeMap<u32, (V, S, u64)> = BTreeMap::new();
    for tick in 1..=span {
        if front.is_empty() {
            break;
        }
        for (a, h) in std::mem::take(&mut front) {
            for &e in &cells.outs[a as usize] {
                let i = e as usize;
                let dst = eng.g.edges.dst[i];
                if cells.kind[dst as usize] != Kind::Hidden || fired.contains(&dst) {
                    continue;
                }
                let tr = trace.entry(dst).or_insert((ZERO, 0.0, tick));
                let f = rules::decay(tick - tr.2, p.t_heat);
                for x in tr.0.iter_mut() {
                    *x *= f;
                }
                tr.1 *= f;
                tr.2 = tick;
                let mut m = ZERO;
                axpy(&mut m, eng.g.edges.gain[i], &h);
                axpy(&mut tr.0, 1.0, &m);
                tr.1 += norm(&m);
            }
        }
        let mut ready: Vec<u32> = trace
            .iter()
            .filter(|&(&c, tr)| tr.2 == tick && tr.1 >= rules::theta_eff(&eng.peek(c as usize, t)))
            .map(|(&c, _)| c)
            .collect();
        ready.truncate(budget);
        for c in ready {
            let (u, _, _) = trace.remove(&c).expect("ready cell has a trace");
            let h_prev = cells.h[c as usize];
            let h = rules::activate(&cells.w[c as usize], &h_prev, &u);
            fired.insert(c);
            front.push((c, h));
            out.push(DreamFire {
                cell: c,
                h_prev,
                u,
                h,
            });
        }
    }
}

/// Reverse replay of the ring segment `[a.t - span, a.t)` with one-hop feedback.
fn replay_segment(
    eng: &mut Engine,
    a: &Anchor,
    span: u64,
    touched: &mut BTreeSet<u32>,
    rep: &mut ReplayReport,
) {
    let p = eng.p.clone();
    let lo = a.t.saturating_sub(span);
    let total = eng.ring.total();
    let oldest = total - eng.ring.len() as u64;
    if a.seq < oldest || a.seq > total {
        rep.lost += 1;
        return;
    }
    // Hidden fires of the segment, collected newest first (ring order is time order).
    let mut events: Vec<FireEvent> = Vec::new();
    let mut seq = a.seq;
    let mut complete = oldest == 0; // nothing ever overwritten: the segment is whole
    while seq > oldest {
        seq -= 1;
        let ev = *eng.ring.get(seq).expect("retained");
        if ev.t < lo {
            complete = true;
            break;
        }
        let c = ev.cell.idx as usize;
        if eng.g.cells.slots.is_live(ev.cell) && eng.g.cells.kind[c] == Kind::Hidden {
            events.push(ev);
        }
    }
    if !complete {
        rep.lost += 1;
        return;
    }
    events.reverse();
    rep.segments += 1;
    // Per cell, its fires in the segment: (t, index into events), ascending.
    let mut by_cell: BTreeMap<u32, Vec<(u64, usize)>> = BTreeMap::new();
    for (k, ev) in events.iter().enumerate() {
        by_cell.entry(ev.cell.idx).or_default().push((ev.t, k));
    }
    // Latest fire of `c` with lo2 <= t < hi.
    let latest = |c: u32, lo2: u64, hi: u64| -> Option<(u64, usize)> {
        by_cell
            .get(&c)?
            .iter()
            .rev()
            .find(|&&(ft, _)| ft < hi && ft >= lo2)
            .copied()
    };
    let window = 2 * p.t_heat.ceil() as u64;
    let mut delta: BTreeMap<usize, V> = BTreeMap::new();
    // Seed: the input cell's desired input change, sent to the hidden cells that fed it.
    let since_byte = a.t.saturating_sub(eng.senses.ticks_per_byte);
    for &(src, gain) in &a.sources {
        if let Some((ft, k)) = latest(src, since_byte, a.t) {
            let f = gain * rules::decay(a.t - ft - 1, p.t_heat_input);
            axpy(delta.entry(k).or_insert(ZERO), f, &a.toward);
        }
    }
    // Reverse in time: every contribution to a fire comes from a strictly later fire.
    for k in (0..events.len()).rev() {
        let Some(mut d) = delta.remove(&k) else {
            continue;
        };
        let ev = events[k];
        let rms = rms_clip(&mut d);
        if rms < p.replay_min_delta {
            continue;
        }
        rep.delta_rms_sum += rms as f64;
        let b = ev.cell.idx as usize;
        let g = rules::pre_activation_delta(&dequant(&ev.h), &d);
        // Local W step on the fired cell, from its own recorded transition h_prev -> h.
        if ev.prev != NO_PREV
            && ev.prev >= oldest
            && let Some(prev) = eng.ring.get(ev.prev)
        {
            let h_prev = dequant(&prev.h);
            let cells = &mut eng.g.cells;
            rules::w_step(&mut cells.w[b], &g, &h_prev, cells.w_norm0[b], &p);
            touched.insert(b as u32);
            rep.w_updates += 1;
        }
        // One hop back: each hidden in-neighbour that fired inside B's trace window.
        for &e in &eng.g.cells.ins[b] {
            let i = e as usize;
            let src = eng.g.edges.src[i];
            if eng.g.cells.kind[src as usize] != Kind::Hidden {
                continue;
            }
            if let Some((ft, j)) = latest(src, ev.t.saturating_sub(window).max(lo), ev.t) {
                let f =
                    p.replay_gamma * eng.g.edges.gain[i] * rules::decay(ev.t - ft - 1, p.t_heat);
                axpy(delta.entry(j).or_insert(ZERO), f, &g);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Ring;
    use crate::num::ZERO_M;
    use crate::rules::Params;
    use crate::seed::{self, SeedConfig};
    use crate::senses::{N_BYTES, Senses};
    use crate::store::{CellInit, Graph};

    fn q(x: [(usize, i8); 1]) -> [i8; D] {
        let mut h = [0; D];
        h[x[0].0] = x[0].1;
        h
    }

    /// H1 -> H2 (gain 0.5) -> input B (gain 1); W = 0.1 I on both. Hand-built ring:
    /// H1@0 (64 e2), H2@0 (64 e3), H1@1 (32 e0), H2@2 (32 e1); anchor at t = 3, toward = e1.
    #[test]
    fn two_hop_feedback_by_hand() {
        let p = Params {
            replay_per_pass: 4,
            dreams_per_pass: 0,
            ..Params::reservoir()
        };
        let mut g = Graph::default();
        let mut w = ZERO_M;
        for (i, row) in w.iter_mut().enumerate() {
            row[i] = 0.1;
        }
        let cell = |kind, w| CellInit {
            kind,
            h: ZERO,
            w,
            q: ZERO_M,
            log_theta: 0.0,
            rate: p.target_rate,
            t: 0,
        };
        for _ in 0..N_BYTES {
            g.add_cell(cell(Kind::Input, ZERO_M));
        }
        let h1 = g.add_cell(cell(Kind::Hidden, w));
        let h2 = g.add_cell(cell(Kind::Hidden, w));
        g.add_edge(h1.idx, h2.idx, 0.5, 0);
        g.add_edge(h2.idx, 66, 1.0, 0);
        let mut eng = Engine::new(p, g, Senses::new(0, 8), 64);
        eng.ring = Ring::new(64);
        for (cell, t, h, prev) in [
            (h1, 0, q([(2, 64)]), NO_PREV),
            (h2, 0, q([(3, 64)]), NO_PREV),
            (h1, 1, q([(0, 32)]), 0),
            (h2, 2, q([(1, 32)]), 1),
        ] {
            eng.ring.push(FireEvent { cell, t, h, prev });
        }
        let mut toward = ZERO;
        toward[1] = 1.0;
        eng.plasticity.anchors = vec![Anchor {
            surprise: 1.0,
            t: 3,
            seq: 4,
            toward,
            sources: vec![(h2.idx, 1.0)],
        }];
        let rep = wayback(&mut eng, 3, 0);
        assert_eq!((rep.segments, rep.w_updates, rep.lost), (1, 2, 0));
        let w2 = eng.g.cells.w[h2.idx as usize];
        let w1 = eng.g.cells.w[h1.idx as usize];
        // H2: g = (1 - (32/127)^2) e1 = 0.93651 e1; h_prev = (64/127) e3; step 0.01 / 0.25495;
        // dW[1][3] = 0.018511, then the norm cap (birth 0.1 sqrt 32) scales by 0.999478.
        assert!((w2[1][3] - 0.018501).abs() < 2e-6, "{}", w2[1][3]);
        assert!((w2[0][0] - 0.0999478).abs() < 2e-6, "{}", w2[0][0]);
        // Hop to H1: delta = gamma 0.5 * gain 0.5 * decay(0) * g = 0.234128 e1 (h1 is 0 on e1).
        // dW[1][2] = 0.039223 * 0.234128 * 0.50394 = 0.0046278, cap factor 0.999967.
        assert!((w1[1][2] - 0.0046277).abs() < 2e-6, "{}", w1[1][2]);
        assert_eq!(w1[1][3], 0.0);
        assert!(
            eng.plasticity.anchors.is_empty(),
            "anchors are consumed by the pass"
        );
    }

    fn run(p: Params, text: &[u8], ticks: u64) -> Engine {
        let mut eng = seed::build(
            &SeedConfig {
                hidden: 300,
                ..SeedConfig::default()
            },
            p,
        );
        let mut src = text.iter().copied();
        while eng.next_time().is_some_and(|t| t < ticks) {
            eng.step(&mut src);
        }
        eng
    }

    /// With W unchanged, the forward re-run reproduces the recorded states up to i8
    /// quantization and the trace-window truncation, and the readout step only touches Q.
    #[test]
    fn forward_rerun_reproduces_recorded_states() {
        let text = b"the cat sat on the mat. ".repeat(80);
        let p = Params {
            consolidate_every: 200,
            replay_per_pass: 16,
            eta_w: 0.0,
            ..Params::reservoir()
        };
        let e = run(p, &text, 8 * 1500);
        let n = e.stats.replay_state_n;
        let err = e.stats.replay_state_err / n.max(1) as f64;
        assert!(n > 100 && e.stats.replay_readout_updates > 0, "{n}");
        eprintln!("forward state error {err} over {n} fires");
        assert!(err < 0.05, "mean |h - recorded| = {err}");
    }

    #[test]
    fn zero_rate_changes_nothing_and_replay_is_deterministic() {
        let text = b"the cat sat on the mat. ".repeat(80);
        let base = Params {
            consolidate_every: 200,
            ..Params::reservoir()
        };
        let on = Params {
            replay_per_pass: 16,
            ..base.clone()
        };
        let frozen = Params {
            eta_w: 0.0,
            replay_readout: false,
            ..on.clone()
        };
        let hash = |mut e: Engine| {
            e.plasticity.anchors.clear();
            crate::observe::state_hash(&e)
        };
        let off = run(base, &text, 8 * 1500);
        assert_eq!(hash(run(frozen, &text, 8 * 1500)), hash(off));
        let a = run(on.clone(), &text, 8 * 1500);
        let w_moved = (N_BYTES..a.g.cells.slots.capacity()).any(|c| {
            a.g.cells.w[c]
                != run(
                    Params {
                        replay_per_pass: 0,
                        ..on.clone()
                    },
                    &text,
                    1,
                )
                .g
                .cells
                .w[c]
        });
        assert!(w_moved, "replay must change W");
        assert_eq!(hash(a), hash(run(on, &text, 8 * 1500)));
    }
}
