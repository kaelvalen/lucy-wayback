//! The event kernel: routes messages, fires cells, applies the rules. Work per event is
//! O(D^2 + degree) of the one cell involved; no step ever iterates over all cells.

use std::io;

use crate::buffer::{FireEvent, Ring};
use crate::num::{M, S, V, axpy, matvec_t, neg, norm, norm2, quantize_unit, scale};
use crate::observe::{Observatory, Stats};
use crate::rules::{self, Params, Scalars};
use crate::scheduler::{Payload, Scheduler};
use crate::senses::{self, Senses};
use crate::store::{Credit, EdgeId, Graph, Id, Kind, Wave};

/// Below this many fires in a tick, phase A runs inline (thread hand-off would cost more).
const PARALLEL_MIN_FIRES: usize = 8;

/// Result of phase A of one hidden fire.
struct FireOut {
    c: usize,
    q: M,
    gains: Vec<(u32, S)>,
    h: V,
    tau: S,
    e2: S,
}

#[derive(Clone)]
pub struct Engine {
    pub p: Params,
    pub g: Graph,
    pub senses: Senses,
    pub ring: Ring,
    pub stats: Stats,
    pub(crate) sched: Scheduler,
    /// Run phase A of each tick's fires on the rayon pool. Results are identical either way.
    pub parallel: bool,
    /// Structural plasticity bookkeeping (when and from where to replay the ring).
    pub plasticity: crate::plasticity::Plasticity,
    /// Decay factor tables derived from `p` (rebuild if `p` changes).
    decay: rules::Decay,
    /// Logical clock: time of the last processed event.
    pub now: u64,
}

impl Engine {
    pub fn new(p: Params, g: Graph, senses: Senses, ring_cap: usize) -> Self {
        let mut sched = Scheduler::default();
        sched.push_arrive(0);
        Self {
            decay: rules::Decay::new(&p),
            plasticity: Default::default(),
            parallel: false,
            p,
            g,
            senses,
            ring: Ring::new(ring_cap),
            stats: Stats::default(),
            sched,
            now: 0,
        }
    }

    pub fn next_time(&self) -> Option<u64> {
        self.sched.peek_t()
    }

    pub fn pending_events(&self) -> usize {
        self.sched.pending()
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.sched
    }

    /// Scalars of cell `c` as they will be at time `t`, without touching it.
    pub fn peek(&self, c: usize, t: u64) -> Scalars {
        let dt = t.saturating_sub(self.g.cells.last_t[c]);
        self.decay.scalars(self.g.cells.scalars(c), dt, &self.p)
    }

    /// Process events until the clock would reach `until` or nothing is pending.
    pub fn run(
        &mut self,
        src: &mut dyn Iterator<Item = u8>,
        until: u64,
        obs: &mut Observatory,
    ) -> io::Result<()> {
        while let Some(t) = self.next_time() {
            if t >= until {
                break;
            }
            obs.observe_until(self, t)?;
            self.step(src);
        }
        obs.finish(self)
    }

    /// Process one event. Returns false if the queue is empty.
    pub fn step(&mut self, src: &mut dyn Iterator<Item = u8>) -> bool {
        let Some(ev) = self.sched.pop() else {
            return false;
        };
        debug_assert!(ev.t >= self.now, "time went backwards");
        self.now = ev.t;
        self.stats.events += 1;
        match ev.payload {
            Payload::Arrive => self.arrive(ev.t, src),
            Payload::Deliver { edge, m, wave } => {
                self.deliver(ev.cell as usize, edge, &m, wave, ev.t)
            }
            Payload::Fire { generation } => {
                // The whole Fire phase of this tick is one batch (tick-synchronous).
                let mut batch = vec![(ev.cell, generation)];
                batch.extend(self.sched.take_fires(ev.t));
                self.stats.events += batch.len() as u64 - 1;
                if self.parallel {
                    self.fire_batch(&batch, ev.t);
                } else {
                    for &(idx, generation) in &batch {
                        self.fire(Id { idx, generation }, ev.t);
                    }
                }
            }
        }
        true
    }

    /// Lazily bring a cell's decaying state to time `t`. The only place decay happens.
    fn touch(&mut self, c: usize, t: u64) {
        let cells = &mut self.g.cells;
        let dt = t - cells.last_t[c];
        if dt == 0 {
            return;
        }
        cells.last_t[c] = t;
        let s = self.decay.scalars(cells.scalars(c), dt, &self.p);
        cells.set_scalars(c, s);
        let f = self.decay.heat(dt, &self.p);
        if f < S::EPSILON {
            // Trace has decayed below f32 resolution relative to any fresh input: drop it (frees credit memory).
            cells.clear_input(c);
        } else {
            scale(&mut cells.u[c], f);
            for cr in &mut cells.credit[c] {
                scale(&mut cr.m, f);
            }
        }
    }

    fn deliver(&mut self, c: usize, edge: EdgeId, m: &V, wave: Wave, t: u64) {
        if !self.g.edges.slots.is_live(edge) {
            return; // edge removed while the message was in flight
        }
        self.touch(c, t);
        let cells = &mut self.g.cells;
        axpy(&mut cells.u[c], 1.0, m);
        cells.heat[c] += norm(m);
        match cells.credit[c].iter_mut().find(|cr| cr.edge == edge) {
            Some(cr) => axpy(&mut cr.m, 1.0, m),
            None => cells.credit[c].push(Credit { edge, m: *m }),
        }
        cells.wave[c] = cells.wave[c].merge(wave);
        match cells.kind[c] {
            // Input cells never fire from prediction: their heat is the TRACE read at arrival.
            Kind::Input => self.senses.mark_hot(c),
            Kind::Hidden => {
                if !cells.fire_pending[c] && cells.heat[c] >= rules::theta_eff(&cells.scalars(c)) {
                    cells.fire_pending[c] = true;
                    let generation = cells.slots.id(c as u32).generation;
                    self.sched.push_fire(t, c as u32, generation);
                }
            }
        }
    }

    /// Sequential hidden fire (the reference path; `fire_batch` must match it bit for bit).
    fn fire(&mut self, id: Id, t: u64) {
        if !self.g.cells.slots.is_live(id) {
            return;
        }
        let c = id.idx as usize;
        self.touch(c, t);
        let cells = &mut self.g.cells;
        cells.fire_pending[c] = false;
        // Predict this cycle's input from the state the last fire produced (h, before updating it).
        let e = rules::prediction_error(&cells.u[c], &cells.q[c], &cells.h[c]);
        rules::nlms(&mut cells.q[c], &e, &cells.h[c], &self.p);
        // Hidden credit: inputs move toward the prediction (toward = -e).
        self.learn_in_gains(c, &neg(&e));
        let cells = &mut self.g.cells;
        let h = rules::activate(&cells.w[c], &cells.h[c], &cells.u[c]);
        cells.h_prev[c] = cells.h[c];
        cells.h[c] = h;
        let e2 = norm2(&e);
        cells.tau[c] = rules::ema(cells.tau[c], e2, self.p.alpha_tau);
        let wave = cells.wave[c];
        self.spike(c, t, wave);
        self.stats.on_hidden_fire(e2, wave);
    }

    /// Parallel path: fire every cell of one tick's Fire phase (cells in ascending order).
    ///
    /// Phase A computes each cell's fire from the state at the start of the phase: tension,
    /// predictor update, new in-gains, new state. Nothing a fire writes is read by another fire
    /// of the same tick (a cell writes its own state and its own in-edges' gains; out-edge
    /// vitality is not read), so phase A runs in parallel when `parallel` is set. Phase B then
    /// applies the results and spikes strictly in cell order, so every message reads out-edge
    /// gains exactly as a sequential engine would: bit-identical for any thread count.
    fn fire_batch(&mut self, batch: &[(u32, u32)], t: u64) {
        let live: Vec<usize> = batch
            .iter()
            .filter(|&&(idx, generation)| self.g.cells.slots.is_live(Id { idx, generation }))
            .map(|&(idx, _)| idx as usize)
            .collect();
        for &c in &live {
            self.touch(c, t); // a no-op: the delivery that scheduled the fire touched it at t
        }
        let outs: Vec<FireOut> = if self.parallel && live.len() >= PARALLEL_MIN_FIRES {
            use rayon::prelude::*;
            live.par_iter().map(|&c| self.fire_compute(c)).collect()
        } else {
            live.iter().map(|&c| self.fire_compute(c)).collect()
        };
        for o in outs {
            let c = o.c;
            let Graph { cells, edges } = &mut self.g;
            cells.fire_pending[c] = false;
            cells.q[c] = o.q;
            for &(e, g) in &o.gains {
                edges.gain[e as usize] = g;
            }
            cells.h_prev[c] = cells.h[c];
            cells.h[c] = o.h;
            cells.tau[c] = o.tau;
            let wave = cells.wave[c];
            self.spike(c, t, wave);
            self.stats.on_hidden_fire(o.e2, wave);
        }
    }

    /// Phase A of one hidden fire: pure function of the state at the start of the Fire phase.
    fn fire_compute(&self, c: usize) -> FireOut {
        let (cells, edges, p) = (&self.g.cells, &self.g.edges, &self.p);
        // Predict this cycle's input from the state the last fire produced (h, before updating it).
        let e = rules::prediction_error(&cells.u[c], &cells.q[c], &cells.h[c]);
        let mut q = cells.q[c];
        rules::nlms(&mut q, &e, &cells.h[c], p);
        // Hidden credit: inputs move toward the prediction (toward = -e), then synaptic scaling;
        // same operations, in the same order, as `learn_in_gains`.
        let mut gains = Vec::new();
        if !cells.credit[c].is_empty() {
            let toward = neg(&e);
            let ins = &cells.ins[c];
            let mut local: Vec<S> = ins.iter().map(|&i| edges.gain[i as usize]).collect();
            for cr in &cells.credit[c] {
                if edges.slots.is_live(cr.edge) {
                    let k = ins
                        .iter()
                        .position(|&i| i == cr.edge.idx)
                        .expect("live in-edge");
                    local[k] += rules::credit_step(&toward, &cr.m, p.eta_g);
                }
            }
            let n2: S = local.iter().map(|g| g.powi(2)).sum();
            let k = rules::scale_factor(n2, cells.gain_norm0[c]);
            gains.extend(ins.iter().zip(&local).map(|(&i, &g)| (i, g * k)));
        }
        let h = rules::activate(&cells.w[c], &cells.h[c], &cells.u[c]);
        let e2 = norm2(&e);
        FireOut {
            c,
            q,
            gains,
            h,
            tau: rules::ema(cells.tau[c], e2, p.alpha_tau),
            e2,
        }
    }

    /// Credit every in-edge that contributed this cycle, then synaptic scaling (fix 4). O(in-degree).
    fn learn_in_gains(&mut self, c: usize, toward: &V) {
        let Graph { cells, edges } = &mut self.g;
        if cells.credit[c].is_empty() {
            return;
        }
        for cr in &cells.credit[c] {
            if edges.slots.is_live(cr.edge) {
                edges.gain[cr.edge.idx as usize] += rules::credit_step(toward, &cr.m, self.p.eta_g);
            }
        }
        let n2: S = cells.ins[c]
            .iter()
            .map(|&e| edges.gain[e as usize].powi(2))
            .sum();
        let k = rules::scale_factor(n2, cells.gain_norm0[c]);
        for &e in &cells.ins[c] {
            edges.gain[e as usize] *= k;
        }
    }

    /// Fire bookkeeping, reset of the integration window, and broadcast of `gain * h` to t+1.
    fn spike(&mut self, c: usize, t: u64, wave: Wave) {
        let Engine {
            g, sched, ring, p, ..
        } = self;
        let mut s = g.cells.scalars(c);
        rules::on_fire(&mut s, p);
        g.cells.set_scalars(c, s);
        g.cells.clear_input(c);
        let h = g.cells.h[c];
        let edges = &mut g.edges;
        // h is stored once; each message carries its edge's gain at send time (m = h * gain).
        let state = sched.push_state(t + 1, &h);
        for &e in &g.cells.outs[c] {
            let i = e as usize;
            // Edges into input cells earn vitality from reality (at arrival), not from use.
            if g.cells.kind[edges.dst[i] as usize] != Kind::Input {
                edges.vitality[i] = rules::edge_vitality(edges.vitality[i], t - edges.last_t[i], p);
                edges.last_t[i] = t;
            }
            let id = edges.slots.id(e);
            sched.push_deliver(
                t + 1,
                edges.dst[i],
                id,
                edges.gain[i],
                state,
                wave.next_hop(),
            );
        }
        let seq = ring.total();
        ring.push(FireEvent {
            cell: g.cells.slots.id(c as u32),
            t,
            h: quantize_unit(&h),
            prev: g.cells.last_seq[c],
        });
        g.cells.last_seq[c] = seq;
        self.stats.on_fire(c, t);
    }

    /// A byte arrives: read the prediction (hot input cells), let reality teach every hot cell and
    /// the true cell, then force the true cell to fire. O(hot) — no pass over all 256.
    fn arrive(&mut self, t: u64, src: &mut dyn Iterator<Item = u8>) {
        let Some(byte) = src.next() else { return };
        let b = byte as usize;
        let every = u64::from(self.p.consolidate_every);
        if every > 0 && self.stats.bytes_total > 0 && self.stats.bytes_total.is_multiple_of(every) {
            let rep = crate::plasticity::consolidate(self, t);
            self.stats.on_consolidation(&rep);
        }
        let mut touched = self.senses.take_hot();
        let truth_hot = touched.contains(&(b as u32));
        self.stats.on_coverage(touched.len(), truth_hot);
        if !truth_hot {
            touched.push(b as u32);
        }
        for &c in &touched {
            self.touch(c as usize, t);
        }
        // Readout: every input cell scores a logit for "my byte comes next". A cold cell (u = 0)
        // scores exactly its bias; a hot one adds dot(Q u, code) / D.
        let mut scores = std::mem::take(&mut self.senses.scores);
        scores.copy_from_slice(&self.senses.bias);
        let preds: Vec<(usize, V)> = touched
            .iter()
            .map(|&c| {
                let c = c as usize;
                let cells = &self.g.cells;
                let (pred, score) =
                    rules::readout(&cells.q[c], &cells.u[c], &self.senses.codes[c], scores[c]);
                scores[c] = score;
                (c, pred)
            })
            .collect();
        let guess = senses::guess(&scores);
        // Competitive readout (lateral inhibition across the sheet): p = softmax(scores).
        let mut probs = std::mem::take(&mut self.senses.probs);
        rules::softmax(&scores, &mut probs);
        self.stats.on_byte(guess == b as u32, &scores, probs[b], b);

        // Reality teaches with the softmax error y - p: the true cell up, each hot wrong cell
        // down by its probability. Q and in-gains only move for touched cells (a cold cell has
        // u = 0, so its Q and in-gain gradients are exactly 0); every bias takes y - p.
        for (c, _pred) in preds {
            let truth = c == b;
            let e = rules::readout_error(&self.senses.codes[c], truth, probs[c]);
            let cells = &mut self.g.cells;
            cells.tau[c] = rules::ema(cells.tau[c], norm2(&e), self.p.alpha_tau);
            // Input credit: move inputs so the cell's own readout gets closer to reality.
            let toward = matvec_t(&cells.q[c], &e);
            let u = cells.u[c];
            rules::readout_step(&mut cells.q[c], &e, &u, &self.p);
            self.learn_in_gains(c, &toward);
            if truth {
                self.plasticity.observe_error(b, 1.0 - probs[b], &self.p);
                if self.p.replay_per_pass > 0 {
                    let (cells, edges) = (&self.g.cells, &self.g.edges);
                    let sources = cells.credit[b]
                        .iter()
                        .filter(|cr| edges.slots.is_live(cr.edge))
                        .map(|cr| cr.edge.idx as usize)
                        .filter(|&i| cells.kind[edges.src[i] as usize] == Kind::Hidden)
                        .map(|i| (edges.src[i], edges.gain[i]))
                        .collect();
                    let anchor = crate::replay::Anchor {
                        surprise: -probs[b].max(1e-30).ln(),
                        t,
                        seq: self.ring.total(),
                        toward,
                        sources,
                    };
                    let cap = self.p.replay_per_pass as usize;
                    crate::replay::offer(&mut self.plasticity.anchors, anchor, cap);
                }
                // In-edges that fed the cell whose byte really arrived earn vitality.
                let Graph { cells, edges } = &mut self.g;
                for cr in &cells.credit[c] {
                    if edges.slots.is_live(cr.edge) {
                        let i = cr.edge.idx as usize;
                        edges.vitality[i] =
                            rules::edge_vitality(edges.vitality[i], t - edges.last_t[i], &self.p);
                        edges.last_t[i] = t;
                    }
                }
            }
            self.g.cells.clear_input(c);
        }
        for (c, (bias, &pc)) in self.senses.bias.iter_mut().zip(&probs).enumerate() {
            *bias = rules::bias_step(*bias, c == b, pc, &self.p);
        }
        self.senses.scores = scores;
        self.senses.probs = probs;

        let cells = &mut self.g.cells;
        cells.h_prev[b] = cells.h[b];
        cells.h[b] = self.senses.codes[b];
        self.spike(
            b,
            t,
            Wave {
                origin: self.stats.bytes_total,
                depth: 0,
            },
        );
        self.sched.push_arrive(t + self.senses.ticks_per_byte);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::{D, ZERO, ZERO_M, basis, sub};
    use crate::observe::{Observatory, state_hash};
    use crate::scheduler::Phase;
    use crate::seed::{self, SeedConfig};
    use crate::senses::N_BYTES;
    use crate::store::CellInit;

    const A: u32 = b'A' as u32;
    const B: u32 = b'B' as u32;
    const C: u32 = b'C' as u32;

    /// 256 inputs + one hidden cell H: A -> H (0.5), C -> H (0.5, silent), H -> B (1.0).
    fn tiny() -> (Engine, usize) {
        tiny_with(Params::default())
    }

    fn tiny_with(p: Params) -> (Engine, usize) {
        let mut g = Graph::default();
        let cell = |kind, log_theta| CellInit {
            kind,
            h: ZERO,
            w: ZERO_M,
            q: ZERO_M,
            log_theta,
            rate: p.target_rate,
            t: 0,
        };
        for _ in 0..N_BYTES {
            g.add_cell(cell(Kind::Input, 0.0));
        }
        let hid = g.add_cell(cell(Kind::Hidden, 0.25f32.ln())).idx;
        g.add_edge(A, hid, 0.5, 0);
        g.add_edge(C, hid, 0.5, 0);
        g.add_edge(hid, B, 1.0, 0);
        g.fix_gain_norms(0..g.cells.slots.capacity());
        let mut senses = Senses::new(0, 8);
        senses.codes[A as usize] = basis(0);
        senses.codes[B as usize] = basis(1);
        (Engine::new(p, g, senses, 64), hid as usize)
    }

    fn run_to(eng: &mut Engine, src: &mut dyn Iterator<Item = u8>, last_t: u64) {
        while eng.next_time().is_some_and(|t| t <= last_t) {
            eng.step(src);
        }
    }

    /// Q predicts from the state produced by the last fire (h), not the one before (h_prev).
    #[test]
    fn prediction_uses_current_state() {
        let (mut eng, hid) = tiny();
        eng.g.cells.h[hid] = basis(2);
        eng.g.cells.h_prev[hid] = basis(5); // must be ignored
        eng.g.cells.q[hid][0][2] = 0.4;
        let mut src = b"A".iter().copied();
        run_to(&mut eng, &mut src, 1);
        // u = 0.5 e0, Q h = 0.4 e0 -> e = 0.1 e0, tau = 0.05 * 0.01.
        assert!((eng.g.cells.tau[hid] - 0.0005).abs() < 1e-8);
        // NLMS on h = e2: Q[0][2] += 0.1 / (1e-3 + 1) * 0.1; nothing learned on the e5 column.
        let q = eng.g.cells.q[hid];
        assert!((q[0][2] - (0.4 + 0.01 / 1.001)).abs() < 1e-7, "{}", q[0][2]);
        assert_eq!(q[0][5], 0.0);
        // Then h_prev = old h, h = tanh(W h + u) with W = 0.
        assert_eq!(eng.g.cells.h_prev[hid], basis(2));
        assert!((eng.g.cells.h[hid][0] - 0.5f32.tanh()).abs() < 1e-7);
    }

    #[test]
    fn hand_computed_two_hop_prediction() {
        let (mut eng, hid) = tiny();
        let mut src = b"AB".iter().copied();
        run_to(&mut eng, &mut src, 8);

        // Timing: A forced at 0, H fires at 1 (one hop), B forced at 8 (byte 2 of K=8).
        let fires: Vec<(u32, u64)> = eng.ring.iter().map(|e| (e.cell.idx, e.t)).collect();
        assert_eq!(fires, vec![(A, 0), (hid as u32, 1), (B, 8)]);

        // H: u = 0.5 e0, Q = 0, h_prev = 0 -> e = 0.5 e0, |e|^2 = 0.25, tau = 0.05 * 0.25.
        assert!((eng.g.cells.tau[hid] - 0.0125).abs() < 1e-7);
        assert_eq!(
            eng.g.cells.q[hid], ZERO_M,
            "NLMS with h_prev = 0 is a no-op"
        );
        // h = tanh(0.5) e0, quantized: round(0.46212 * 127) = 59.
        assert_eq!(eng.ring.get(1).map(|e| e.h[0]), Some(59));
        // Credit on A->H: 0.5 - 0.05 * 0.25 / 32 = 0.499609375; C->H untouched (0.5);
        // scaling back to norm sqrt(0.5): k = 0.70710678 / 0.70683062 = 1.0003907.
        let g = &eng.g.edges.gain;
        assert!((g[0] - 0.4998046).abs() < 1e-6, "{}", g[0]);
        assert!((g[1] - 0.5001953).abs() < 1e-6, "{}", g[1]);

        // Byte A: every logit is 0 -> uniform softmax, guess byte 0 (wrong), 8 bits. All biases
        // step by eta (y - 1/256): A up to eta*255/256, every other one down to -eta/256.
        // Byte B: logits = biases (Q_B = 0, so B's trace adds nothing yet) -> guess A (wrong),
        // p_B = e^(-eta/256) / (e^(eta*255/256) + 255 e^(-eta/256)) = 1 / (255 + e^eta).
        let eta = eng.p.eta_bias;
        let p_b = 1.0 / (255.0 + eta.exp());
        assert_eq!((eng.stats.bytes_total, eng.stats.correct), (2, 0));
        let s_b = eng.stats.readout_bits_sum - 8.0;
        assert!((s_b + (p_b as f64).log2()).abs() < 1e-4, "{s_b}");
        let p_a = eta.exp() * p_b;
        let (ba, bb) = (eng.senses.bias[A as usize], eng.senses.bias[B as usize]);
        assert!(
            (ba - (eta * 255.0 / 256.0 - eta * p_a)).abs() < 1e-7,
            "{ba}"
        );
        assert!(
            (bb - (-eta / 256.0 + eta * (1.0 - p_b))).abs() < 1e-7,
            "{bb}"
        );
        // B's readout error: e = (1 - p_B) code_B = (1 - p_B) e1 -> tau = 0.05 (1 - p_B)^2.
        // Readout step on u_B = tanh(0.5) e^(-6/4) e0 = 0.103114 e0 (|u|^2 = 0.0106325):
        // Q_B[1][0] = 0.3 / (1 + 0.0106325) * 0.103114 * (1 - p_B) = 0.0306087 (1 - p_B).
        assert!((eng.g.cells.tau[B as usize] - 0.05 * (1.0 - p_b).powi(2)).abs() < 1e-7);
        let q = eng.g.cells.q[B as usize][1][0];
        assert!((q - 0.0306087 * (1.0 - p_b)).abs() < 1e-6, "{q}");
        assert_eq!(g[2], 1.0, "toward = Q_B^T e = 0: gain unchanged");
        assert_eq!(
            eng.g.cells.heat[B as usize], 0.0,
            "prediction window cleared on arrival"
        );
        // Hop depth of the window between A and B: H fired one hop from byte 1; no reverberation.
        assert_eq!(
            (
                eng.stats.depth_sum,
                eng.stats.depth_windows,
                eng.stats.reverb_fires
            ),
            (1, 1, 0)
        );
    }

    /// Deterministic bigram A -> B: the readout must learn to call both bytes.
    #[test]
    fn readout_learns_a_bigram() {
        let p = Params {
            target_rate: 0.1,
            adapt_step: 0.0,
            ..Params::default()
        };
        let (mut eng, _) = tiny_with(p);
        let text = b"AB".repeat(500);
        let mut src = text.iter().copied();
        run_to(&mut eng, &mut src, 8 * 800 - 1);
        let before = eng.stats.correct;
        run_to(&mut eng, &mut src, 8 * 1000);
        let late = eng.stats.correct - before;
        assert!(late >= 195, "late accuracy {late}/200");
    }

    #[test]
    fn u_equals_sum_of_credit_under_decay() {
        let mut eng = seed::build(
            &SeedConfig {
                hidden: 64,
                ..SeedConfig::default()
            },
            Params::default(),
        );
        let c = 300;
        eng.g.cells.log_theta[c] = 50.0; // never fires
        let ins = eng.g.cells.ins[c].clone();
        for (k, t) in [1u64, 1, 2, 5, 5, 9, 12].into_iter().enumerate() {
            let e = eng.g.edges.slots.id(ins[k % ins.len()]);
            let m: V = std::array::from_fn(|i| ((i * 7 + k) % 5) as S - 2.0);
            eng.deliver(c, e, &m, Wave::default(), t);
        }
        let mut sum = ZERO;
        for cr in &eng.g.cells.credit[c] {
            axpy(&mut sum, 1.0, &cr.m);
        }
        let diff = norm(&sub(&sum, &eng.g.cells.u[c]));
        assert!(diff < 1e-5 * norm(&eng.g.cells.u[c]).max(1.0), "{diff}");
        assert!(eng.g.cells.credit[c].len() <= ins.len());
    }

    #[test]
    fn at_most_one_fire_per_cell_per_tick() {
        let (mut eng, hid) = tiny();
        eng.sched = Scheduler::default();
        let e = eng.g.edges.slots.id(0);
        for _ in 0..5 {
            eng.deliver(hid, e, &[1.0; D], Wave::default(), 3);
        }
        let fires = eng
            .sched
            .sorted()
            .iter()
            .filter(|ev| ev.phase == Phase::Fire)
            .count();
        assert_eq!(fires, 1);
    }

    #[test]
    fn peek_matches_touch() {
        let mut eng = seed::build(
            &SeedConfig {
                hidden: 64,
                ..SeedConfig::default()
            },
            Params::default(),
        );
        let c = 290;
        eng.g.cells.heat[c] = 3.0;
        eng.g.cells.adapt[c] = 2.0;
        let peeked = eng.peek(c, 777);
        eng.touch(c, 777);
        assert_eq!(peeked, eng.g.cells.scalars(c));
    }

    fn small_run(every: u64, seed: u64) -> (u64, u64) {
        let cfg = SeedConfig {
            seed,
            hidden: 512,
            ..SeedConfig::default()
        };
        let mut eng = seed::build(&cfg, Params::default());
        let text = b"the cat sat on the mat. the dog sat on the log. ".repeat(20);
        let mut obs = Observatory::new(&eng, every, seed, None);
        eng.run(&mut text.iter().copied(), 6000, &mut obs).unwrap();
        (state_hash(&eng), obs.metrics_hash())
    }

    /// Tick-synchronous parallel fire phase: same state, bit for bit, as the sequential one.
    #[test]
    fn parallel_fire_phase_is_bit_identical() {
        let run = |parallel: bool| {
            let cfg = SeedConfig {
                hidden: 3000,
                ..SeedConfig::default()
            };
            let mut eng = seed::build(&cfg, Params::default());
            eng.parallel = parallel;
            let text = b"where's the doggie ? there's the doggie . ".repeat(30);
            let mut obs = Observatory::new(&eng, 1000, 0, None);
            eng.run(&mut text.iter().copied(), 6000, &mut obs).unwrap();
            (state_hash(&eng), obs.metrics_hash(), eng.stats.events)
        };
        assert_eq!(run(false), run(true));
    }

    #[test]
    fn deterministic_and_observation_free() {
        let a = small_run(500, 3);
        assert_eq!(a, small_run(500, 3));
        assert_eq!(
            a.0,
            small_run(77, 3).0,
            "observing must not perturb the dynamics"
        );
        assert_ne!(a.0, small_run(500, 4).0);
    }

    #[test]
    fn dormant_cells_cost_nothing() {
        let cfg = SeedConfig {
            hidden: 512,
            ..SeedConfig::default()
        };
        let text = b"abracadabra ".repeat(30);
        let mut counts = Vec::new();
        for extra in [0, 2048] {
            let mut eng = seed::build(&cfg, Params::default());
            seed::add_dormant(&mut eng, &cfg, extra);
            let mut obs = Observatory::new(&eng, 1000, 0, None);
            eng.run(&mut text.iter().copied(), 3000, &mut obs).unwrap();
            let dormant_touched = (256 + 512..256 + 512 + extra)
                .filter(|&c| eng.g.cells.last_t[c] != 0)
                .count();
            assert_eq!(dormant_touched, 0);
            counts.push((eng.stats.events, eng.ring.total()));
        }
        assert_eq!(
            counts[0], counts[1],
            "same activity regardless of total cells"
        );
    }
}
