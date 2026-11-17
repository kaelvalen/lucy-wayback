//! The event kernel: routes messages, fires cells, applies the rules. Work per event is
//! O(D^2 + degree) of the one cell involved; no step ever iterates over all cells.

use std::io;

use crate::buffer::{FireEvent, Ring};
use crate::num::{S, V, ZERO, axpy, neg, norm, norm2, quantize_unit, scale, sub};
use crate::observe::{Observatory, Stats};
use crate::rules::{self, Params, Scalars};
use crate::scheduler::{Payload, Phase, Scheduler};
use crate::senses::{self, N_BYTES, Senses};
use crate::store::{Credit, EdgeId, Graph, Id, Kind, Wave};

pub struct Engine {
    pub p: Params,
    pub g: Graph,
    pub senses: Senses,
    pub ring: Ring,
    pub stats: Stats,
    pub(crate) sched: Scheduler,
    /// Logical clock: time of the last processed event.
    pub now: u64,
}

impl Engine {
    pub fn new(p: Params, g: Graph, senses: Senses, ring_cap: usize) -> Self {
        let mut sched = Scheduler::default();
        sched.push(0, Phase::Sense, 0, Payload::Arrive);
        Self {
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
        rules::decay_scalars(self.g.cells.scalars(c), dt, &self.p)
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
            Payload::Fire { generation } => self.fire(
                Id {
                    idx: ev.cell,
                    generation,
                },
                ev.t,
            ),
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
        let s = rules::decay_scalars(cells.scalars(c), dt, &self.p);
        cells.set_scalars(c, s);
        let f = rules::decay(dt, self.p.t_heat);
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
                    self.sched
                        .push(t, Phase::Fire, c as u32, Payload::Fire { generation });
                }
            }
        }
    }

    fn fire(&mut self, id: Id, t: u64) {
        if !self.g.cells.slots.is_live(id) {
            return;
        }
        let c = id.idx as usize;
        self.touch(c, t);
        let cells = &mut self.g.cells;
        cells.fire_pending[c] = false;
        let e = rules::prediction_error(&cells.u[c], &cells.q[c], &cells.h_prev[c]);
        rules::nlms(&mut cells.q[c], &e, &cells.h_prev[c], &self.p);
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
        for &e in &g.cells.outs[c] {
            let i = e as usize;
            edges.vitality[i] = rules::edge_vitality(edges.vitality[i], t - edges.last_t[i], p);
            edges.last_t[i] = t;
            let mut m = h;
            scale(&mut m, edges.gain[i]);
            let payload = Payload::Deliver {
                edge: edges.slots.id(e),
                m,
                wave: wave.next_hop(),
            };
            sched.push(t + 1, Phase::Deliver, edges.dst[i], payload);
        }
        ring.push(FireEvent {
            cell: g.cells.slots.id(c as u32),
            t,
            h: quantize_unit(&h),
        });
        self.stats.on_fire(c, t);
    }

    /// A byte arrives: read the prediction (hot input cells), let reality teach every hot cell and
    /// the true cell, then force the true cell to fire. O(hot) — no pass over all 256.
    fn arrive(&mut self, t: u64, src: &mut dyn Iterator<Item = u8>) {
        let Some(byte) = src.next() else { return };
        let b = byte as usize;
        let mut touched = self.senses.take_hot();
        if !touched.contains(&(b as u32)) {
            touched.push(b as u32);
        }
        for &c in &touched {
            self.touch(c as usize, t);
        }
        let heats: Vec<(u32, S)> = touched
            .iter()
            .map(|&c| (c, self.g.cells.heat[c as usize]))
            .collect();
        let guess = senses::guess(&heats);
        let surprise = senses::surprise_bits(&heats, N_BYTES, b as u32);
        self.stats.on_byte(guess == Some(b as u32), surprise);

        let code = self.senses.codes[b];
        for &c in &touched {
            let c = c as usize;
            let target = if c == b { code } else { ZERO };
            let err = sub(&target, &self.g.cells.u[c]);
            self.g.cells.tau[c] = rules::ema(self.g.cells.tau[c], norm2(&err), self.p.alpha_tau);
            // Input credit: inputs move toward reality (toward = target - u).
            self.learn_in_gains(c, &err);
            self.g.cells.clear_input(c);
        }

        let cells = &mut self.g.cells;
        cells.h_prev[b] = cells.h[b];
        cells.h[b] = code;
        self.spike(
            b,
            t,
            Wave {
                origin: self.stats.bytes_total,
                depth: 0,
            },
        );
        self.sched.push(
            t + self.senses.ticks_per_byte,
            Phase::Sense,
            0,
            Payload::Arrive,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::{D, ZERO_M, basis};
    use crate::observe::{Observatory, state_hash};
    use crate::seed::{self, SeedConfig};
    use crate::store::CellInit;

    const A: u32 = b'A' as u32;
    const B: u32 = b'B' as u32;
    const C: u32 = b'C' as u32;

    /// 256 inputs + one hidden cell H: A -> H (0.5), C -> H (0.5, silent), H -> B (1.0).
    fn tiny() -> (Engine, usize) {
        let p = Params::default();
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

        // Byte B: the trace on B is the prediction -> correct guess, near-zero surprise.
        assert_eq!((eng.stats.bytes_total, eng.stats.correct), (2, 1));
        let s_b = eng.stats.surprise_sum - 8.0; // first byte had nothing hot: 8 bits
        assert!((0.0..1e-4).contains(&s_b), "{s_b}");
        // B's error: u_B = tanh(0.5) e^(-6/4) e0 = 0.103112 e0, code_B = e1
        // -> |e|^2 = 1 + 0.103112^2, tau = 0.05 * 1.0106321.
        assert!(
            (eng.g.cells.tau[B as usize] - 0.0505316).abs() < 1e-6,
            "{}",
            eng.g.cells.tau[B as usize]
        );
        assert!(
            (g[2] - 1.0).abs() < 1e-6,
            "single in-edge: scaling restores the birth norm"
        );
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
