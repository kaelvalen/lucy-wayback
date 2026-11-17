//! Event queue on a logical clock: a bucketed time wheel, single-threaded and deterministic.
//!
//! All dynamics delays are short (fires at t, messages at t+1, bytes every K ticks), so events
//! live in a ring of per-tick buckets; anything beyond the wheel (a long byte interval) waits in
//! a small overflow heap. Within a tick the order is fixed: senses, then deliveries stably sorted
//! by destination cell (= insertion order per cell), then fires sorted by cell. This is exactly
//! the (t, phase, cell, insertion) order of a global priority queue, at O(1) per event.
//!
//! Messages are small: the sender's state `h` is stored once per fire in the destination tick's
//! arena, and each message holds an index plus the gain at send time; `m = h * gain` is formed
//! at delivery with the same float operation the sender would have used.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::num::{S, V, scale};
use crate::store::{EdgeId, Wave};

/// Order of work inside one tick: senses, then every delivery, then fires. A cell therefore
/// integrates all messages of tick t before deciding to fire at t; its output arrives at t+1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    Sense,
    Deliver,
    Fire,
}

/// An event as processed by the kernel (and as listed for hashing/inspection).
#[derive(Clone, Debug)]
pub enum Payload {
    /// Next corpus byte arrives.
    Arrive,
    /// Message `m` over `edge`, carrying the ancestry of the activity that produced it.
    Deliver { edge: EdgeId, m: V, wave: Wave },
    /// Fire the cell if it still has this generation.
    Fire { generation: u32 },
}

#[derive(Clone, Debug)]
pub struct Event {
    pub t: u64,
    pub phase: Phase,
    pub cell: u32,
    pub payload: Payload,
}

#[derive(Clone, Copy, Debug)]
struct Msg {
    dst: u32,
    edge: EdgeId,
    gain: S,
    h: u32,
    wave: Wave,
}

#[derive(Default)]
struct Bucket {
    sense: usize,
    deliver: Vec<Msg>,
    /// Delivery order: `(dst << 32) | insertion index`, sorted once when the tick starts delivering.
    order: Vec<u64>,
    delivered: usize,
    fire: Vec<(u32, u32)>,
    fired: usize,
    arena: Vec<V>,
}

impl Bucket {
    fn remaining(&self) -> bool {
        self.sense > 0 || self.delivered < self.deliver.len() || self.fired < self.fire.len()
    }

    fn message(&self, m: &Msg) -> V {
        let mut v = self.arena[m.h as usize];
        scale(&mut v, m.gain);
        v
    }
}

const WHEEL: usize = 64;

pub struct Scheduler {
    buckets: Vec<Bucket>,
    /// Lowest tick that may still hold events (the bucket being consumed).
    now: u64,
    in_wheel: usize,
    /// Byte arrivals too far ahead for the wheel.
    overflow: BinaryHeap<Reverse<u64>>,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self {
            buckets: (0..WHEEL).map(|_| Bucket::default()).collect(),
            now: 0,
            in_wheel: 0,
            overflow: BinaryHeap::new(),
        }
    }
}

impl Scheduler {
    fn bucket(&mut self, t: u64) -> &mut Bucket {
        assert!(
            t >= self.now && t < self.now + WHEEL as u64,
            "event at {t} outside wheel at {}",
            self.now
        );
        &mut self.buckets[t as usize % WHEEL]
    }

    pub fn push_arrive(&mut self, t: u64) {
        if t < self.now + WHEEL as u64 {
            self.bucket(t).sense += 1;
            self.in_wheel += 1;
        } else {
            self.overflow.push(Reverse(t));
        }
    }

    /// Store a fired cell's state for messages delivered at `t`; returns its arena slot.
    pub fn push_state(&mut self, t: u64, h: &V) -> u32 {
        let b = self.bucket(t);
        b.arena.push(*h);
        (b.arena.len() - 1) as u32
    }

    pub fn push_deliver(
        &mut self,
        t: u64,
        dst: u32,
        edge: EdgeId,
        gain: S,
        state: u32,
        wave: Wave,
    ) {
        let b = self.bucket(t);
        debug_assert!(
            b.delivered == 0,
            "delivery added to a tick already being delivered"
        );
        b.deliver.push(Msg {
            dst,
            edge,
            gain,
            h: state,
            wave,
        });
        self.in_wheel += 1;
    }

    pub fn push_fire(&mut self, t: u64, cell: u32, generation: u32) {
        let b = self.bucket(t);
        debug_assert!(b.fired == 0, "fire added to a tick already firing");
        b.fire.push((cell, generation));
        self.in_wheel += 1;
    }

    pub fn pop(&mut self) -> Option<Event> {
        loop {
            if self.in_wheel == 0 {
                let Reverse(t) = self.overflow.pop()?;
                // Jump: recycle every bucket the clock skips over (they may hold spent state).
                for skipped in self.now..t.min(self.now + WHEEL as u64) {
                    self.recycle(skipped);
                }
                self.now = self.now.max(t);
                self.push_arrive(t);
            }
            let t = self.now;
            let b = &mut self.buckets[t as usize % WHEEL];
            if b.sense > 0 {
                b.sense -= 1;
                self.in_wheel -= 1;
                return Some(Event {
                    t,
                    phase: Phase::Sense,
                    cell: 0,
                    payload: Payload::Arrive,
                });
            }
            if b.delivered < b.deliver.len() {
                if b.delivered == 0 {
                    // Sort 8-byte keys, not 40-byte messages; keys are unique, so this equals a
                    // stable sort by destination (per-cell insertion order kept).
                    b.order.clear();
                    b.order.extend(
                        b.deliver
                            .iter()
                            .enumerate()
                            .map(|(i, m)| (u64::from(m.dst) << 32) | i as u64),
                    );
                    b.order.sort_unstable();
                }
                let msg = b.deliver[b.order[b.delivered] as u32 as usize];
                b.delivered += 1;
                self.in_wheel -= 1;
                let m = b.message(&msg);
                return Some(Event {
                    t,
                    phase: Phase::Deliver,
                    cell: msg.dst,
                    payload: Payload::Deliver {
                        edge: msg.edge,
                        m,
                        wave: msg.wave,
                    },
                });
            }
            if b.fired < b.fire.len() {
                if b.fired == 0 {
                    b.fire.sort_by_key(|f| f.0);
                }
                let (cell, generation) = b.fire[b.fired];
                b.fired += 1;
                self.in_wheel -= 1;
                return Some(Event {
                    t,
                    phase: Phase::Fire,
                    cell,
                    payload: Payload::Fire { generation },
                });
            }
            // Tick exhausted: recycle its bucket and move on.
            self.recycle(t);
            self.now += 1;
            while let Some(&Reverse(t)) = self.overflow.peek() {
                if t >= self.now + WHEEL as u64 {
                    break;
                }
                self.overflow.pop();
                self.push_arrive(t);
            }
        }
    }

    fn recycle(&mut self, t: u64) {
        let b = &mut self.buckets[t as usize % WHEEL];
        debug_assert!(!b.remaining(), "recycling a tick with pending events");
        b.deliver.clear();
        b.order.clear();
        b.fire.clear();
        b.arena.clear();
        b.delivered = 0;
        b.fired = 0;
    }

    pub fn peek_t(&self) -> Option<u64> {
        if self.in_wheel > 0 {
            (self.now..self.now + WHEEL as u64)
                .find(|&t| self.buckets[t as usize % WHEEL].remaining())
        } else {
            self.overflow.peek().map(|r| r.0)
        }
    }

    pub fn pending(&self) -> usize {
        self.in_wheel + self.overflow.len()
    }

    /// Pending events in processing order (for hashing / inspection, not the hot path).
    pub fn sorted(&self) -> Vec<Event> {
        let mut out = Vec::with_capacity(self.pending());
        for t in self.now..self.now + WHEEL as u64 {
            let b = &self.buckets[t as usize % WHEEL];
            for _ in 0..b.sense {
                out.push(Event {
                    t,
                    phase: Phase::Sense,
                    cell: 0,
                    payload: Payload::Arrive,
                });
            }
            let deliver: Vec<Msg> = if b.delivered == 0 {
                let mut d = b.deliver.clone();
                d.sort_by_key(|m| m.dst);
                d
            } else {
                b.order[b.delivered..]
                    .iter()
                    .map(|&k| b.deliver[k as u32 as usize])
                    .collect()
            };
            for msg in &deliver {
                let payload = Payload::Deliver {
                    edge: msg.edge,
                    m: b.message(msg),
                    wave: msg.wave,
                };
                out.push(Event {
                    t,
                    phase: Phase::Deliver,
                    cell: msg.dst,
                    payload,
                });
            }
            let mut fire = b.fire[b.fired..].to_vec();
            if b.fired == 0 {
                fire.sort_by_key(|f| f.0);
            }
            for &(cell, generation) in &fire {
                out.push(Event {
                    t,
                    phase: Phase::Fire,
                    cell,
                    payload: Payload::Fire { generation },
                });
            }
        }
        let mut far: Vec<u64> = self.overflow.iter().map(|r| r.0).collect();
        far.sort_unstable();
        out.extend(far.into_iter().map(|t| Event {
            t,
            phase: Phase::Sense,
            cell: 0,
            payload: Payload::Arrive,
        }));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::ZERO;
    use crate::store::Id;

    fn edge(i: u32) -> EdgeId {
        Id {
            idx: i,
            generation: 0,
        }
    }

    #[test]
    fn ordering_time_phase_cell_insertion() {
        let mut s = Scheduler::default();
        let w = Wave::default();
        let h = s.push_state(5, &ZERO);
        s.push_fire(5, 1, 0);
        s.push_deliver(5, 9, edge(0), 1.0, h, w);
        s.push_deliver(5, 2, edge(1), 1.0, h, w);
        s.push_fire(3, 7, 0);
        s.push_arrive(5);
        s.push_deliver(5, 2, edge(2), 1.0, h, w); // same cell, later insertion
        s.push_arrive(500); // beyond the wheel
        assert_eq!(s.pending(), 7);
        assert_eq!(s.peek_t(), Some(3));
        let listed: Vec<_> = s.sorted().iter().map(|e| (e.t, e.phase, e.cell)).collect();
        let mut popped = Vec::new();
        let mut edges = Vec::new();
        while let Some(e) = s.pop() {
            if let Payload::Deliver { edge, .. } = e.payload {
                edges.push(edge.idx);
            }
            popped.push((e.t, e.phase, e.cell));
        }
        let want = vec![
            (3, Phase::Fire, 7),
            (5, Phase::Sense, 0),
            (5, Phase::Deliver, 2),
            (5, Phase::Deliver, 2),
            (5, Phase::Deliver, 9),
            (5, Phase::Fire, 1),
            (500, Phase::Sense, 0),
        ];
        assert_eq!(popped, want);
        assert_eq!(listed, want);
        assert_eq!(edges, vec![1, 2, 0], "per-cell insertion order kept");
        assert_eq!((s.pending(), s.peek_t()), (0, None));
    }

    #[test]
    fn listing_mid_tick_matches_remaining_order() {
        let mut s = Scheduler::default();
        let h = s.push_state(1, &ZERO);
        for (dst, e) in [(9, 0), (2, 1), (5, 2), (2, 3)] {
            s.push_deliver(1, dst, edge(e), 1.0, h, Wave::default());
        }
        s.pop(); // first delivery (cell 2, edge 1) consumed
        let listed: Vec<_> = s.sorted().iter().map(|e| e.cell).collect();
        assert_eq!(listed, vec![2, 5, 9]);
        let rest: Vec<_> = std::iter::from_fn(|| s.pop()).map(|e| e.cell).collect();
        assert_eq!(rest, listed);
    }

    #[test]
    fn message_is_state_times_send_gain() {
        let mut s = Scheduler::default();
        let mut h = ZERO;
        h[0] = 0.5;
        h[3] = -2.0;
        let slot = s.push_state(1, &h);
        s.push_deliver(1, 4, edge(0), 0.25, slot, Wave::default());
        let Some(Event {
            payload: Payload::Deliver { m, .. },
            ..
        }) = s.pop()
        else {
            panic!()
        };
        assert_eq!((m[0], m[3], m[1]), (0.125, -0.5, 0.0));
    }
}
