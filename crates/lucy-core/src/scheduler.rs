//! Time-ordered event queue on a logical clock. Single-threaded and fully deterministic:
//! ties are broken by (phase, cell index, insertion sequence).

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::num::V;
use crate::store::{EdgeId, Wave};

/// Order of work inside one tick: senses, then every delivery, then fires. A cell therefore
/// integrates all messages of tick t before deciding to fire at t; its output arrives at t+1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    Sense,
    Deliver,
    Fire,
}

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
    pub seq: u64,
    pub payload: Payload,
}

impl Event {
    pub fn key(&self) -> (u64, Phase, u32, u64) {
        (self.t, self.phase, self.cell, self.seq)
    }
}

impl PartialEq for Event {
    fn eq(&self, o: &Self) -> bool {
        self.key() == o.key()
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Event {
    fn cmp(&self, o: &Self) -> Ordering {
        self.key().cmp(&o.key())
    }
}

#[derive(Default)]
pub struct Scheduler {
    heap: BinaryHeap<Reverse<Event>>,
    seq: u64,
}

impl Scheduler {
    pub fn push(&mut self, t: u64, phase: Phase, cell: u32, payload: Payload) {
        self.heap.push(Reverse(Event {
            t,
            phase,
            cell,
            seq: self.seq,
            payload,
        }));
        self.seq += 1;
    }

    pub fn pop(&mut self) -> Option<Event> {
        self.heap.pop().map(|Reverse(e)| e)
    }

    pub fn peek_t(&self) -> Option<u64> {
        self.heap.peek().map(|r| r.0.t)
    }

    pub fn pending(&self) -> usize {
        self.heap.len()
    }

    /// Pending events in processing order (for hashing / inspection, not the hot path).
    pub fn sorted(&self) -> Vec<&Event> {
        let mut v: Vec<&Event> = self.heap.iter().map(|r| &r.0).collect();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_time_phase_cell_seq() {
        let mut s = Scheduler::default();
        s.push(5, Phase::Fire, 1, Payload::Fire { generation: 0 });
        s.push(5, Phase::Deliver, 9, Payload::Fire { generation: 0 });
        s.push(5, Phase::Deliver, 2, Payload::Fire { generation: 0 }); // seq 2
        s.push(3, Phase::Fire, 7, Payload::Fire { generation: 0 });
        s.push(5, Phase::Sense, 0, Payload::Arrive);
        s.push(5, Phase::Deliver, 2, Payload::Fire { generation: 0 }); // seq 5, same cell
        let order: Vec<_> = std::iter::from_fn(|| s.pop())
            .map(|e| (e.t, e.phase, e.cell, e.seq))
            .collect();
        assert_eq!(
            order,
            vec![
                (3, Phase::Fire, 7, 3),
                (5, Phase::Sense, 0, 4),
                (5, Phase::Deliver, 2, 2),
                (5, Phase::Deliver, 2, 5),
                (5, Phase::Deliver, 9, 1),
                (5, Phase::Fire, 1, 0),
            ]
        );
    }
}
