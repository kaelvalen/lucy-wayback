//! Fire-event ring buffer: the recording that reverse replay (wayback) will read. Phase 1 only writes.

use crate::num::D;
use crate::store::CellId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FireEvent {
    pub cell: CellId,
    pub t: u64,
    /// State after the fire, quantized: `round(h * 127)` (h is a tanh output, so exact range).
    pub h: [i8; D],
    /// Ring sequence number of the same cell's previous fire (`NO_PREV` if none): its state is
    /// the `h` this fire started from (wayback replay reads it in O(1)).
    pub prev: u64,
}

pub const NO_PREV: u64 = u64::MAX;

/// Fixed-capacity, overwrite-oldest ring addressed by a global, monotonic sequence number.
#[derive(Clone)]
pub struct Ring {
    buf: Vec<FireEvent>,
    cap: usize,
    total: u64,
}

impl Ring {
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0, "ring capacity must be > 0");
        Self {
            buf: Vec::with_capacity(cap),
            cap,
            total: 0,
        }
    }

    pub fn push(&mut self, ev: FireEvent) {
        if self.buf.len() < self.cap {
            self.buf.push(ev);
        } else {
            self.buf[(self.total % self.cap as u64) as usize] = ev;
        }
        self.total += 1;
    }

    /// Events ever pushed (the next sequence number).
    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Event with global sequence number `seq`, if still retained.
    pub fn get(&self, seq: u64) -> Option<&FireEvent> {
        let oldest = self.total - self.buf.len() as u64;
        (oldest..self.total)
            .contains(&seq)
            .then(|| &self.buf[(seq % self.cap as u64) as usize])
    }

    /// Retained events, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &FireEvent> {
        let oldest = self.total - self.buf.len() as u64;
        (oldest..self.total).map(move |s| &self.buf[(s % self.cap as u64) as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Id;

    fn ev(t: u64) -> FireEvent {
        FireEvent {
            cell: Id {
                idx: t as u32,
                generation: 0,
            },
            t,
            h: [0; D],
            prev: NO_PREV,
        }
    }

    #[test]
    fn wraps_and_keeps_newest() {
        let mut r = Ring::new(3);
        for t in 0..5 {
            r.push(ev(t));
        }
        assert_eq!(r.total(), 5);
        assert_eq!(r.iter().map(|e| e.t).collect::<Vec<_>>(), vec![2, 3, 4]);
        assert!(r.get(1).is_none() && r.get(5).is_none());
        assert_eq!(r.get(3).map(|e| e.t), Some(3));
    }
}
