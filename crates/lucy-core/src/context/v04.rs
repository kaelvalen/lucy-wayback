//! `Context` as saved in format 4 (before learned mixing). Never edit: it describes bytes on disk.

use serde::Deserialize;

use super::{Cell, Context, classes};

#[derive(Deserialize)]
pub struct ContextV04 {
    cells: Vec<Cell>,
    free: Vec<u32>,
    live: usize,
    hist: Vec<u8>,
    chunk: [u32; 2],
    now: u64,
    cursor: u32,
    grown: u64,
    pruned: u64,
    folded: u64,
    starved: u64,
    classes: classes::Classes,
}

impl From<ContextV04> for Context {
    /// Format 5 added only the mixer, which starts empty (filled from the fixed weighting if
    /// mixing is ever turned on).
    fn from(o: ContextV04) -> Context {
        Context {
            cells: o.cells,
            free: o.free,
            live: o.live,
            hist: o.hist,
            chunk: o.chunk,
            now: o.now,
            cursor: o.cursor,
            grown: o.grown,
            pruned: o.pruned,
            folded: o.folded,
            starved: o.starved,
            classes: o.classes,
            ..Context::default()
        }
    }
}
