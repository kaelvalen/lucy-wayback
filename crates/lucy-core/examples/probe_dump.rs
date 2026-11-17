//! Offline probe data: for every byte arrival in a segment, the set of hidden cells that fired in
//! the K ticks before it (from the ring buffer) and the true byte. Read-only: never feeds back.
//!
//! usage: cargo run --release --example probe_dump -- CORPUS SEED START_BYTE N_BYTES OUT_PREFIX
//! writes OUT_PREFIX.{indptr (u32 LE), indices (u16 LE cell idx), labels (u8), meta.json}
//!
//! Uses only engine API present since `phase1-baseline`, so the same file can be dropped into a
//! worktree of any commit to probe it.

use std::fs;

use lucy_core::rules::Params;
use lucy_core::seed::{self, SeedConfig};
use lucy_core::store::Kind;

fn le<T: Copy, const N: usize>(xs: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    xs.iter().flat_map(|&x| f(x)).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(
        a.len() == 6,
        "usage: probe_dump CORPUS SEED START_BYTE N_BYTES OUT_PREFIX"
    );
    let corpus = fs::read(&a[1]).expect("read corpus");
    let seed: u64 = a[2].parse().expect("seed");
    let start: u64 = a[3].parse().expect("start byte");
    let end = start + a[4].parse::<u64>().expect("n bytes");
    let out = &a[5];
    assert!(end as usize <= corpus.len(), "segment past end of corpus");

    let cfg = SeedConfig {
        seed,
        ..SeedConfig::default()
    };
    let k = cfg.ticks_per_byte;
    let mut eng = seed::build(&cfg, Params::default());
    let mut src = corpus.iter().copied();
    let (mut indptr, mut indices, mut labels) = (vec![0u32], Vec::<u16>::new(), Vec::<u8>::new());
    let mut next = start.max(1);
    let mut cells: Vec<u16> = Vec::new();

    while let Some(t) = eng.next_time() {
        // Byte i arrives at tick i*K; every event processed so far happened before it.
        while next < end && next * k <= t {
            let from = next * k - k;
            cells.clear();
            let mut seq = eng.ring.total();
            while seq > 0 {
                seq -= 1;
                let Some(ev) = eng.ring.get(seq) else { break };
                if ev.t < from {
                    break;
                }
                if eng.g.cells.kind[ev.cell.idx as usize] == Kind::Hidden {
                    cells.push(ev.cell.idx as u16);
                }
            }
            cells.sort_unstable();
            cells.dedup();
            indices.extend_from_slice(&cells);
            indptr.push(indices.len() as u32);
            labels.push(corpus[next as usize]);
            next += 1;
        }
        if next >= end {
            break;
        }
        eng.step(&mut src);
    }

    fs::write(format!("{out}.indptr"), le(&indptr, u32::to_le_bytes)).expect("write indptr");
    fs::write(format!("{out}.indices"), le(&indices, u16::to_le_bytes)).expect("write indices");
    fs::write(format!("{out}.labels"), &labels).expect("write labels");
    let meta = format!(
        "{{\"corpus\": \"{}\", \"seed\": {seed}, \"start_byte\": {start}, \"n\": {}, \"ticks_per_byte\": {k}, \"cells\": {}, \"first_hidden\": 256}}\n",
        a[1],
        labels.len(),
        eng.g.cells.slots.capacity()
    );
    fs::write(format!("{out}.meta.json"), meta).expect("write meta");
    eprintln!(
        "{} samples, {:.1} active hidden cells/sample",
        labels.len(),
        indices.len() as f64 / labels.len() as f64
    );
}
