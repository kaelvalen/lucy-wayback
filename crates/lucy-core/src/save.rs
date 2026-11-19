//! Save/resume: the whole run state (engine, scheduler, ring, plasticity, observatory and probe
//! windows) in one bincode file. Loading and continuing is bit-identical to never stopping, on
//! the same binary and CPU (the same condition as `check-determinism`).
//!
//! **Tissue continuity.** Lucy must survive code updates. A save starts with `LUCYSV` and a
//! two-digit format version. Loading an older format runs the migration chain v(N) -> v(N+1)
//! -> ... -> current; new fields get values that reproduce the old behaviour. On every change of
//! a saved layout:
//! 1. freeze the changed structs as they are now in `save/vNN.rs` (or next to their private
//!    types, like `context/v04.rs`), with a `From` into the new types;
//! 2. bump `CURRENT` and add the step to `load_versioned`;
//! 3. add a fixture of the old format to `tests/fixtures` (see `tests/fixtures/README.md`).

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::kernel::Engine;
use crate::observe::Observatory;

mod v04;

/// Current format version. Format history: 4 = depth 64 / vote cap (857a46f), 5 = learned mixing.
pub const CURRENT: u32 = 5;
/// Oldest format that still migrates.
pub const OLDEST: u32 = 4;
const PREFIX: &[u8; 6] = b"LUCYSV";

fn magic(version: u32) -> [u8; 8] {
    let mut m = [0u8; 8];
    m[..6].copy_from_slice(PREFIX);
    m[6..].copy_from_slice(format!("{version:02}").as_bytes());
    m
}

/// Write atomically (temp file, then rename), so a crash mid-save keeps the previous save.
pub fn save(path: &Path, eng: &Engine, obs: &Observatory) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(&magic(CURRENT))?;
        bincode::serialize_into(&mut w, &(eng, obs)).map_err(io::Error::other)?;
        w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    }
    fs::rename(&tmp, path)
}

/// Load a save, migrating older formats (see `load_versioned`).
pub fn load(path: &Path) -> io::Result<(Engine, Observatory)> {
    load_versioned(path).map(|(e, o, _)| (e, o))
}

/// Load a save of any supported format, migrating it in memory to the current one; also returns
/// the format it was stored in. The file is never written. The observatory comes back without a
/// writer and without prober config/corpus, and the count baselines are empty: the caller
/// reattaches them (`Observatory::set_out`, `Prober::reattach`, `stats.baselines.rebuild`).
pub fn load_versioned(path: &Path) -> io::Result<(Engine, Observatory, u32)> {
    let bad = |msg: String| io::Error::new(io::ErrorKind::InvalidData, msg);
    let mut r = BufReader::new(File::open(path)?);
    let mut m = [0u8; 8];
    r.read_exact(&mut m)?;
    let version = std::str::from_utf8(&m[6..])
        .ok()
        .filter(|_| &m[..6] == PREFIX)
        .and_then(|v| v.parse::<u32>().ok())
        .ok_or_else(|| bad(format!("{}: not a Lucy save", path.display())))?;
    let de = |e: bincode::Error| bad(format!("{}: format {version}: {e}", path.display()));
    let (mut eng, obs) = match version {
        CURRENT => bincode::deserialize_from::<_, (Engine, Observatory)>(r).map_err(de)?,
        4 => {
            let (e, o) =
                bincode::deserialize_from::<_, (v04::EngineV04, Observatory)>(r).map_err(de)?;
            (e.into(), o)
        }
        v => {
            return Err(bad(format!(
                "{}: format {v} has no migration (supported: {OLDEST}..={CURRENT})",
                path.display()
            )));
        }
    };
    eng.rebuild_derived();
    Ok((eng, obs, version))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::state_hash;
    use crate::seed::{self, SeedConfig};

    /// Every historical format in `tests/fixtures` loads, migrates, and continues exactly as
    /// the binary of its own format did: same metrics lines (minus wall clock) to the end of
    /// the corpus. This is what "Lucy survives code updates" means.
    #[test]
    fn fixtures_migrate_and_continue_like_their_own_binary() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
        let corpus = fs::read(dir.join("corpus.txt")).unwrap();
        let strip = |text: &str| -> Vec<serde_json::Value> {
            text.lines()
                .map(|l| {
                    let mut v: serde_json::Value = serde_json::from_str(l).unwrap();
                    v.as_object_mut().unwrap().remove("wall");
                    v
                })
                .collect()
        };
        for version in OLDEST..=CURRENT {
            let (mut eng, mut obs, v) =
                load_versioned(&dir.join(format!("v{version:02}.bin"))).unwrap();
            assert_eq!(v, version);
            let out = std::env::temp_dir().join(format!(
                "lucy-fixture-{version}-{}.jsonl",
                std::process::id()
            ));
            obs.set_out(Some(Box::new(File::create(&out).unwrap())));
            let seen = eng.stats.bytes_total as usize;
            eng.stats.baselines.rebuild(&corpus[..seen]);
            let mut src = corpus.iter().copied().skip(seen);
            eng.run_until(&mut src, 320_000, &mut obs).unwrap();
            obs.finish(&mut eng).unwrap();
            drop(obs);
            let got = strip(&fs::read_to_string(&out).unwrap());
            fs::remove_file(&out).unwrap();
            let want = strip(
                &fs::read_to_string(dir.join(format!("v{version:02}-continue.jsonl"))).unwrap(),
            );
            assert!(!want.is_empty());
            assert_eq!(
                got, want,
                "format {version} does not continue like its own binary"
            );
        }
    }

    #[test]
    fn rejects_unknown_formats() {
        let p = std::env::temp_dir().join(format!("lucy-bad-{}.bin", std::process::id()));
        fs::write(&p, b"LUCYSV03rest").unwrap();
        let err = |p: &Path| load(p).err().expect("must not load").to_string();
        let e = err(&p);
        assert!(e.contains("no migration"), "{e}");
        fs::write(&p, b"NOTLUCY!").unwrap();
        assert!(err(&p).contains("not a Lucy save"));
        fs::remove_file(&p).unwrap();
    }

    #[test]
    fn save_resume_is_bit_identical() {
        for tissue in [
            crate::rules::Tissue::Context,
            crate::rules::Tissue::Reservoir,
        ] {
            resume_matches(tissue);
        }
    }

    fn resume_matches(tissue: crate::rules::Tissue) {
        let cfg = SeedConfig {
            hidden: 256,
            ..SeedConfig::default()
        };
        let p = crate::rules::Params {
            consolidate_every: 300, // exercise plasticity and wayback state across the cut
            tissue,
            ctx_t_v: 300.0, // prune across the cut too
            cls_mode: 1,    // class cells across the cut too
            cls_min_visits: 4,
            ..Default::default()
        };
        // Two phrases in turn: the first one's contexts fade and are pruned during the second.
        let phrase = |s: &[u8]| s.iter().copied().cycle().take(2000).collect::<Vec<u8>>();
        let mut corpus = phrase(b"the cat sat on the mat. the dog sat on the log. ");
        corpus.extend(phrase(b"a bird flew over my house; ducks swim. "));
        let k = cfg.ticks_per_byte;
        let (n, total) = (1500 * k + 3, 3000 * k);
        let fresh = || {
            let eng = seed::build(&cfg, p.clone());
            let obs = Observatory::new(&eng, 997, cfg.seed, None);
            (eng, obs)
        };

        let (mut a, mut oa) = fresh();
        let mut src = corpus.iter().copied();
        a.run_until(&mut src, total, &mut oa).unwrap();
        oa.finish(&mut a).unwrap();

        let (mut b, mut ob) = fresh();
        let mut src = corpus.iter().copied();
        b.run_until(&mut src, n, &mut ob).unwrap();
        let path = std::env::temp_dir().join(format!("lucy-save-test-{}.bin", std::process::id()));
        save(&path, &b, &ob).unwrap();
        drop((b, ob));
        let (mut b, mut ob) = load(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let seen = b.stats.bytes_total as usize;
        b.stats.baselines.rebuild(&corpus[..seen]);
        let mut src = corpus.iter().copied().skip(b.stats.bytes_total as usize);
        b.run_until(&mut src, total, &mut ob).unwrap();
        ob.finish(&mut b).unwrap();

        match tissue {
            crate::rules::Tissue::Context => {
                assert!(a.ctx.pruned > 0 && a.ctx.cells() > 1 && a.ctx.classes() > 0)
            }
            crate::rules::Tissue::Reservoir => assert!(a.plasticity.passes > 0),
        }
        assert_eq!(state_hash(&a), state_hash(&b));
        assert_eq!(oa.metrics_hash(), ob.metrics_hash());
        assert_eq!(oa.reports, ob.reports);
    }
}
