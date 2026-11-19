//! Save/resume: the whole run state (engine, scheduler, ring, plasticity, observatory and probe
//! windows) in one bincode file. Loading and continuing is bit-identical to never stopping, on
//! the same binary and CPU (the same condition as `check-determinism`).

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::kernel::Engine;
use crate::observe::Observatory;

/// File magic + format version: bump the version whenever a saved struct changes.
const MAGIC: &[u8; 8] = b"LUCYSV01";

/// Write atomically (temp file, then rename), so a crash mid-save keeps the previous save.
pub fn save(path: &Path, eng: &Engine, obs: &Observatory) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(MAGIC)?;
        bincode::serialize_into(&mut w, &(eng, obs)).map_err(io::Error::other)?;
        w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    }
    fs::rename(&tmp, path)
}

/// Load a save. The observatory comes back without a writer and without prober config/corpus:
/// the caller reattaches them (`Observatory::set_out`, `Prober::reattach`).
pub fn load(path: &Path) -> io::Result<(Engine, Observatory)> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: not a Lucy save of this format", path.display()),
        ));
    }
    let (mut eng, obs): (Engine, Observatory) =
        bincode::deserialize_from(r).map_err(io::Error::other)?;
    eng.rebuild_derived();
    Ok((eng, obs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::state_hash;
    use crate::seed::{self, SeedConfig};

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
        let mut src = corpus.iter().copied().skip(b.stats.bytes_total as usize);
        b.run_until(&mut src, total, &mut ob).unwrap();
        ob.finish(&mut b).unwrap();

        match tissue {
            crate::rules::Tissue::Context => assert!(a.ctx.pruned > 0 && a.ctx.cells() > 1),
            crate::rules::Tissue::Reservoir => assert!(a.plasticity.passes > 0),
        }
        assert_eq!(state_hash(&a), state_hash(&b));
        assert_eq!(oa.metrics_hash(), ob.metrics_hash());
        assert_eq!(oa.reports, ob.reports);
    }
}
