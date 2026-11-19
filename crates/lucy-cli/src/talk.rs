//! `lucy talk`: a REPL on a saved context tissue. A typed line arrives through the normal byte
//! path (Lucy learns it), then Lucy continues the text by sampling her blended prediction. Her
//! own bytes are context only, never evidence. Idle time and `/sleep` run sleep passes; the
//! tissue is saved on exit.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use lucy_core::kernel::Engine;
use lucy_core::observe::Observatory;
use lucy_core::rules::Tissue;
use lucy_core::save;

/// RNG stream id for sampling (any fixed id distinct from the dynamics streams).
const STREAM_TALK: u64 = 0x7a1c;

pub struct Talk {
    pub resume: PathBuf,
    pub save: Option<PathBuf>,
    pub temperature: f64,
    pub max_bytes: usize,
    pub entropy: bool,
    pub idle_secs: u64,
    pub seed: u64,
}

/// Default save path: never the womb itself; `x.bin` -> `x-talk.bin` (a talk save is reused).
fn talk_path(resume: &Path) -> PathBuf {
    let stem = resume.file_stem().unwrap_or_default().to_string_lossy();
    match stem.ends_with("-talk") {
        true => resume.to_path_buf(),
        false => resume.with_file_name(format!("{stem}-talk.bin")),
    }
}

/// One character per byte: entropy in whole bits, `+` for 10 or more.
fn entropy_marks(hs: &[f64]) -> String {
    hs.iter()
        .map(|&h| match h.round() as u32 {
            n @ 0..=9 => char::from_digit(n, 10).unwrap(),
            _ => '+',
        })
        .collect()
}

impl Talk {
    pub fn run(&self) -> Result<()> {
        let (mut eng, obs) = save::load(&self.resume)
            .with_context(|| format!("loading {}", self.resume.display()))?;
        ensure!(
            eng.p.tissue == Tissue::Context,
            "talk needs a save of a context-tissue run"
        );
        eng.wake();
        ensure!(eng.pending_events() == 1, "talk: unexpected pending events");
        ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "--temperature must be >= 0"
        );
        let out_path = self.save.clone().unwrap_or_else(|| talk_path(&self.resume));
        let mut rng = lucy_core::seed::stream(self.seed, STREAM_TALK);
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for line in io::stdin().lock().split(b'\n') {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        println!(
            "Lucy: {} context cells, {} bytes lived. Type a line; /sleep, /save, /quit. Saves to {}",
            eng.ctx.cells(),
            eng.stats.bytes_total,
            out_path.display()
        );
        let mut stdout = io::stdout();
        loop {
            print!("> ");
            stdout.flush()?;
            let msg = match self.idle_secs {
                0 => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
                s => rx.recv_timeout(Duration::from_secs(s)),
            };
            let mut line = match msg {
                Err(RecvTimeoutError::Timeout) => {
                    println!("\n(Lucy sleeps: {} cells pruned)", sleep(&mut eng));
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
                Ok(line) => line.context("reading stdin")?,
            };
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            match line.as_slice() {
                b"/quit" => break,
                b"/sleep" => {
                    println!("(Lucy sleeps: {} cells pruned)", sleep(&mut eng));
                    continue;
                }
                b"/save" => {
                    save_to(&out_path, &eng, &obs)?;
                    continue;
                }
                _ => {}
            }
            // Learn the line through the same arrival path as the corpus (one Arrive per byte).
            for &b in &line {
                eng.step(&mut std::iter::once(b));
            }
            let p = eng.p.clone();
            let mut hs = Vec::new();
            let mut ended = false;
            // If her first byte ends the line, the text goes on: she writes the next line.
            while hs.len() < self.max_bytes && (hs.is_empty() || (ended && hs.len() == 1)) {
                eng.ctx.generate(
                    &p,
                    self.temperature,
                    self.max_bytes - hs.len(),
                    b'\n',
                    &mut rng,
                    |x, h| {
                        hs.push(h);
                        ended = x == b'\n';
                        let _ = stdout.write_all(&[x]);
                        let _ = stdout.flush();
                    },
                );
            }
            if !ended {
                // Close Lucy's turn as context, so the next typed line starts a new line.
                let mut probs = vec![0.0; 256];
                eng.ctx.predict(&p, &mut probs);
                eng.ctx.pass(b'\n', &p, &probs);
                println!();
            }
            if self.entropy {
                println!("{}", entropy_marks(&hs));
            }
        }
        println!();
        save_to(&out_path, &eng, &obs)
    }
}

/// A full sleep: enough sweep passes to visit every slot once.
fn sleep(eng: &mut Engine) -> u64 {
    let p = eng.p.clone();
    eng.ctx.sleep_all(&p)
}

fn save_to(path: &Path, eng: &Engine, obs: &Observatory) -> Result<()> {
    save::save(path, eng, obs).with_context(|| format!("saving {}", path.display()))?;
    println!("(saved {})", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_overwrites_the_womb_by_default() {
        assert_eq!(
            talk_path(Path::new("runs/womb.bin")),
            Path::new("runs/womb-talk.bin")
        );
        assert_eq!(
            talk_path(Path::new("runs/womb-talk.bin")),
            Path::new("runs/womb-talk.bin")
        );
        assert_eq!(entropy_marks(&[0.2, 3.6, 12.0]), "04+");
    }
}
