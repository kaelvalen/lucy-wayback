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

/// One mark per printed character (as a UTF-8 terminal shows `text`): the highest entropy of
/// its bytes in whole bits, `+` for 10 or more. An invalid sequence prints as one U+FFFD.
// shortcut: assumes one column per character; wide (CJK, emoji) characters need unicode-width.
fn entropy_marks(text: &[u8], hs: &[f64]) -> String {
    let mark = |h: f64| match h.round() as u32 {
        n @ 0..=9 => char::from_digit(n, 10).unwrap(),
        _ => '+',
    };
    // (bytes, shown) per character; a newline shows no mark.
    let mut units = Vec::new();
    for chunk in text.utf8_chunks() {
        units.extend(chunk.valid().chars().map(|c| (c.len_utf8(), c != '\n')));
        if !chunk.invalid().is_empty() {
            units.push((chunk.invalid().len(), true));
        }
    }
    let mut i = 0;
    units
        .into_iter()
        .filter_map(|(n, shown)| {
            let h = hs[i..i + n].iter().copied().fold(0.0, f64::max);
            i += n;
            shown.then(|| mark(h))
        })
        .collect()
}

/// What to do with a sampled byte while watching for another speaker's tag.
#[derive(Debug, PartialEq)]
enum Step {
    /// Print these bytes now (the byte itself, or a held run that turned out not to be a tag).
    Emit(Vec<u8>),
    /// Maybe the start of a tag: hold it back.
    Hold,
    /// A second speaker tag (`*` + uppercase + `:` at a turn start): Lucy's turn is over.
    Stop,
}

/// Lucy's first speaker tag is her own turn; the next one is someone else's.
struct Speaker {
    tags: u32,
    held: Vec<u8>,
    prev: Option<u8>,
}

impl Speaker {
    fn new(prev: Option<u8>) -> Self {
        Self {
            tags: 0,
            held: Vec::new(),
            prev,
        }
    }

    fn push(&mut self, x: u8) -> Step {
        let at_turn_start = self.prev.is_none_or(|b| b == b' ' || b == b'\n');
        self.prev = Some(x);
        if self.held.is_empty() {
            return match x == b'*' && at_turn_start {
                true => {
                    self.held.push(x);
                    Step::Hold
                }
                false => Step::Emit(vec![x]),
            };
        }
        self.held.push(x);
        let name = &self.held[1..];
        match x {
            b':' if name.len() > 1 => {
                self.tags += 1;
                match self.tags {
                    1 => Step::Emit(std::mem::take(&mut self.held)),
                    _ => Step::Stop,
                }
            }
            b'A'..=b'Z' => Step::Hold,
            _ => Step::Emit(std::mem::take(&mut self.held)),
        }
    }
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
            let (text, hs) = self.reply(&mut eng, &p, &mut rng, &mut stdout)?;
            if text.last() != Some(&b'\n') {
                // Close Lucy's turn as context, so the next typed line starts a new line.
                let mut probs = vec![0.0; 256];
                eng.ctx.predict(&p, &mut probs);
                eng.ctx.pass(b'\n', &p, &probs);
                println!();
            }
            if self.entropy {
                println!("{}", entropy_marks(&text, &hs));
            }
        }
        println!();
        save_to(&out_path, &eng, &obs)
    }
}

impl Talk {
    /// Lucy's turn: sample bytes (context only) until a newline, `max_bytes`, or another
    /// speaker's tag, which is rewound so it is neither printed nor kept as context. If her
    /// first byte ends the line, the text goes on and she writes the next line. Returns the
    /// printed bytes and their entropies.
    fn reply(
        &self,
        eng: &mut Engine,
        p: &lucy_core::rules::Params,
        rng: &mut impl rand::Rng,
        out: &mut impl Write,
    ) -> Result<(Vec<u8>, Vec<f64>)> {
        let ctx = &mut eng.ctx;
        let mut probs = vec![0.0; 256];
        let mut speaker = Speaker::new(ctx.last());
        let (mut text, mut hs, mut held_h) = (Vec::new(), Vec::new(), Vec::new());
        let mut mark = None;
        for _ in 0..self.max_bytes {
            let before = speaker.held.is_empty().then(|| ctx.mark());
            let x = ctx.sample(p, self.temperature, rng, &mut probs);
            held_h.push(ctx.entropy);
            match speaker.push(x) {
                Step::Stop => {
                    ctx.rewind(mark.take().expect("a held tag has a mark"));
                    return Ok((text, hs));
                }
                Step::Hold => {
                    mark = mark.or(before);
                    ctx.pass(x, p, &probs);
                }
                Step::Emit(bytes) => {
                    ctx.pass(x, p, &probs);
                    mark = None;
                    out.write_all(&bytes)?;
                    out.flush()?;
                    text.extend(bytes);
                    hs.append(&mut held_h);
                    if x == b'\n' && text.len() > 1 {
                        break;
                    }
                }
            }
        }
        // Out of bytes in the middle of a held run: it is context already, so show it.
        out.write_all(&speaker.held)?;
        text.append(&mut speaker.held);
        hs.append(&mut held_h);
        Ok((text, hs))
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
        assert_eq!(entropy_marks(b"ab ", &[0.2, 3.6, 12.0]), "04+");
        // "é" is two bytes but one column; an invalid byte prints as one U+FFFD; '\n' none.
        assert_eq!(entropy_marks("é!".as_bytes(), &[1.0, 5.0, 2.0]), "52");
        assert_eq!(entropy_marks(b"\xffa\n", &[3.0, 1.0, 9.0]), "31");
    }

    fn run(speaker: &mut Speaker, s: &[u8]) -> (Vec<u8>, bool) {
        let mut out = Vec::new();
        for &x in s {
            match speaker.push(x) {
                Step::Emit(b) => out.extend(b),
                Step::Hold => {}
                Step::Stop => return (out, true),
            }
        }
        (out, false)
    }

    #[test]
    fn stops_at_the_second_speaker_tag() {
        let mut sp = Speaker::new(Some(b'?'));
        let (out, stop) = run(&mut sp, b" *CHI: yeah . *EXP: now I have");
        assert!(stop);
        assert_eq!(out, b" *CHI: yeah . ");
        // Not tags: '*' inside a word, lowercase after '*', a lone "*:".
        let mut sp = Speaker::new(Some(b' '));
        let s = b"a*b *Mom *: ok";
        assert_eq!(run(&mut sp, s), (s.to_vec(), false));
        // A line start counts as a turn start too.
        let mut sp = Speaker::new(None);
        let (out, stop) = run(&mut sp, b"*MOT: hi .\n*CHI: no");
        assert_eq!((out.as_slice(), stop), (&b"*MOT: hi .\n"[..], true));
    }
}
