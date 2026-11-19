//! Word-level offline bound (step 8, `lucy word-bound`; docs/DESIGN.md). A measurement, not part of
//! the tissue: does a word level (previous words -> next word, with Brown-style word classes) add
//! enough at word-initial bytes to be worth building online?
//!
//! - Pass 1 runs the context tissue alone and records each byte's chunk boundary: Lucy's own
//!   surprise (`−log2 p(x) > bits`: x starts a new chunk) or the space rule (x is alphanumeric
//!   after a non-alphanumeric byte). Chunks are words; the lexicon and the word bigrams are taken
//!   from the whole pass (hindsight), and the most frequent words are clustered by their next word.
//! - Pass 2 reruns the tissue (one thread per variant) with word-level voters mixed into the byte
//!   vote (`Context::extra`, learned mixing). Word tables count online and prequentially; only the
//!   lexicon and the class map are hindsight. At byte i the word state uses boundaries up to i−1,
//!   so a boundary is never known before its byte arrives:
//!   - continuation voters (kinds 0–3): words that extend the current partial chunk, in the
//!     context of the previous 1–2 words / their classes;
//!   - next-word voters (kinds 4–7): the partial taken as a complete word w0, giving the first
//!     bytes of the next word after (w0) / (w−1, w0) / their classes.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::context::{Context, Extra};
use crate::observe::Fnv;
use crate::rules::Params;
use crate::senses::N_BYTES;

#[derive(Clone, Copy, Debug)]
pub enum Boundary {
    Surprise(f64),
    Space,
}

pub struct Config {
    pub boundary: Boundary,
    /// Class counts to try (each a pass-2 variant); words-only and no-words always run.
    pub classes: Vec<usize>,
    /// Words clustered (most frequent); the rest share one class.
    pub top_words: usize,
    /// Next-word features: the most frequent words, plus one "other".
    pub top_next: usize,
    pub sweeps: usize,
    /// Longest chunk prefix the tables index.
    pub prefix_cap: usize,
}

/// Prequential tallies; `word_rare`: word-initial bytes whose (w−1, w0) context was seen < 3 times.
#[derive(Clone, Copy, Debug, Default)]
pub struct Acc {
    pub n: u64,
    pub right: u64,
    pub bits: f64,
    pub wi: [u64; 2],
    pub rare: [u64; 2],
    pub word_rare: [u64; 2],
}

impl Acc {
    fn add(&mut self, o: &Acc) {
        self.n += o.n;
        self.right += o.right;
        self.bits += o.bits;
        for (a, b) in [
            (&mut self.wi, o.wi),
            (&mut self.rare, o.rare),
            (&mut self.word_rare, o.word_rare),
        ] {
            a[0] += b[0];
            a[1] += b[1];
        }
    }

    pub fn line(&self) -> String {
        let r = |x: [u64; 2]| 100.0 * x[0] as f64 / x[1].max(1) as f64;
        format!(
            "top1 {:6.2} %  bits {:.4}  word-initial {:6.2} %  rare {:6.2} %  word-rare {:6.2} % ({:.1} % of word starts)",
            100.0 * self.right as f64 / self.n.max(1) as f64,
            self.bits / self.n.max(1) as f64,
            r(self.wi),
            r(self.rare),
            r(self.word_rare),
            100.0 * self.word_rare[1] as f64 / self.wi[1].max(1) as f64,
        )
    }
}

pub struct Variant {
    pub name: String,
    pub whole: Acc,
    pub last: Acc,
}

fn word_initial(prev: Option<u8>, x: u8) -> bool {
    x.is_ascii_alphanumeric() && !prev.unwrap_or(b' ').is_ascii_alphanumeric()
}

/// Brown-style exchange: assign items (sparse feature counts) to k classes, maximizing
/// `Σ_class Σ_f N(class, f) ln(N(class, f) / N(class))`. Items come most frequent first; the
/// first k seed one class each, the rest join greedily, then `sweeps` exchange sweeps.
// Index loops: `i` addresses `items`, `tot` and `class` together.
#[allow(clippy::needless_range_loop)]
pub fn exchange(items: &[Vec<(u32, f64)>], features: usize, k: usize, sweeps: usize) -> Vec<usize> {
    let k = k.min(items.len()).max(1);
    let xlnx = |x: f64| if x > 0.0 { x * x.ln() } else { 0.0 };
    let tot: Vec<f64> = items.iter().map(|v| v.iter().map(|x| x.1).sum()).collect();
    let mut n = vec![0.0f64; k * features];
    let mut t = vec![0.0f64; k];
    let add = |n: &mut [f64], t: &mut [f64], j: usize, i: usize, s: f64| {
        for &(f, v) in &items[i] {
            n[j * features + f as usize] += s * v;
        }
        t[j] += s * tot[i];
    };
    let best = |n: &[f64], t: &[f64], i: usize| -> usize {
        // Pure per class with an index tie-break: the same on any thread count.
        (0..k)
            .into_par_iter()
            .map(|j| {
                let g: f64 = items[i]
                    .iter()
                    .map(|&(f, v)| {
                        let c = n[j * features + f as usize];
                        xlnx(c + v) - xlnx(c)
                    })
                    .sum::<f64>()
                    - (xlnx(t[j] + tot[i]) - xlnx(t[j]));
                (g, j)
            })
            .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)))
            .unwrap()
            .1
    };
    let mut class = vec![0usize; items.len()];
    for i in 0..items.len() {
        class[i] = if i < k { i } else { best(&n, &t, i) };
        add(&mut n, &mut t, class[i], i, 1.0);
    }
    for _ in 0..sweeps {
        let mut moved = 0;
        for i in 0..items.len() {
            add(&mut n, &mut t, class[i], i, -1.0);
            let j = best(&n, &t, i);
            moved += usize::from(j != class[i]);
            class[i] = j;
            add(&mut n, &mut t, j, i, 1.0);
        }
        if moved == 0 {
            break;
        }
    }
    class
}

/// Pass 1: the tissue alone; boundary bit per byte (`true`: the byte starts a new chunk).
fn boundaries(data: &[u8], p: &Params, rule: Boundary) -> Vec<bool> {
    let mut ctx = Context::default();
    let mut probs = vec![0.0; N_BYTES];
    let mut out = Vec::with_capacity(data.len());
    for (i, &x) in data.iter().enumerate() {
        if i > 0 && i % p.consolidate_every.max(1) as usize == 0 {
            ctx.sleep(p);
        }
        ctx.arrive(x, p, &mut probs);
        out.push(match rule {
            Boundary::Surprise(bits) => i == 0 || -f64::from(probs[x as usize]).log2() > bits,
            Boundary::Space => i == 0 || word_initial(Some(data[i - 1]), x),
        });
    }
    out
}

/// Next-byte count tables keyed by (namespace, context ids, prefix bytes).
#[derive(Default)]
struct Tables(HashMap<u64, Vec<(u8, f32)>>);

fn key(ns: u64, ids: &[u32], prefix: &[u8]) -> u64 {
    let mut h = Fnv::default();
    h.u64(ns);
    for &i in ids {
        h.u64(i as u64);
    }
    h.bytes(prefix);
    h.finish()
}

impl Tables {
    fn add(&mut self, k: u64, b: u8) {
        let v = self.0.entry(k).or_default();
        match v.binary_search_by_key(&b, |x| x.0) {
            Ok(i) => v[i].1 += 1.0,
            Err(i) => v.insert(i, (b, 1.0)),
        }
    }

    fn extra(&self, k: u64, kind: usize) -> Option<Extra> {
        let v = self.0.get(&k)?;
        let succ: Vec<(u8, f64)> = v.iter().map(|&(b, c)| (b, f64::from(c))).collect();
        let total = succ.iter().map(|x| x.1).sum();
        Some(Extra { kind, succ, total })
    }

    fn total(&self, k: u64) -> f32 {
        self.0.get(&k).map_or(0.0, |v| v.iter().map(|x| x.1).sum())
    }
}

const NS_W1: u64 = 1;
const NS_W2: u64 = 2;
const NS_C1: u64 = 3;
const NS_C2: u64 = 4;

/// Pass 2 for one variant (`classes`: None = no class voters; `words`: false = tissue alone).
fn pass2(
    data: &[u8],
    p: &Params,
    b: &[bool],
    lex: &HashMap<&[u8], u32>,
    class: Option<&[u32]>,
    words: bool,
    cap: usize,
) -> (Acc, Acc) {
    let window = 1 << 20;
    let mut ctx = Context::default();
    let mut probs = vec![0.0; N_BYTES];
    let mut tables = Tables::default();
    let (mut whole, mut last, mut cur) = (Acc::default(), Acc::default(), Acc::default());
    let (mut w1, mut w2): (Option<u32>, Option<u32>) = (None, None);
    let mut start = 0usize; // first byte of the current partial chunk
    let cls = |w: u32| class.map(|c| c[w as usize]);
    for (i, &x) in data.iter().enumerate() {
        if i > 0 && i % p.consolidate_every.max(1) as usize == 0 {
            ctx.sleep(p);
        }
        let partial = &data[start..i];
        let w0 = (!partial.is_empty())
            .then(|| lex.get(partial).copied())
            .flatten();
        ctx.extra.clear();
        let rare_key = match (w1, w0) {
            (Some(a), Some(z)) => Some(key(NS_W2, &[a, z], b"")),
            _ => None,
        };
        if words {
            let mut push = |k: Option<u64>, kind: usize| {
                if let Some(e) = k.and_then(|k| tables.extra(k, kind)) {
                    ctx.extra.push(e);
                }
            };
            if partial.len() <= cap {
                push(w1.map(|a| key(NS_W1, &[a], partial)), 0);
                push(w1.zip(w2).map(|(a, z)| key(NS_W2, &[z, a], partial)), 1);
                push(w1.and_then(cls).map(|c| key(NS_C1, &[c], partial)), 2);
                let c2 = w1.and_then(cls).zip(w2.and_then(cls));
                push(c2.map(|(c, d)| key(NS_C2, &[d, c], partial)), 3);
            }
            push(w0.map(|z| key(NS_W1, &[z], b"")), 4);
            push(rare_key, 5);
            push(w0.and_then(cls).map(|c| key(NS_C1, &[c], b"")), 6);
            let c2 = w0.and_then(cls).zip(w1.and_then(cls));
            push(c2.map(|(c, d)| key(NS_C2, &[d, c], b"")), 7);
        }
        let ok = ctx.arrive(x, p, &mut probs) == x;
        let o = u64::from(ok);
        cur.n += 1;
        cur.right += o;
        cur.bits -= f64::from(probs[x as usize]).log2();
        if ctx.deep_visits < 3 {
            cur.rare[0] += o;
            cur.rare[1] += 1;
        }
        if word_initial(i.checked_sub(1).map(|j| data[j]), x) {
            cur.wi[0] += o;
            cur.wi[1] += 1;
            if rare_key.is_none_or(|k| tables.total(k) < 3.0) {
                cur.word_rare[0] += o;
                cur.word_rare[1] += 1;
            }
        }
        // x starts a new chunk: the partial is a complete word; count it in its contexts.
        if b[i] && i > 0 {
            if let Some(&w) = lex.get(&data[start..i]) {
                let word = &data[start..i];
                for j in 0..word.len().min(cap + 1) {
                    let pre = &word[..j];
                    if let Some(a) = w1 {
                        tables.add(key(NS_W1, &[a], pre), word[j]);
                        if let Some(z) = w2 {
                            tables.add(key(NS_W2, &[z, a], pre), word[j]);
                        }
                    }
                    if let Some(c) = w1.and_then(cls) {
                        tables.add(key(NS_C1, &[c], pre), word[j]);
                        if let Some(d) = w2.and_then(cls) {
                            tables.add(key(NS_C2, &[d, c], pre), word[j]);
                        }
                    }
                }
                (w2, w1) = (w1, Some(w));
            }
            start = i;
        }
        if (i + 1) % window == 0 || i + 1 == data.len() {
            whole.add(&cur);
            if cur.n * 10 >= window as u64 * 9 {
                last = cur;
            }
            cur = Acc::default();
        }
    }
    (whole, last)
}

/// The whole measurement. Variants: tissue alone, words, words + each class count.
pub fn word_bound(data: &[u8], p: &Params, cfg: &Config) -> (usize, Vec<Variant>) {
    assert!(
        p.ctx_mix,
        "word-bound mixes word voters through learned mixing (ctx_mix)"
    );
    let b = boundaries(data, p, cfg.boundary);
    // Lexicon (ids by first occurrence) and word sequence.
    let mut lex: HashMap<&[u8], u32> = HashMap::new();
    let mut seq: Vec<u32> = Vec::new();
    let starts: Vec<usize> = (0..data.len()).filter(|&i| b[i]).collect();
    for (s, &a) in starts.iter().enumerate() {
        let e = starts.get(s + 1).copied().unwrap_or(data.len());
        let n = lex.len() as u32;
        seq.push(*lex.entry(&data[a..e]).or_insert(n));
    }
    let v = lex.len();
    let mut freq = vec![0u64; v];
    for &w in &seq {
        freq[w as usize] += 1;
    }
    let mut order: Vec<u32> = (0..v as u32).collect();
    order.sort_by_key(|&w| (std::cmp::Reverse(freq[w as usize]), w));
    let rank_of = |n: usize| {
        let mut r = vec![u32::MAX; v];
        for (i, &w) in order.iter().take(n).enumerate() {
            r[w as usize] = i as u32;
        }
        r
    };
    let (item_rank, next_rank) = (rank_of(cfg.top_words), rank_of(cfg.top_next));
    let mut items: Vec<HashMap<u32, f64>> = vec![HashMap::new(); cfg.top_words.min(v)];
    for w in seq.windows(2) {
        let r = item_rank[w[0] as usize];
        if r != u32::MAX {
            let f = next_rank[w[1] as usize].min(cfg.top_next as u32);
            *items[r as usize].entry(f).or_default() += 1.0;
        }
    }
    let items: Vec<Vec<(u32, f64)>> = items
        .into_iter()
        .map(|m| {
            let mut v: Vec<(u32, f64)> = m.into_iter().collect();
            v.sort_unstable_by_key(|x| x.0);
            v
        })
        .collect();
    let maps: Vec<(usize, Vec<u32>)> = cfg
        .classes
        .iter()
        .map(|&k| {
            let c = exchange(&items, cfg.top_next + 1, k, cfg.sweeps);
            let mut m = vec![k as u32; v]; // words outside the clustered set share class k
            for (r, &cl) in c.iter().enumerate() {
                m[order[r] as usize] = cl as u32;
            }
            (k, m)
        })
        .collect();
    let mut jobs: Vec<(String, Option<&[u32]>, bool)> = vec![
        ("tissue alone".into(), None, false),
        ("+ words".into(), None, true),
    ];
    for (k, m) in &maps {
        jobs.push((format!("+ words + {k} classes"), Some(m.as_slice()), true));
    }
    let out = jobs
        .par_iter()
        .map(|(name, cl, words)| {
            let (whole, last) = pass2(data, p, &b, &lex, *cl, *words, cfg.prefix_cap);
            Variant {
                name: name.clone(),
                whole,
                last,
            }
        })
        .collect();
    (v, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exchange_groups_by_next_word() {
        // Items 0 and 2 are followed by feature 0, items 1 and 3 by feature 1.
        let items = vec![
            vec![(0, 9.0), (1, 1.0)],
            vec![(1, 8.0)],
            vec![(0, 7.0)],
            vec![(1, 6.0), (0, 1.0)],
        ];
        let c = exchange(&items, 2, 2, 5);
        assert_eq!(c[0], c[2]);
        assert_eq!(c[1], c[3]);
        assert_ne!(c[0], c[1]);
    }

    /// With words that recur in fixed pairs, the word voters help at word starts.
    #[test]
    fn words_help_on_a_word_level_pattern() {
        let p = Params {
            ctx_depth: 4,
            ctx_mix: true,
            ..Params::default()
        };
        let mut text = Vec::new();
        let words = [&b"alpha beta "[..], b"gamma delta ", b"kappa sigma "];
        for i in 0..3000usize {
            text.extend_from_slice(words[(i * 7 + i / 3) % 3]);
        }
        let cfg = Config {
            boundary: Boundary::Space,
            classes: vec![2],
            top_words: 100,
            top_next: 100,
            sweeps: 3,
            prefix_cap: 24,
        };
        let (_, v) = word_bound(&text, &p, &cfg);
        let wi = |a: &Acc| a.wi[0] as f64 / a.wi[1] as f64;
        assert!(
            wi(&v[1].whole) > wi(&v[0].whole),
            "{} vs {}",
            v[1].whole.line(),
            v[0].whole.line()
        );
    }
}
