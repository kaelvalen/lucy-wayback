# Lucy v2: the cell is a context

## Why the current tissue stalls

Every experiment so far points to one structural flaw, not a missing knob:

- Hidden cells are **anonymous**. A random reservoir cell fires at the homeostatic rate whatever
  the input, so it says almost nothing about the next byte. The probe needs thousands of them to
  pull out 44.5 %.
- Growth then picks **frequent cells over selective cells**. Only 3 % of the true byte's in-edges
  come from the bigram path.
- W learning (C) moves the representation under the readout (moving target), and the gains are
  ≤ 1.5 points.

The literature on sparse, local, online sequence learning converges on the opposite design: **a
cell's identity is its context.**
- In HTM sequence memory, cells in a minicolumn share the same input but fire in different
  contexts. Unpredicted input makes the whole column burst, and learning grows synapses to the
  cells that were just active (Hawkins & Ahmad 2016; Cui et al. 2016).
- Cloned HMMs (Dedieu, George et al. 2019) map many hidden "clones" to one observed symbol. They
  report beating n-grams, sequence memoizers and LSTMs on character-level language modeling.

## The validated core (prototype, CHILDES 1 MB, prequential)

`context_cells_reference.py` uses the laws as they stand, except that a cell is a context:

| | top-1 | bits/byte |
|---|---|---|
| unigram / bigram / trigram (online counts) | 21.0 / 41.3 / 57.3 % | 4.68 / 2.80 / 2.31 (probe split) |
| Lucy today (best: cap 128, no C) | 38.5 % | 3.42 |
| HTM-style clone columns (my first try, single winner) | 43–45 % | — |
| **context cells, K = 8** | **70.5 %** (64 % at 100 k, still rising at 1 MB) | **1.52** |

- Robust: K = 4–16 gives 67–69 % at 300 k bytes, and `grow_min` 2 → 8 halves the cell count for −0.8 points.
- Speed: 1 MB in 36 s in **pure Python** (≈ 28 k bytes/s). Lucy now runs ≈ 1–1.5 k bytes/s.
- It keeps improving with data. That is continual learning, where the current D run was flat.

Mapping to the laws (nothing new is added):

| Law | Context-cell form |
|---|---|
| SPREAD | Byte x arrives, and each active cell passes activation to its child for x. New chain = {root} ∪ {child(c, x)}. That is ≤ K+1 cells, neighbour to neighbour, with no hashing of history. |
| TRACE | The prediction is a confidence-weighted vote of the active chain: w = T/(T+2) · 2^depth / T. A deep cell whose evidence is weak means "I know that I know, but not what". |
| TENSION | When the deepest active cell mispredicts and has ≥ `grow_min` evidence, it grows one child one byte deeper. That child is the clone that disambiguates its context. Growth happens only where Lucy is wrong. |
| DECAY | Successor counts are earned vitality and decay lazily. Cells that stop being visited fade and die in sleep. |

## Two negative results that constrain the design

1. **Sleep must reorganize, not re-count.** Replaying the last 200 misses per 1 000 bytes as extra
   counts made the prototype *worse*: −0.2 to −0.7 points at every checkpoint to 60 k. It is the same
   failure as Lucy's surprise-ranked readout consolidation: replay skews the statistics. So
   wayback's job is structural (prune, merge, chunk, classes), never adding evidence.
2. **Tension marks word boundaries for free.** Accuracy is 35 % at word-initial bytes, 69 % inside
   words and 92 % at spaces. Infants segment speech at transitional-probability dips in the same
   way (Saffran, Aslin & Newport 1996). Chunk cells keyed (previous chunk, current partial chunk)
   gave +0.7 points. That is the seed of the hierarchy. The 35 % at word starts is the real open
   problem: which word comes next needs generalization, not more memory.

## Pinpoint changes (in order)

0. **Save/resume** (unchanged; still first).
1. **`context.rs`: context cells as the default tissue for the byte stream.**
   - Cell: `parent, depth, children: small sorted (byte → cell), succ: small sorted (byte, count),
     total, last_t`.
   - Arrival path: vote → competitive softmax over the 256-byte sheet (keep the existing readout
     sheet) → learn on the chain → grow at the deepest miss.
   - Lazy decay with `t_v`. Prune in consolidation below a vitality floor, with a cell budget as
     metabolism.
   - Deterministic, O(K) per byte.
   - Acceptance: ≥ 65 % top-1 and < 2.0 bits on CHILDES 1 MB, cost per byte independent of cell
     count. Port `context_cells_reference.py` exactly first (same numbers ± tie-break), then go native.
2. **Retire the reservoir from the byte path.** Keep the W/Q hidden cells behind a flag. They
   are the dense part (8 KB per cell, memory-bound, 65 °C at full clock) and do not carry the
   signal. They come back later as the encoder for continuous senses (audio, image), where
   similarity matters.
3. **Wayback becomes structural only.** Its jobs:
   - prune dead cells;
   - fold a child into its parent when its successor distribution adds nothing over the parent
     (low KL, low count). This is MDL-style compression and the "death" half of the split law;
   - build chunk cells at tension peaks.
   - Test: same or better accuracy with fewer cells.
4. **Chunk cells (hierarchy).** At tension-peak boundaries, a chunk cell for the completed chunk
   joins the chain as context for the next chunk. Measured +0.7 points with spaces as boundaries.
   Next: derive boundaries from Lucy's own tension instead of spaces.
5. **Experimental, unvalidated: generalization through successor-similarity classes.**
   - In sleep, sketch each cell's successor distribution (MinHash/LSH over its top successors,
     CPU-cheap), cluster similar contexts, and create *class cells* that pool evidence.
   - A rare or new deep context then backs off to its class before backing off to a shorter
     context. "the dog" and "the cat" share a class, so what follows one informs the other.
   - Target: word-initial accuracy above 35 %, and accuracy on contexts seen < 3 times.
   - This is where Lucy stops being a memory and starts to generalize. As far as I can tell
     nobody has done it as sleep-time, tension-driven class formation in an online context
     graph; it would need its own literature check.
6. **Uncertainty as a first-class signal.** Log the entropy of the blended prediction per byte.
   High entropy at a deep cell is the cue for the later "wait / search / ask" behaviour.

## Honest limits

- The validated core is mathematically close to variable-order Markov models (PPM, context trees,
  sequence memoizers). It is strong on 1 MB because child-directed speech is repetitive and the
  model memorizes well.
- It does not understand anything: no generalization across similar contexts and no long-range
  syntax. Step 5 is the attempt at that, and it is unproven.
- CHILDES-triplets is shuffled, independent snippets with heavy repetition. Numbers on continuous
  adult text would be lower.
