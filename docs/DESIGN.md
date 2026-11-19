# Lucy — Phase 1 design

Lucy is a continuously running, CPU-only, sparse, event-driven learning substrate: a graph of
small cells that only talk to neighbours. No layers, no global loss, no backprop, no tokenizer.
**Compute scales with cells that fire, never with total graph size**: every decay is lazy, and
no step of the hot loop touches a cell that did not receive a message.

Four laws:

| Law | Where it lives |
|---|---|
| SPREAD — activation passes neighbour to neighbour | `kernel::spike` sends `gain * h` along out-edges, arriving at `t+1` |
| DECAY — anything not visited fades | `rules::decay_scalars` + `kernel::touch`, applied lazily on touch |
| TRACE — sub-threshold heat is free "I know that I know" | input cells integrate sub-threshold; their trace is read as the next-byte prediction |
| TENSION — predicted vs actual input | `rules::prediction_error` (hidden), `code - u` (input); drives Q and gain learning |

## Module map (`crates/lucy-core`)

| Module | Role |
|---|---|
| `num` | scalar alias `S = f32`, `D = 32`, `V`, `M`, vector math written to auto-vectorize with a fixed summation order. Quantization later = change this file. |
| `store` | `Slots` (free list + generation counters), struct-of-arrays `CellStore`, `EdgeStore`, `Graph` (add/remove cell and edge). `Wave` ancestry tag. |
| `scheduler` | binary-heap event queue on a `u64` logical clock; key `(t, phase, cell, seq)` |
| `rules` | **every** rule and constant (`Params`). Pure functions; swap this module to change the physics. |
| `kernel` | `Engine`: routes events, calls rules, records fires. |
| `buffer` | fire-event ring (`FireEvent { cell, t, h: [i8; D] }`), recording only in Phase 1 |
| `senses` | byte codes, hot-set of input cells, guess and surprise readout |
| `seed` | deterministic initial graph; dormant-cell extension for the scaling benchmark |
| `observe` | windowed stats, JSONL reports, reservoir sample, state hash, snapshot, unigram baselines |
| `probe` | probe ceiling: per-byte fired-cell windows, softmax-regression probe at checkpoints (observatory only) |

`crates/lucy-cli` is the `lucy` binary. `scripts/plot.py` plots a metrics file.

## Data model

Cells are slab slots addressed by `Id { idx, generation }`. Releasing a slot bumps its generation,
so a stale id never aliases the next occupant (needed for future death/split). Edges use the same
slab. Every cell column (`h, w, q, u, heat, log_theta, adapt, rate, v, tau, h_prev, last_t, kind,
wave, fire_pending, gain_norm0, outs, ins, credit`) is a separate `Vec`.

- `log_theta` replaces `theta` (fix 2); `adapt` is the fast threshold term (fix 5).
- `gain_norm0` is the in-gain L2 norm at birth: the synaptic-scaling target (fix 4).
- `credit` is a per-cell list of `(EdgeId, decayed message sum)` since the last fire, one entry
  per contributing in-edge, so its length is bounded by in-degree with no cap constant. It decays
  with exactly the same factor as `u`, which keeps the invariant **`u == Σ credit`** (tested).
- Edges: `src, dst, gain, vitality, last_t` (+ generation). In-flight messages and credit entries
  carry the full `EdgeId`, so removing an edge mid-run silently invalidates them (future-proof).
- `Kind` is `Input | Hidden`. Output cells were removed from Phase 1 (they return as motor cells).

## Clock and event ordering

Single-threaded logical clock. Within one tick, events run in phase order
**Sense → Deliver → Fire**, then by cell index, then by insertion sequence.

- A cell integrates every message of tick `t` before deciding to fire at `t` (decision 1);
  `fire_pending` guarantees at most one fire per cell per tick.
- Fire outputs arrive at `t + 1`: one hop costs one tick, so nothing depends on intra-tick order.
- Byte `i` arrives at tick `i * K` (`--ticks-per-byte`, default 8).
- The clock jumps from event to event; idle ticks cost nothing.

## Rules (all in `rules.rs`)

**Lazy decay** (on every touch, `dt = now - last_t`):

```
u, heat, credit *= exp(-dt / t_heat)          (trace below f32 epsilon is dropped)
adapt           *= exp(-dt / t_adapt)
rate            *= exp(-dt / t_rate)
v               *= exp(-dt / t_v)
horizon   = min(dt, t_rate)
fired     = rate0 * t_rate * (1 - exp(-horizon / t_rate))      // ∫ rate over the interval
log_theta += eta_theta * (fired / target_rate - horizon)       // fixes 1 + 2
theta_eff = exp(log_theta) * (1 + adapt)                       // fix 5
```

The observatory uses the same pure function to *read* state at any time (`Engine::peek`), so
observing never mutates dynamics: the final state hash does not depend on the metrics interval
(tested).

**Message** (to cell `c` over edge `e` at `t`): touch `c`; `u += m`; `heat += |m|`;
`credit[e] += m`; merge the wave tag. Hidden cell with `heat >= theta_eff` and no fire pending →
schedule FIRE at `t`. Input cells never fire from messages; they are marked hot.

**Hidden FIRE** at `t`:

```
e      = u - Q h          // h = state the last fire produced  // tension
Q     += eta_q / (nlms_eps + |h|^2) * e h^T                     // normalized LMS (fix 3)
gain_i += eta_g * dot(-e, m_i) / D   for each credit    // inputs move toward the prediction
gain_i *= gain_norm0 / |in-gains|    for all in-edges   // synaptic scaling (fix 4)
h_prev = h ;  h = tanh(W h + clip_rms(u))               // clip_rms: |u| <= sqrt(D) (fix 6)
tau    = ema(tau, |e|^2, alpha_tau)
rate  += 1 / t_rate ;  adapt += adapt_step ;  v += alpha_v
reset u, heat, credit, wave
for each out-edge: vitality = vitality*exp(-dt/t_v) + alpha_v ; send gain*h, arrival t+1
ring.push(cell, t, quantize(h))
```

**Byte arrival** (byte number `n`, value `b`, at `t`). Learning is O(hot); the readout is a
fixed 256-wide scan (sensory surface size, independent of N):

1. Readout (**competitive**, lateral inhibition across the sheet). Every input cell `c` scores
   a logit for "my byte comes next": `pred_c = Q_c u_c`,
   `score_c = bias_c + dot(pred_c, code_c) / D`. A cold cell has `u = 0`, so it scores exactly
   its bias. `p = softmax(score)` over all 256 cells; guess = argmax (ties → lowest byte);
   Lucy's own surprise is `−log2 p_b` (`readout_bits`).
2. Reality teaches with the softmax (cross-entropy) error `y_c − p_c`, `y_c = 1[c = b]`: the true
   cell up, each hot wrong cell down by its probability. For touched cells (hot ∪ {b}),
   `e = (y_c − p_c) code_c` is the vector error of the readout:
   - `Q_c += eta_readout / (readout_eps + |u_c|^2) * e u_c^T` (regularized NLMS): moves the
     cell's evidence `dot(Q u, code)/D` by `eta_readout (y − p) |u|²/(readout_eps + |u|²)`, a
     cross-entropy gradient step;
   - in-gains: `toward = Q_c^T e`, then `gain_i += eta_g * dot(toward, m_i) / D` and synaptic
     scaling; `tau = ema(tau, |e|^2)`; clear the window.
   Every bias takes `bias_c += eta_bias (y_c − p_c)`, cold cells included: their Q and in-gain
   gradients are exactly 0 (`u = 0`), but the bias gradient is `−p_c`, and without it a byte
   whose cell is never hot would raise its bias at every occurrence and never be pushed down
   (it would end up winning every guess). The biases converge to log-frequencies (tested).
   Cost: O(hot) for Q, gains and credit; one 256-wide softmax and bias pass per byte, the same
   fixed sensory-sheet scan the old readout already made for its priors. (The review asked for
   cold cells via an O(1) running sum of `exp(bias)`; with cold biases learning every byte that
   sum is no longer O(1)-maintainable, and 256 exps cost ≈ 1 µs per byte, < 2 % of a byte's
   work.) Replaces the regression-to-code rule (`pred → code_b`, priors by LMS).
3. Force `b` to fire: `h_prev = h; h = code_b`; fire bookkeeping; send to out-edges at `t + 1`.

Decision 4 (amended): input cells have Q (their readout), still no W. Bias logits live in `Senses`.

## Structural plasticity (`plasticity.rs`)

First slice: co-activation edge growth into input cells. Every `consolidate_every` bytes the
engine replays its fire ring since the last pass (offline, sleep-like):

1. **Replay.** A byte window = hidden cells that fired between two byte arrivals (input-cell fires
   in the ring). Count `N(h, b)` (h fired in the window before byte b), `f_h`, `n_b`, windows `W`.
2. **Grow.** For each byte cell, up to its **tension budget** (below) unconnected hidden cells with the largest
   *excess* co-occurrence `N(h,b) − f_h·n_b/W > 0` get an edge `h → b` (gain `grow_gain`,
   vitality = `N(h,b)`, what it already earned). Excess, not raw count: raw counts would route
   every new edge to the most frequent byte (space).
3. **Prune** each input cell's in-edges: decayed vitality < `prune_vitality` dies; over
   `input_in_cap`, lowest vitality dies first (ties → lower edge index).

**Growth law: tension, not a fixed rate.** At each arrival of byte b its input cell measures its
miss `x = |code_b − pred_b|²/D` (1 = predicted nothing) and updates an error EMA
(`alpha_err` = 0.03 per arrival) and a running **floor**: the floor follows the error down at once
and otherwise decays toward 0 by `alpha_floor` = 0.001 per arrival (an aspiration level). The
excess `err − floor` is summed over the cell's arrivals between passes; the pass grants
`floor(grow_per_tension × Σexcess + carry)` new edges (fraction carried, at most `input_in_cap`).
A cell whose error falls faster than its floor decays gets exactly zero; a stuck cell's budget
grows with how high and how long it stays stuck. First version had the floor creep *up* toward the
error instead: a cell with no in-edges sits at err = floor = 1 forever and never grows (seen: `?`,
984 occurrences in 100 k bytes, zero in-edges) — the decaying floor fixes exactly that case.

**Vitality of edges into input cells is earned, not used:** it is credited (`+alpha_v`, lazily
decayed with `t_v`) when the edge carried a message into the cell whose byte then really arrived,
and not on transmit. With homeostatic firing rates, usage vitality is ≈ equal for every edge
and a new edge (vitality ≈ 1) would always lose to established ones (≈ 500); earned vitality is a
decayed co-occurrence count, the same statistic growth ranks by. Edges into hidden cells keep
usage vitality. No dynamics read vitality, so with consolidation off accuracy is unchanged.

**Hidden → hidden growth (B, off by default: `hidden_grow_per_pass = 0`).** The same replay
counts sequential co-firing `N(A→B)`, i.e. B fired 1..=`hidden_grow_lag` (3) ticks after A. Chance
is `f_A·f_B·lag/T` over the T replayed ticks. Each pass grows the `hidden_grow_per_pass`
unconnected pairs (no self-loops) with the largest excess. Each new edge starts at gain
`grow_gain` with vitality = N. Targets that got an edge then get the input-cell pruning rules:
decayed vitality below `prune_vitality` dies, and over `hidden_in_cap` (16) the lowest vitality
dies first. Hidden in-edges keep usage vitality, and edges from input cells never compete.
Counting every pair costs ≈ 10 M hash increments per pass (≈ 10× slowdown), so each pass only
considers targets with `B % hidden_grow_stride == pass % stride` (8). Every cell is a candidate
target once per 8 passes, at ≈ 12 % overhead.

Cost per pass ∝ fires since the last pass + the 256 input cells' in-edges. Ordered maps and index
tie-breaks keep it deterministic (tested).

## Constants (`Params`, overridable with `--params file.json`)

| Constant | Default | Rationale |
|---|---|---|
| `t_heat` | 4 ticks | Coincidence/trace window. Half of K, so a message landing early in a byte window still counts (e^-2) but the previous byte's prediction is mostly gone by the next. `u`, `heat`, `credit` share it to keep `u == Σ credit`. |
| `t_heat_input` | 4 ticks (= `t_heat`) | Input cells' trace timescale (u, heat, credit), read once at the next byte. Also used by replay for the input hop (seed now looks back one byte window instead of `2·t_heat`). |
| `t_rate` | 2000 ticks | Rate EMA window. ~10 fires at target rate, enough for a stable estimate. Also the homeostasis evidence horizon: the EMA has forgotten anything older, so silence beyond it must not be counted (or theta → 0 after a long sleep). |
| `t_v` | 100 000 ticks | Vitality is a slow usage memory (future: death/pruning). Long compared with everything else. |
| `t_adapt` | 50 ticks | Fast adaptation must outlast an avalanche (several byte windows) but be gone by the next normal fire (1/target_rate = 200 ticks: e^-4 ≈ 0.02 left). |
| `target_rate` | 0.005 /tick | **The compute-budget knob**: expected fires per tick ≈ N × target_rate (≈ 20 at 4096 hidden). Measured: 20.5. |
| `eta_theta` | 1e-3 /tick | Log-domain, so dimensionless. A silent cell drops theta by e^-2 over one rate window; a cell at 10× target raises it ~1 % per tick. |
| `adapt_step` | 1.0 | One fire doubles the effective threshold immediately; sustained firing every tick saturates near `1 + adapt_step * t_adapt`, which kills runaway loops locally. |
| `eta_q` | 0.1 | NLMS step, dimensionless, stable for (0, 2). Small: Q should average over many fires. |
| `nlms_eps` | 1e-3 | Makes `h_prev = 0` a no-op; tiny compared with `|h|^2 ≤ D`. |
| `eta_g` | 0.05 | Credit step. Normalised by D; synaptic scaling removes any net growth. |
| `alpha_tau` | 0.05 | tau averages over ~20 fires. |
| `alpha_v` | 1.0 | Unit vitality per use (scale is arbitrary until death exists). |
| `consolidate_every` | 1000 bytes | Consolidation period. Must replay before the ring wraps: 1000 bytes × K × ~21 fires/tick ≈ 170 k events < 262 k ring slots (`consolidation_lost_events` reports any loss). 0 = off. |
| `grow_per_tension` | 1.0 | Edges per unit of summed error excess (sweep: 0.01–64 all give 0.30–0.35 accuracy at 200 k bytes; 1.0 best mean over seeds 7, 11). |
| `alpha_err` | 0.03 | Error EMA per own arrival: ~30 occurrences, enough to average one byte's contexts. |
| `alpha_floor` | 0.001 | Floor decay per own arrival: a stuck cell's budget ramps up over ~1000 occurrences (≈ 1 pass for space, ≈ 100 passes for a rare byte). |
| `grow_gain` | 0.1 | New edges start small; synaptic scaling then folds them into the cell's fixed gain budget. |
| `prune_vitality` | 0.5 | An in-edge of an input cell must keep earning: a single co-occurrence decays below 0.5 after ≈ 69 k ticks (≈ 8.7 k bytes). |
| `input_in_cap` | 32 | Homeostatic in-degree cap of input cells (2× the seed fan-in). Over it, the lowest-vitality edge dies. |
| `hidden_grow_per_pass` | 0 (off) | Hidden→hidden edges per pass. 16/64/256 tested: stable, no probe gain at 200 k bytes. |
| `hidden_grow_lag` | 3 ticks | "B fires at t+1..t+3 after A": one to three hops of delivery plus integration. |
| `hidden_in_cap` | 16 | ≈ 2× the seed hidden fan-in (≈ 8). |
| `hidden_grow_stride` | 8 | Cost knob: 1/8 of targets per pass. |
| `replay_per_pass` | 0 (off) | Wayback segments per pass (most surprising arrivals of the pass). |
| `replay_bytes` | 4 | Segment length (bytes before the anchor) = dream length. |
| `eta_w` | 0.01 | W NLMS step through tanh; W norm capped at its birth value. |
| `replay_gamma` | 0.5 | Per-hop decay of the feedback signal. |
| `replay_min_delta` | 1e-3 | Feedback RMS below which the chain stops. |
| `dreams_per_pass` | 32 | Pre-update dreams recorded per pass (dream consistency). |
| `dream_seeds` | 8 | Hidden cells seeding one dream. |
| `eta_bias` | 0.01 | Bias-logit cross-entropy step (competitive readout); 0.003–0.03 swept, 0.01 best. |
| `eta_readout` | 0.3 | Readout Q step: evidence moves by ≈ `eta_readout (y − p) |u|²/(readout_eps + |u|²)` per touch. Separate from hidden `eta_q` (sharing it changed the tissue). Swept 0.03–0.3. |
| `readout_eps` | 1.0 | Regularizer of the readout normalization. With the hidden-cell `nlms_eps` (1e-3) a weak input took a full step that strong inputs then amplified: own surprise 4.6–6.8 bits (overconfident). 0.3–3 all give 3.4–3.5 bits. |

Seed constants (`SeedConfig`, CLI flags): 4096 hidden, fanout 8 (hidden→hidden, distinct, no
self-loops), 16 input→hidden per input, 16 hidden→input per input (prediction pathway), gains
U(−0.5, 0.5), W and Q entries U(−a, a) with a = 0.5/√D and 0.1/√D (spectral radius ≈ 0.5),
initial theta 1.0 (homeostasis moves it), h = 0, initial rate = target (neutral homeostasis),
vitality 1.0. Byte codes have entries in {−1, +1} (inside tanh range; near-orthogonal in 32 dims).
Every random draw comes from a dedicated ChaCha8 stream (`graph`, `codes`, `reservoir`, `dormant`),
so adding dormant cells or changing the reservoir never perturbs the graph.

## Observatory

Every M ticks (`--metrics-every`), one JSON line. The first line is a header with the corpus
**unigram entropy** (surprise baseline), **unigram top-1 accuracy** (accuracy baseline), seed
config and params. Metrics:

- `accuracy` (primary): next-byte guess accuracy per byte in the window.
- `surprise_bits`: `-log2 softmax(ln(max(score, 1e-4)) / tau)[true byte]`, where `tau` is fitted
  online **in the observatory only** (`observe::Calibrator`). Scores are LMS probability estimates,
  so they enter in the log domain (a linear `score / tau` softmax cannot express the thousands-fold
  ratios between byte probabilities). Each byte is scored with the current `beta = 1/tau` first
  (prequential), then `beta` takes a damped stochastic Newton step on the log-loss
  (`grad = E_p[x] - x_true`, curvature = running `Var_p[x]`, rate 0.01 ≈ 100 bytes of memory).
  The dynamics never read it, so changing the calibrator cannot change the state hash.
  `surprise_temperature` reports `tau`.
- `tension`: mean |e|² of hidden fires in the window.
- `hop_depth`: for each inter-byte window, the furthest shortest-path hop the *current* byte's
  wave reached before the next byte, averaged over windows. If it approaches K, K is too small.
  `reverb_fraction`: share of hidden fires whose freshest ancestry is an older byte (or none),
  i.e. self-sustained activity.
- Activity: `fires_per_tick`, `max_fires_per_tick` (avalanche watch), `cells_fired_in_window`.
- `fired` = exact stats over hidden cells that fired in the window; `reservoir` = a fixed,
  seed-derived sample of 1024 hidden cells (O(1024) per report). Each gives tau mean/p50/p90/p99
  and mean theta_eff, v and rate.
- `wall.events_per_sec`: wall-clock, excluded from the determinism hash.

**Two headline metrics, every run.** *Routing* = Lucy's own readout top-1 (`accuracy`).
*Tissue* = the **probe ceiling**: top-1 of an offline linear probe on which hidden cells fired in
the K ticks before each byte, i.e. how much next-byte information the reservoir holds whether or
not Lucy routes it. Both are read against the uni/bi/trigram lines. `lucy probe` (or
`--probe-every N` on any run/sweep) records one window per byte from the ring (O(fires per byte))
and at every N-byte checkpoint trains on the last `--probe-samples` windows (default 200 k) with
the `scripts/probe.py` protocol: temporal 80/20 split, L2-normalized binary rows, softmax
regression, minibatch SGD (batch 256, 6 epochs, lr 0.2/(1+epoch)), plus a one-hot previous-byte
control probe and train-fitted count baselines on the same split. Each checkpoint writes a
`{"kind": "probe"}` line (unhashed: observation only) with `probe_accuracy`/`probe_bits`,
`readout_accuracy` (Lucy on the *same test bytes*), control and baselines, and the
**edge-restricted probe** (`edge_probe_accuracy`): the same trainer, but byte b's logit may only
read hidden cells that have an edge into input cell b at the checkpoint (masked weights start at 0
and never move). It is the ceiling of the *current topology* for a competitive linear readout;
full probe ≥ edge probe ≥ what Lucy's routing could reach with this wiring; `--dump` adds the raw
windows in `probe.py` format, `--snapshots` a cell snapshot. Equivalence with the numpy reference:
on byte-identical windows (CHILDES 300 k–500 k, seed 7) the Rust probe gives 44.45 % / 2.792 bits
vs probe.py 44.52 % / 2.793 (shuffle-order noise; two probe.py runs on identical dynamics differ
by the same 0.07 %), control 41.05 % vs 41.02 %, count baselines identical. ≈ 9 s per 200 k-sample
checkpoint.

**Path diagnostics (`--probe-paths`, observatory only).** Probe rows keep each fire's tick
offset k before the byte (row code `cell << 4 | k`), so each checkpoint also reports:
- the **time-resolved edge-restricted probe**: the edge-restricted probe on cells fired exactly
  k ticks before the byte, k = 1..K;
- the **bigram-path probe**: full probe on only the direct hidden targets of the previous byte's
  input cell (its ceiling should be ≈ the bigram);
- a **path audit** over the bytes since the last checkpoint: the share of the true cell's
  hidden in-edges whose source is a direct target of the previous byte, and the message heat
  those edges (and all in-edges) carry at their arrival tick vs what is left at guess time,
  plus how many ticks before the guess the bigram-path fires happen.

`snapshot` writes one JSON line per live cell (theta, rate, v, tau, heat, degrees, gain norms).

## Determinism

Seeded ChaCha8 for every random draw; no hash-map iteration anywhere; float summation orders fixed
(8-lane dot, event order, list order). `check-determinism` runs twice and compares the FNV-1a hash
of every metrics line (minus wall clock) plus the full final state (cells, edges, ring, pending
events). libm `exp`/`tanh` make runs bit-identical for the same binary on the same CPU
architecture, so `rust-toolchain.toml` pins the compiler exactly (1.99.1) and `flake.nix` provides it.

## Context cells (`context.rs`, v2 byte-stream tissue; `tissue = "context"`, default)

From `docs/PROPOSAL-v2.md`: a cell's identity is its context. Cells form a suffix trie. The root
is the empty context, and the child of a cell for byte y extends its context one byte further
into the past. Each cell keeps `parent, depth, last_t, total`, plus `children` and `succ`
(successor counts), both as small lists sorted by byte. At each arrival of byte x:
1. **SPREAD.** Walk the chain from the root through `children`, keyed by the newest history
   byte, then the next older one, and so on. This gives every live suffix of the last
   `ctx_depth` (K = 8) bytes: ≤ K + 1 cells, no hashing of history, O(K) per byte.
   - Forward spread (c → c + x) is *not* used. It cannot reach a cell whose context minus its
     newest byte has no cell, so its chain would differ from the reference's.
2. **TRACE.** The chain votes, each cell with weight `w = T/(T+2) · 2^depth / T` on its decayed
   counts. Then `p = (score/Z)(1 − ε) + ε/256` with ε = 1e-3, written to the 256-byte readout
   sheet. The logits passed to the observatory are `ln p`, so `readout_bits` is exactly the
   reference's bits.
3. **Learn.** Every chain cell counts x.
4. **TENSION.** If the deepest cell mispredicted and has ≥ `ctx_grow_min` (2) evidence, it grows
   one child one byte deeper. Under the budget `ctx_budget` (2 M cells) the miss grows nothing
   and is counted in `context_starved`.
5. **DECAY.** Counts decay lazily as `exp(−dt/ctx_t_v)`, with dt in bytes, applied when a cell is
   touched (`ctx_t_v` = 1e7; 0 turns decay off). The growth gate counts undecayed visits.
   - Every `consolidate_every` bytes, sleep visits the next `ctx_sweep` (4096) slots and prunes
     leaves whose vitality is below `ctx_prune_floor` (0.5). Freed slots are reused first.
   - The sweep size is fixed, so pruning costs O(1) per byte at any cell count.

**The reservoir is off the byte path (step 2).** With `tissue = "context"` no hidden cells are
seeded, the arriving byte spikes nothing, and reservoir consolidation does not run. Each byte is
one Arrive event. The Phase 1 path is `--tissue reservoir` (or `"tissue": "reservoir"` in
`--params`). Its metrics are identical to the pre-v2 binary (`f270859`): all 40 windows of a
400 k-tick CHILDES run match. The state hash differs only because it now covers the (empty)
context tissue. `lucy probe` and `bench-scaling` measure the reservoir and refuse to run without
`--tissue reservoir`. W/Q cells are meant to return as the encoder for continuous senses.

**Sleep is structural only (step 3).** Sleep never adds evidence: replaying misses as extra
counts made the prototype worse (PROPOSAL-v2, negative result 1). It visits `ctx_sweep` slots per
pass and removes leaves in two ways:
- **prune**: vitality (decayed count) below `ctx_prune_floor`;
- **fold** (`ctx_fold_bits` > 0, off by default): the leaf saves fewer bits over its parent
  than the threshold, measured as vitality · KL(p_leaf ‖ p_parent). Its counts are already in
  the parent, because every chain cell counts every byte, so folding loses only that.

Fold, CHILDES 1 MB (default decay 1e7, 900 k–1 M window):

| `ctx_fold_bits` | top-1 | bits | cells | folded |
|---|---|---|---|---|
| 0 (off) | 70.6 % | 1.504 | 150 365 | 0 |
| 0.25 | 70.4 % | 1.508 | 147 446 | 8 326 |
| 0.5 | 70.4 % | 1.511 | 143 727 | 15 647 |
| 1 | 70.0 % | 1.526 | 102 338 | 80 213 |
| 2 | 69.3 % | 1.575 | 60 568 | 161 003 |

- **Decision (2026-11-19): fold is dropped.** The code stays, off (`ctx_fold_bits = 0`). In a
  count-based context tree a redundant child costs memory, not accuracy, and a folded cell
  regrows without its history. Pruning by decay stays.
- **Fold does not pass "same or better accuracy with fewer cells"** at any threshold. It is
  a cells-for-accuracy trade, so it stays off.
- A prequential variant was worse and was not kept. It credited each leaf with the bits it
  actually saved over its parent while it was the deepest cell, then folded leaves with negative
  net credit. Folding only leaves below −2 bits cost −0.5 points for −3 % cells.
- The reason: cells only grow where Lucy was wrong, and a folded cell regrows at count 1 on the
  next miss, losing its evidence.
- **Pruning by decay does pass the test**: `ctx_t_v` 1e6 gives 70.5 % / 1.505 bits with
  143 701 cells, against 150 743 cells with no decay at the same 70.5 %.

**Chunks (step 4, `ctx_chunk`, off by default).** Boundaries come from one of two rules:
1. after a non-alphanumeric byte;
2. from Lucy's own tension: a byte whose surprise `−log2 p(x)` exceeds `ctx_chunk_bits` starts a
   new chunk, i.e. a transitional-probability dip (Saffran et al. 1996).

The tissue tracks the lengths of the previous and the current partial chunk. The chain may then
reach back to the start of the previous chunk, up to `ctx_chunk_cap` (24) bytes, instead of K.
The deepest cell of such a chain is the chunk cell keyed by (previous chunk, current partial
chunk), and it grows like any other cell. Past depth K the vote weight is capped at `2^K`.
Uncapped, one count in a depth-15 cell outvoted the whole chain (−0.2 points with spaces, −0.8
with tension).

CHILDES 1 MB, last window (capped weights):

| | top-1 | bits | cells |
|---|---|---|---|
| K 8 | 70.6 % | 1.504 | 150 365 |
| K 8 + chunks, spaces | 70.6 % | 1.511 | 193 106 |
| K 8 + chunks, tension 3 bits | 70.7 % | 1.512 | 261 184 |
| K 4 | 67.9 % | 1.631 | 22 136 |
| **K 4 + chunks, spaces** | **70.1 %** | 1.521 | 159 871 |
| **K 4 + chunks, tension 3 bits** | **70.1 %** | **1.516** | 197 564 |
| K 12 (uniform depth) | 70.2 % | 1.572 | 249 379 |

- **The mechanism works where fixed depth is short.** At K = 4, chunks add +2.2 points and
  −0.11 bits.
- **Tension boundaries are as good as spaces.** Lucy's own surprise peaks segment the stream.
- **Decision (2026-11-19): chunks stay off by default.** The self-derived surprise boundaries
  are kept, because hierarchy will need them later.
- **At K = 8 chunks are neutral, so they stay off.** Child-directed words are short, and 8 bytes
  already span most (previous word, partial word) contexts.
  - Chunk-anchored depth still beats uniform depth: K 12 is worse than K 8.
- **Not kept: growing the whole path to the anchor at once**, which creates the chunk cell on the
  first miss. Neutral to −0.1 points, at 1.7–7× the cells.
- The proposal's +0.7 points with space boundaries is not reproduced at K = 8.

**Exact port first (`a3b0897`).** Successors kept in first-seen order reproduce
`scripts/context_cells_reference.py` on CHILDES 1 MB at every 100 k checkpoint: identical cell
counts (27 468 … 152 039), and top-1 and bits equal to every printed digit (70.5 % / 1.517 bits
at 1 MB). The native version sorts lists by byte, so ties go to the lowest byte; with decay off
it gives 70.5 % / 1.504 bits.

## Save/resume (`save.rs`)

`--save PATH` writes the whole run state at the end of the run, and also every `--save-every`
bytes if that is set: engine (cells, edges, scheduler wheel + overflow heap + per-tick arenas,
ring, senses, plasticity incl. anchors, logical clock), observatory (window counters, calibrator,
baseline window sums, metrics hash) and the prober's windows. The format is bincode behind an 8-byte magic
`LUCYSV03`, written to a temp file and then renamed. Nothing else needs saving: the decay tables are
rebuilt from `Params`, and every RNG draw comes from `stream(seed, id)` at its point of use.
`--resume PATH --ticks <absolute end>` continues the run: graph flags, seed and params come from
the save, the corpus skips the `bytes_total` bytes already consumed, and metrics are appended.
The save point comes before the final partial report, so the resumed metrics lines match an
uninterrupted run's. The interrupted file additionally has one partial line at the cut.
- Test `save_resume_is_bit_identical`: N ticks → save → load → M ticks equals N + M ticks,
  with plasticity and wayback on. Both the state hash and the metrics hash must match.
- CHILDES, 4096 cells: a run to 400 k ticks and a run cut at 160 k (saving every 7 000 bytes) then
  resumed give the same state hash `a787ae53737c02a7`.
- A save is ≈ 123 MB at 4096 cells, mostly W/Q and the ring.
- **The count baselines are not saved.** The trigram table alone was 64 MB of a 71 MB context
  save. On resume they are rebuilt by replaying the `bytes_total` corpus bytes already consumed:
  deterministic, ≈ 30 ns/byte. A 1 MB context tissue now saves in 13 MB.
- A save made by `lucy talk` has empty baselines, because no corpus goes with it.
- For the 68 MB run: `--save runs/d.bin --save-every 2000000`.

## Scaling

`lucy bench-scaling` builds the same seed graph, adds (k−1)×N dormant hidden cells wired only
among themselves, and streams the same corpus. Activity must be identical (asserted) and ns/event
near-constant. A unit test also asserts that dormant cells are never touched (`last_t` stays 0).

## Phase 2 hooks already in place

- Ring buffer with global sequence numbers and generation-tagged cell ids (wayback replay).
- Slab generations, edge/cell removal, in-flight invalidation (growth, split, death).
- Vitality on cells and edges (pruning, sleep).
- Per-edge credit vectors (deeper credit assignment).
- All state behind SoA columns and the `S` alias (cold-tier mmap, quantization).

## Reverse replay ("wayback") with one-hop feedback into W (`replay.rs`, built; off by default: `replay_per_pass = 0`)

Review changes (required): (a) within each consolidation pass, dreams are generated with the
**pre-update W** and their transitions recorded *before* any feedback update; after the update,
W is pulled toward reproducing those recorded pre-update transitions (pulling toward dreams
generated after the update would protect nothing). (b) Replay anchors are ranked by the
competitive readout's own surprise `−log p(true byte)`. Success metric: tissue (full probe
ceiling) rises above 44.5 % while the forgetting measure stays flat.

Goal: make W learn (open finding 1) from reality, offline, without global passes, using only
what one cell can see. It runs inside consolidation, after growth and pruning.

**1. What is replayed.** Ring segments that *end in a high-tension event*. Anchors are the byte
arrivals with the largest competitive-readout surprise `−log p(true byte)`. During each pass the kernel keeps the
top `replay_per_pass` (≈ 32) anchors in a bounded heap: O(log R) per arrival. Each anchor also
keeps its ring sequence number and the readout error vector `e_b`. A segment is the ring
events in `[t_a − replay_bytes·K, t_a]` (≈ 4 bytes = 32 ticks ≈ 650 fires). The ring holds
≈ 1500 bytes and passes run every 1000, so anchors from the last pass are still retained. Lost
ones are skipped and counted.

**2. Order.** Strictly reverse in time: anchor tick first, then t_a−1, t_a−2, … Within a tick,
cells go by index, so the order is deterministic.

**3. The one-hop feedback signal, exactly.** Each replayed fire event carries a *desired change
of the state it produced*, `δ ∈ R^D`. It moves back exactly one hop per replay step and is
computed only from the receiving cell's own quantities and the connecting edge's gain:
- Seed (anchor, input cell b): the desired change of b's input is `Qᵀ_b e_b`, the `toward`
  vector `arrive` already computes. For every in-edge h→b whose message was in b's trace window
  (the cell's `credit` list at the anchor, or the ring: h fired in `[t_a − t_heat·2, t_a)`):
  `δ_h += g_hb · Qᵀ_b e_b`.
- Hidden hop: B fired at t with recorded state `h_B`, and A fired at t′ in
  `[t − t_heat·2, t)` with edge A→B. The pre-activation of B moved by `clip(u_B)` with
  `u_B ∋ g_AB · h_A`, so A's share of B's desired change is
  `δ_A += γ · g_AB · ((1 − h_B²) ⊙ δ_B)`. Here `γ ≈ 0.5` is a per-hop decay, and the clip factor
  counts as 1 when `u_B` was not clipped.
- After a tick is processed, each accumulated `δ` is RMS-clipped to 1 (no explosion) and dropped
  below `1e-3` RMS (the chain ends where the signal does). The chain never reaches further back
  than the segment.

This is backpropagation through time cut into local one-hop messages along the recorded
trajectory. Every term is one cell's own state or one edge's gain. Nothing global is formed.

**4. How W updates (local).** For a replayed fire of A with `δ_A`, its recorded previous state
`h_A⁻` and new state `h_A`, use NLMS through tanh on A's own map:
`ΔW_A = η_w · ((1 − h_A²) ⊙ δ_A) h_A⁻ᵀ / (ε + |h_A⁻|²)`, with `η_w ≈ 0.01` (≪ `eta_q` = 0.1, so
Q can track the drift). `h_A⁻` is A's previous ring entry. `FireEvent` gains `prev_seq`, set from
a per-cell `last_seq` column, so the lookup is O(1); if that entry was overwritten the update is
skipped. Then comes a W norm homeostasis like synaptic scaling: `W_A ← W_A · min(1, ‖W_A⁰‖_F/‖W_A‖_F)`,
using the birth norm. That keeps each cell's spectral contribution at most where it started. Ring
states are i8 (`h·127`), which costs about 0.4 % quantization noise in the update. Optional, and
not in the first slice: edge gains on the hop get `Δg_AB ∝ dot((1 − h_B²) ⊙ δ_B, h_A)/D`.

**5. Protecting old sequences: interleaved dreaming.** Each pass interleaves real reverse replays
with an equal number of *self-generated* ones:
- Dream generation: pick k ≈ 8 random hidden cells from a dedicated `DREAM` ChaCha stream. Force
  them with their current h, then run the network *forward* in a sandbox: a scratch scheduler and
  ring, no sensory input, homeostasis and all learning frozen, the same rules module. It runs for
  `replay_bytes·K` ticks, with fires per tick capped at the target budget. Each dream fire records
  `(h⁻, u, h)` at full precision. The dream is produced *before* the real replays update W.
- Dream consistency: after the real replays, the dreamed transitions are recomputed with the new
  W, `ĥ = tanh(W h⁻ + clip(u))`. The same NLMS rule then pulls each one back toward its recorded
  value, `δ = h − ĥ`, without chaining. This removes, on the network's own spontaneous
  trajectories, exactly the interference the real update caused. It is pseudo-rehearsal: what
  the network already does by itself is protected without storing old data.
- Dream cost is bounded by `replay_per_pass × replay_bytes·K × target fires` per pass, i.e. it
  scales with activity.

**6. Stability risks and their guards.**
1. W spectral growth → saturated or chaotic reservoir. Guard: Frobenius norm cap at the birth
   norm. Watch: saturation fraction (|h| > 0.95), branching ratio, fires per tick.
2. Vanishing or exploding δ along the reverse chain. Guard: per-hop γ, RMS clip, drop threshold,
   segment-length limit. Watch: mean ‖δ‖ by hop.
3. Moving target for Q (W shifts the distribution of h that Q reads). Guard: η_w ≪ eta_q.
   Watch: tension and readout accuracy right after each pass (a sawtooth means η_w is too high).
4. Interference with old sequences. Guard: dream consistency. Watch: the probe on a *fixed early
   segment* re-run at checkpoints, and the readout on held-out early text.
5. Off-policy replay: ring states were produced by an older W. Guard: replay only the last pass's
   anchors; skip segments older than `t_v`.
6. Homeostasis coupling: changing W changes firing rates, theta adapts, and theta is already low
   (≈ 0.06). Watch theta per pass. Freeze homeostasis during dreams.
7. Dreams that die or avalanche without input. Guard: seed count k, a fires-per-tick cap, a tick
   cap. Watch: dream activity per pass.
8. Credit only reaches cells that fired. Cells that *should* have fired get nothing (sparse by
   design). The first slice accepts this.
9. Determinism and cost: fixed budgets, ordered iteration, a dedicated RNG stream. Test:
   `η_w = 0` leaves the state hash unchanged; a hand-computed two-hop δ; replay off/on
   determinism.

**Readout consolidation (step 4 of the pass, `replay_readout`, on with replay).** The readout has
to be consolidated in the same sleep pass, or it reads states that W no longer produces (routing
fell 36 → 29 % without it). After the W updates and the dream pull-back, every replayed segment is
re-run *forward* with the recorded spike timing. That covers who fired when, including the byte
arrivals, which the ring already records as the input cells' forced fires (cell = byte, t =
tick). States are recomputed under the new W: `h = tanh(W h⁻ + clip(u))`, u from the recomputed
in-neighbour states in the trace window, and no tissue learning. At each recorded byte the
input-cell traces are rebuilt from the recomputed hidden fires since the previous byte. The
competitive readout scores them against the current biases, and each hot cell's Q takes the
usual `eta_readout` step on `(y − p) code`. Biases are not replayed, because surprising segments
would skew the frequencies; in-gains are not replayed either. **Segments for the readout
(`readout_uniform`, default on):** a uniform sample of the pass's byte arrivals, as many as
`replay_per_pass`, not the surprise-ranked W-replay segments. Trained only on the hardest
segments (forward top-1 21.6 %), Q saw a skewed sample. W feedback keeps the surprise ranking. Check: with W frozen, the forward
re-run reproduces the recorded states to 0.0065 mean |Δh| per component (i8 quantization and
window truncation). Metrics: `replay_readout_updates`, `replay_readout_accuracy`,
`replay_state_err`.

**As built.** `FireEvent.prev` links each ring fire to the same cell's previous fire (a
`last_seq` column), so `h⁻` is an O(1) lookup; `w_norm0` holds the birth norm. Anchors are kept
by `kernel::arrive` (top `replay_per_pass` by `−ln p(true byte)`) and consumed by the pass.
Dreams run in a sandbox with each cell firing at most once per dream (frozen thresholds would
otherwise let a loop fire every tick). The clip factor of `clip_rms(u)` is taken as 1 (u is not
recorded in the ring). Forgetting measure (`--probe-old N`): at each checkpoint a clone of the
engine with consolidation off (W frozen) re-reads the first N corpus bytes; probe and readout on
that fixed segment are reported as `old_probe_accuracy` / `old_readout_accuracy`. Tests: a
hand-computed two-hop replay (W deltas to 2e-6), `eta_w = 0` leaves the state bit-identical to
replay off, replay on is deterministic. New metrics: `replay_segments`, `replay_lost`,
`replay_w_updates`, `replay_delta_rms`, `dream_fires`, `dream_updates`, `saturation`,
`w_norm_ratio`.

**Measure before/after:** routing and tissue headlines; the probe ceiling is the
direct test, since W shapes what the tissue holds. Also the early-segment probe (forgetting),
saturation, branching, and theta.

## Results history

All learning measurements use CHILDES child-directed speech (`corpus/childes-1mb.txt`,
1 000 000 bytes, speaker tags kept). Baselines for that file: unigram entropy **4.683 bits**,
unigram top-1 accuracy **20.96 %** (always guess space), in-sample bigram oracle top-1 41.2 %
(conditional entropy 2.73 bits, a ceiling reference only).

### Engineering corpus (history only)

`phase1-baseline` on 1 MiB of Newton's *Opticks* + Neovim docs (unigram 4.747 bits, top-1
17.2 %): 2.5 M events/s, rate at target, no death or avalanche, but accuracy at chance (0.1 %)
and σ-softmax surprise 15.4 bits. Readout experiments on the first 50 k bytes: heat, alignment
`dot(u, code)`, with or without input-cell scaling all at chance; per-input-cell Q readout ≈ 10 %
accuracy. This led to the commits below.

### CHILDES, seed 7, 1 000 000 bytes, one commit at a time

Mean of the last 10 report windows (100 k ticks each).

| Commit | Change | Accuracy | Surprise (bits) | Hidden tension |
|---|---|---|---|---|
| `620087c` (`phase1-baseline`) | rules as approved | 0.11 % | 16.37 (σ-softmax) | 3.72 |
| `03f61de` | input-cell Q readout + prior | **22.23 %** | 11.99 (σ-softmax) | 3.72 |
| `295ff41` | calibrated surprise (observatory only) | 22.23 % | **6.69** (calibrated) | 3.72 |
| `3c3db0a` | prediction-lag fix | 22.20 % | 6.66 | **2.06** |

- The readout beats unigram top-1 (20.96 %), but only by ≈ 1.3 points (best window 24.3 %).
- `295ff41` has the same final state hash as `03f61de` (`6a60a0d83cfd6644`): calibration is
  observatory-only, as intended.
- The lag fix cuts hidden tension by 45 % (Q predicts much better) but does not change accuracy:
  hidden prediction quality does not reach the readout (see finding 2).
- Throughput with four runs in parallel: 1.5–1.8 M events/s per run.

### Diagnosis: is context in the reservoir? (linear probe)

`examples/probe_dump.rs` records, for each byte of CHILDES bytes 300 k–500 k (seed 7), the hidden
cells that fired in the K ticks before it; `scripts/probe.py` (numpy, softmax regression, plain
SGD) trains on the first 80 % and tests on the last 20 %. Count baselines are fitted on the same
split. Dumps at HEAD and at `3c3db0a` are byte-identical (no dynamics change since).

| Predictor | Top-1 | Bits |
|---|---|---|
| unigram / bigram / trigram (counts, back-off) | 20.9 % / 41.1 % / 57.0 % | 4.68 / 2.80 / 2.31 |
| control probe: one-hot previous byte | 41.0 % | 2.77 |
| reservoir probe at `3c3db0a` (= HEAD) | **44.5 %** | 2.79 |
| reservoir probe at `phase1-baseline` | 44.5 % | 2.79 |

The control matching the bigram validates the probe. The reservoir's recent firing linearly
encodes more than bigram context, but Lucy's readout reached only 22 %: **routing, not
representation, was the bottleneck.** This motivated structural plasticity.

### Structural plasticity (`e05d23f`), CHILDES 500 k bytes, seed 7, last 10 windows

| | Accuracy | True cell hot | Hot inputs/byte | Lift vs random | Input in-degree |
|---|---|---|---|---|---|
| `consolidate_every = 0` | 22.1 % | 38.5 % | 97.7 | 1.01× | 16.0 |
| `consolidate_every = 1000` | **33.7 %** | **69.0 %** | 15.3 | **11.5×** | 5.9 |

With log-domain calibration (`3d54c54`) surprise is 3.83 bits (unigram 4.68, online KT bigram 2.76,
trigram 2.12). Accuracy reaches ≈ 32 % within 12.5 k bytes and plateaus. (Corrected later: the
mean in-degree ≈ 6 hides a bimodal split, see below; the cap *does* bind for frequent bytes.)

### Growth law A (`61550d0`): tension-driven growth, CHILDES 1 MiB, seed 7

Headlines: *routing* = Lucy readout top-1 and *tissue* = probe ceiling, both on the probe's
test bytes at each 100 k checkpoint (probe = last 200 k windows).

| | routing (mean of 8 checkpoints) | tissue | readout, windows 750–800 k | true cell hot | in-edges/input cell |
|---|---|---|---|---|---|
| fixed rate (`a9f823d`) | 32.6 % | 44.2–45.1 % | 33.4 % | 68.0 % | 5.9 |
| tension law | 32.4 % | identical | 32.0 % | 65.0 % | 5.4 |

- **Tissue is bit-for-bit unchanged.** Edges into input cells never reach the hidden dynamics
  (input cells fire only when their byte arrives), so any input-side plasticity leaves the probe
  ceiling where the fixed reservoir puts it: ≈ 44.7 %.
- **Where the 12.5 k plateau comes from.** All 16 seed hidden→input edges per byte start with
  vitality 1. Unless they earn, they decay below `prune_vitality` at ≈ 9–12 k bytes. After that,
  growth and pruning balance (grown ≈ pruned in every window).
- **In-degree is bimodal, not ≈ 6.** At 100 k bytes, 33 bytes sit at the cap (32) and churn. 29
  *seen* bytes have zero in-edges, among them `?` with 984 occurrences. The other 187 byte values
  never occur.
- **The growth law works as a law, but does not lift accuracy.**
  - Budgets go to stuck cells, and a cell with falling error gets 0.
  - Sweeping `grow_per_tension` 0.01–64 × `alpha_floor` {0.001, 0.01} gives 30–35 % for every
    setting at 200 k bytes. That spread is the window-to-window noise.
  - Raising `input_in_cap` from 32 to 128 lifts true-cell-hot to 85 % (seeds 7 and 11), and
    accuracy still does not move (30–34 %).
- **Conclusion: the plateau is in the readout, not in growth or coverage.** The per-arrival
  vector miss `|code − Q u|²/D` stays at ≈ 0.96, so `dot(Qu, code)/D` adds only a few hundredths
  to priors that differ by tenths. The readout cannot use the context its in-edges deliver. The
  probe gets 44.7 % from the same firing with a joint 256-way softmax over all cells.

### Hidden→hidden growth B (`c81d1e3`), CHILDES, seed 7 (and 11 at 200 k)

| `hidden_grow_per_pass` | tissue @200 k (s7 / s11) | tissue, 1 MiB mean of 10 checkpoints | routing (1 MiB) | hidden in-degree | reverb | theta | max fires/tick | branching |
|---|---|---|---|---|---|---|---|---|
| 0 | 44.45 / 45.03 % | **44.5 %** | 32.4 % | 9.0 | 34 % | 0.069 | 298 | 0.95 |
| 16 | 44.52 / 44.92 % | | | 9.5 | | | | 0.95 |
| 64 | 44.47 / 44.67 % | | | 10.4 | | | | 0.96 |
| 256 | 43.87 / 44.92 % | 43.9 % | 32.7 % | 13.4 | **19 %** | **0.124** | 214 | 0.96 |

(Reverb, theta, max fires and branching are means over the last 100 k bytes.)

- **The probe ceiling does not rise above 44.5 %.** At 256 per pass it is 0.6 points lower at
  every 1 MiB checkpoint.
- **Recurrence added this way is safe.** Fires per tick stay at 20.5 (homeostasis holds the
  budget), the branching ratio stays at 0.95–0.96 (subcritical), and max fires per tick go
  *down*.
- **It improves both watch items.** Reverb falls from 34 % to 19 %, and theta rises from 0.069 to
  0.124. Edges that follow existing sequences make waves more byte-locked, so less activity
  sustains itself.
- **Kept, default off; noted as a tissue-health improvement** (reverb 34 % → 19 %, theta 0.069 →
  0.124 at 256/pass): a candidate to switch on together with C, where W learning may need the
  calmer, more byte-locked reservoir.
- **Reading.** "Fires after A more often than chance" mostly duplicates what the existing
  dynamics already do. It sharpens the trajectories it copies, but adds no new next-byte
  information. Information has to come from a teacher signal, which motivates plan C:
  reality-anchored W learning.

### Competitive readout (`ac1bb4c`) vs regression-to-code, CHILDES 1 MiB, seed 7

Three-way split at each 100 k checkpoint: *routing* (Lucy's readout on the probe's test bytes),
*edge-restricted probe* (ceiling of the current wiring), *full probe* (tissue). Means of 10
checkpoints:

| readout | routing | edge-restricted probe | full probe | gap to edge probe | Lucy's own surprise | calibrated surprise |
|---|---|---|---|---|---|---|
| regression to code + LMS prior (`909c19a`) | 32.4 % | 38.6 % | 44.5 % | 6.2 pts | n/a | 3.87 bits |
| competitive (softmax) | **36.4 %** | 37.8 % | 44.5 % | **1.4 pts** | **3.41 bits** | 3.41 bits |

- **Routing now nearly reaches the current topology's ceiling.** The gap closes from 6.2 to 1.4
  points. Lucy's own probabilities are calibrated without fitting (own surprise = calibrated
  3.41 bits, versus unigram 4.68, KT bigram 2.76).
- **The remaining limit is the wiring.** The edge-restricted probe stays at 37–39 %, against
  44.5 % for all cells. Each input cell reads ≈ 5.5 sources, and the next lever is which cells
  wire in: growth law A now has a readout able to use what it adds.
- The edge probe is ≈ 0.8 points *lower* under the competitive readout. The growth law reads
  `1 − p(true)`, which changes which edges survive.
- Getting there needed two fixes found in the first sweep:
  - The readout needs its own step (`eta_readout`); sharing `eta_q` with the hidden cells also
    changed the tissue.
  - The normalization needs a regularizer (`readout_eps` = 1). With 1e-3, own surprise was
    4.6–6.8 bits from overconfident logits.

### Wayback C (`505b1b6`), CHILDES 1 MiB, seeds 7 and 11

`replay_per_pass = 128`, `eta_w = 0.3`, 32 dreams per pass, against replay off (the same
competitive-readout engine). Means of the 10 checkpoints (200 k probe windows). Forgetting =
full probe on the first 20 k bytes, re-read through a frozen clone.

| seed | replay | tissue (full probe) | edge probe | routing | own surprise | forgetting probe: first → last checkpoint | saturation | reverb |
|---|---|---|---|---|---|---|---|---|
| 7 | off | 44.50 % | 37.8 % | 36.4 % | 3.41 bits | 44.9 → 41.7 % | 0.0 % | 34 % |
| 7 | on | **45.27 %** | 38.9 % | **29.6 %** | **5.82 bits** | 45.9 → 43.1 % | 3.0 % | 38 % |
| 11 | off | 45.15 % | 38.9 % | 36.6 % | 3.35 bits | 46.2 → 42.8 % | 0.0 % | 36 % |
| 11 | on | **45.78 %** | 39.8 % | **29.5 %** | **5.88 bits** | 46.8 → 44.1 % | 3.6 % | 43 % |

- **Tissue rises above 44.5 %.** The gain is +0.7 / +0.6 points, and replay is ahead at 19 of
  20 checkpoints across both seeds. The edge-restricted probe rises too (+1.1 / +0.9).
- **Forgetting stays flat relative to replay off.** The early-segment probe falls by ≈ 3 points
  over 1 MiB *with or without* replay, because Q, gains and thresholds drift as well. With
  replay it is higher at every checkpoint and falls slightly less (−2.8 vs −3.2, −2.7 vs −3.4).
- **Routing breaks (stability risk 3, moving target).** Readout accuracy falls 36.4 → 29.5 %
  and keeps falling (last 5 checkpoints 28 %). Lucy's own surprise rises to 5.8 bits, worse
  than unigram: confident logits on hidden states that W has moved since Q learned them. Seen
  at 200 k too: `eta_w` 0.03 keeps routing (35 %) but gives no tissue gain.
- **Health.** Fires per tick stay at budget, and branching stays at 0.94–0.95. Saturation goes
  from 0 to 3–4 % of state components and reverb rises 4–7 points. With B on as well (200 k), one
  reservoir cell briefly ran away (fired every tick for ~100 ticks, log-theta +21) and
  homeostasis absorbed it within one window.
- **Before turning it on:** the readout must track W. Options are a slower `eta_w` with more
  segments, or re-fitting the readout from replay. Either way, C stays off by default.

### Growth law A with the competitive readout, and C with readout consolidation (`7b9fc45`), CHILDES 1 MiB

Means over 10 checkpoints (`scripts/compare.py`); bigram on the same test bytes 41.2 %.

| run | routing s7 / s11 | edge-restricted s7 / s11 | tissue s7 / s11 | own surprise | forgetting probe (s11) |
|---|---|---|---|---|---|
| old readout (`909c19a`) | 32.4 / 34.2 % | 38.6 / 38.7 % | 44.50 / 45.15 % | – | – |
| competitive, cap 32 | 36.4 / 36.6 % | 37.8 / 38.9 % | 44.50 / 45.15 % | 3.41 / 3.35 | 46.2 → 42.8 % |
| competitive, cap 128 | **38.5 / 35.8 %** | **39.9 / 40.1 %** | 44.50 / 45.15 % | 3.42 / 3.43 | (same tissue) |
| + C, no readout consolidation | 29.6 / 29.5 % | 38.9 / 39.8 % | 45.27 / 45.78 % | 5.82 / 5.88 | 46.8 → 44.1 % |
| + C with readout consolidation | 30.2 / 31.2 % | 38.4 / 39.8 % | **45.29 / 46.60 %** | 4.30 / 4.57 | 47.7 → 44.7 % |

- **Step 2 is confirmed on seed 11:** routing +2.4 points over the old readout (seed 7: +4.0).
- **Cap 128** lets input cells keep ≈ 20 in-edges (vs 5.5). The edge-restricted ceiling rises
  by 1.2–2.1 points on both seeds. Routing rises on seed 7 (+2.1) and dips on seed 11 (−0.8).
  Nothing clears the bigram yet.
- **Readout consolidation keeps C's tissue gain but does not save routing.**
  - The tissue gain holds: +0.8 / +1.5 points.
  - Routing is +0.6 / +1.7 points better than without consolidation, but still 6 points below
    no replay. Own surprise improves 5.8 → 4.3–4.6 bits (no replay: 3.4).
  - The edge-restricted probe under C is as high as without replay, so the information is in
    the wiring. The readout no longer reads it (8-point gap, against 1.4).
  - Likely causes: (1) consolidation trains Q only on the most surprising segments (forward
    replay top-1 21.6 %), a skewed sample; (2) the tissue change itself, with reverb up 4–7
    points and hot inputs per byte 12.4 → 10.6, gives noisier input traces.
- Saturation stays at 2.6–3.5 % (alarm: 10 %).

### Is the bigram path a timing problem? (`070721c`, `0a0b1fc`), CHILDES 1 MiB, cap 128

Path diagnostics, means of the last 5 checkpoints (T_in = 8; T_in = 16 is the same within 0.01):

| | seed 7 | seed 11 |
|---|---|---|
| time-resolved edge probe, k = 1 … 8 ticks before the byte | 25, 33, 35, 36, 36, 37, 38, 32 % | 26, 32, 35, 36, 38, 38, 39, 29 % |
| bigram-path probe (only the previous byte's direct targets) | 40.5 % | 42.2 % |
| bigram (same test bytes) | 41.2 % | 41.2 % |
| share of the true cell's hidden in-edges on the bigram path | 3.0 % | 3.1 % |
| bigram-path fires per byte (of 16 direct targets) | 2.5 | 2.7 |

At 30 k bytes (cap 32, T_in = 4) the audit found bigram-path fires 6.9 ticks before the guess.
Their heat fell from 0.74 at arrival to 0.17 at the guess, so decay was real.

Input-cell trace timescale `t_heat_input`, means of 10 checkpoints:

| T_in | routing s7 / s11 | edge-restricted s7 / s11 | own surprise s7 / s11 |
|---|---|---|---|
| 4 (= t_heat) | 38.5 / 35.8 % | 39.9 / 40.1 % | 3.42 / 3.43 |
| 8 (= K) | 38.1 / 36.0 % | 39.8 / 40.1 % | 3.30 / 3.56 |
| 16 (= 2K) | 36.5 / 35.9 % | 39.8 / 40.1 % | 3.58 / 3.63 |

- **The representation is fine.** The previous byte's direct targets alone carry the bigram:
  40.5–42.2 %, about the bigram itself.
- **Decay is real, but fixing it does not help.** A longer input trace keeps more heat to the
  guess (all edges: 0.43 → 0.54 of 0.72 arrival heat per byte at T_in 8 → 16). Routing and the
  edge-restricted ceiling do not move. `t_heat_input` stays at 4.
- **The limit is the wiring.** Only 3 % of the true cell's in-edges come from the previous byte's
  direct targets, and about 2.5 of those 16 cells fire per byte. Growth ranks candidates by
  *excess* co-occurrence `N − f_h n_b / W`. That favours often-firing cells with a small lift
  over the sparse, byte-specific bigram-path cells. The edge-restricted ceiling (≈ 40 %) is just
  under the bigram.

### Step 3 combinations (`d0301ab`, readout consolidation on surprise-ranked segments), cap 128

| run | routing s7 / s11 | edge s7 / s11 | tissue s7 / s11 | own surprise | forgetting s7 | reservoir theta |
|---|---|---|---|---|---|---|
| cap 128, no replay | 38.5 / 35.8 % | 39.9 / 40.1 % | 44.50 / 45.15 % | 3.42 / 3.43 | 44.9 → 41.7 % | 0.07 / 0.10 |
| + C | 32.4 / 31.5 % | 40.6 / 40.8 % | 45.23 / 45.70 % | 3.82 / 4.09 | 46.3 → 44.1 % | 0.10 / 0.17 |
| + C + B | 32.1 / 29.7 % | 41.5 / 41.2 % | 45.39 / 45.46 % | 4.14 / 4.43 | 45.2 → 44.7 % | **runaway** |

- C still lifts tissue (+0.7 / +0.6) and the edge-restricted ceiling, but routing drops 6 points.
- **C + B is unstable locally.** Reservoir cells run away: a reservoir-mean theta above 100 in
  362 of 801 windows (seed 7) and 58 (seed 11), overflowing to infinity in 5 windows. Global
  activity still holds at 20.2 fires per tick (max 398–611), and saturation stays at 2.2–2.5 %.
  B's sequence-copying edges plus W updates build tight self-exciting loops that homeostasis
  can only contain.

### D: long-horizon run on the 68 MB corpus (`c81d1e3`: old regression readout, growth law A, no B/C), seed 7

`childes-dialogue1.txt` (68.8 MB). Checkpoints every 2 MB: probe on the last 200 k windows, metrics,
cell snapshot. The run was **stopped at 12.76 MB** (machine shutdown) after ≈ 11 h. Most of that
time it shared 8 physical cores with up to 16 sweep processes (see Memory behaviour), so it ran
well below its 2.6 M events/s alone. No save/resume exists yet, so it cannot be continued.

| checkpoint | routing | tissue (probe) | control | bigram / trigram | probe bits | readout (2 MB windows) | fires/tick (max) | reverb | theta (reservoir, min–max) | branching | in-edges / input cell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 MB | 31.9 % | 44.60 % | 41.2 % | 41.4 / 57.1 % | 2.798 | 32.3 % | 20.54 (446) | 31.7 % | 0.040–0.490 | 0.948 | 5.39 |
| 4 MB | 32.1 % | 44.93 % | 41.7 % | 41.7 / 57.4 % | 2.792 | 32.0 % | 20.58 (708) | 35.0 % | 0.040–0.108 | 0.946 | 5.70 |
| 6 MB | 32.0 % | 44.11 % | 40.9 % | 40.9 / 57.1 % | 2.828 | 32.1 % | 20.59 (430) | 35.2 % | 0.036–0.129 | 0.946 | 5.71 |
| 8 MB | 32.5 % | 45.16 % | 41.8 % | 41.9 / 57.6 % | 2.785 | 32.1 % | 20.59 (489) | 35.3 % | 0.032–0.111 | 0.946 | 5.76 |
| 10 MB | 32.2 % | 44.90 % | 40.9 % | 41.2 / 57.1 % | 2.803 | 32.3 % | 20.59 (500) | 35.3 % | 0.038–0.100 | 0.946 | 5.80 |
| 12 MB | 32.3 % | 44.53 % | 41.2 % | 41.3 / 57.3 % | 2.787 | 32.3 % | 20.59 (421) | 35.3 % | 0.036–0.115 | 0.946 | 5.78 |

- **Answer to "improve, plateau or destabilize?": flat and stable.** By 2 MB the run sits on the
  same plateau as at 1 MiB (routing 32 %, tissue 44.5 %) and stays there to 12 MB:
  - routing 31.9–32.5 %, tissue 44.1–45.2 %;
  - checkpoint differences follow the test segment's difficulty (the control and the bigram
    move with them).
- **No slow drift or instability over 12 MB (≈ 100 M ticks):**
  - fires per tick are held at 20.6 (budget 20.5) and branching at 0.946, subcritical;
  - reverb settles at 35 % (alarm 60 %);
  - reservoir theta stays at 0.03–0.13 after the first 2 MB, low but not falling;
  - single-tick maxima of 420–710 fires are transient bursts, not trends;
  - in-degree of input cells creeps 5.4 → 5.8.
- **Without W learning and with the old readout, more data buys nothing:** the tissue is a fixed
  random reservoir, and the routing plateau is the readout's and the wiring's.

### v2: context cells, CHILDES 1 MB (`childes-1mb.txt`), prequential, last 100 k window

Online uni/bi/trigram on the same window: 21.0 / 41.3 / 57.3 %. Lucy Phase 1 best: 38.5 % / 3.42 bits.

| step | commit | top-1 | bits/byte | cells | ns/byte | notes |
|---|---|---|---|---|---|---|
| reference (Python) | `5640ea1` | 70.5 % | 1.517 | 152 039 | ≈ 36 000 | `context_cells_reference.py` |
| 1a exact port | `a3b0897` | 70.5 % | 1.517 | 152 039 | — | identical at every checkpoint |
| 1b native, no decay | `277995d` | 70.5 % | 1.504 | 150 743 | — | sorted lists, lowest-byte ties |
| 1b native, default (t_v 1e6, budget 2 M) | `277995d` | 70.3 % | 1.516 | 117 278 | ≈ 8 000–16 000 | still spikes into a 32-cell reservoir, which dominates the time |
| 2 reservoir off the byte path | `7dc2a99` | 70.3 % | 1.516 | 117 278 | ≈ 5 300–10 900 (whole run); 1 650 (tissue alone) | 1 MB in 8.2 s; the observatory's calibrator (2.4 µs) and 256 logits (1 µs) cost more than the tissue |
| fix: growth gate on visits | `4303646` | 70.5 % | 1.505 | 143 701 | | decayed totals made any decay need a 3rd visit before growth (−0.2 to −0.4 points) |
| 3 structural sleep (prune + fold, fold off; t_v 1e7) | `3af9ea1` | 70.6 % | 1.504 | 150 365 | ≈ 2 700–3 250 (whole run, solo) | fold never wins (see Sleep); 1 MB in 3.1 s solo |
| 4 chunk cells (off by default; K 8 + tension chunks) | `f6a1e4a` | 70.7 % | 1.512 | 261 184 | | +2.2 points at K 4; tension boundaries = spaces |
| 5 class cells (off by default; similarity, weight 1) | (5) | 70.7 % | 1.477 | 150 365 + 4 096 classes | | random control 1.572 bits; offline bound 70.8 % / 1.457 |

**68 MB (`childes-dialogue1.txt`), context tissue alone** (scratch bench calling `Context::arrive` and
`sleep` every 1 000 bytes; three runs at once, so ns/byte is noisy; growth gate fixed):

| `ctx_t_v` | 4 MB | 16 MB | 32 MB | 68 MB | bits at 68 MB | live cells at 68 MB | pruned | ns/byte |
|---|---|---|---|---|---|---|---|---|
| 1e6 | 70.9 % | 71.9 % | 71.9 % | 71.9 % | 1.361 | 257 k (flat from 16 MB) | 4.5 M | 1 300–2 550 |
| **1e7 (default)** | 71.1 % | 73.2 % | 73.4 % | 73.5 % | 1.262 | 960 k (near equilibrium) | 1.25 M | 1 400–2 070 |
| off | 71.1 % | 73.3 % | 73.7 % | 74.1 % | 1.233 | 1.8 M (still growing) | 0 | 1 250–2 110 |

- Accuracy keeps rising with data where the reservoir's D run stayed flat at 32 %.
- Decay sets an equilibrium cell count; 1e6 is too short (−2.2 points at 68 MB).
- Per byte the work is O(K), but wall time per byte rises ≈ 1.5× from 0.4 M to 1.8 M cells:
  the trie leaves the caches.

Decay timescale at 1 MB, before the growth-gate fix (no budget): `ctx_t_v` 3e5 → 69.9 % / 1.534
bits, 81 649 cells; 1e5 → 68.6 % / 1.611, 42 478 cells. A budget of 65 536 cells costs 0.1–0.4 points at 1 MB
(70.2 % with decay, 70.1 % without).

## Next (queued)

v2 order (`docs/PROPOSAL-v2.md`): steps 0–4 are done (see Context cells and the v2 table).
The Phase 1 reservoir items are paused while the reservoir is off the byte path: bigram-path
wiring, C's routing collapse, f16/bf16 W/Q.

### Step 5: class cells from successor similarity (built: `cls_mode`, off by default)

**Target.** The weakest byte class is word-initial bytes. Current tissue (default decay):

| | word-initial | inside a word | non-alphanumeric |
|---|---|---|---|
| CHILDES 1 MB | 29.2 % | 72.7 % | 86.0 % |
| 68 MB, last MiB | 32.4 % | 79.3 % | 88.5 % |

Which word comes next is where memory runs out and generalization has to start.

**Prior art (honest).** Grouping contexts by the distribution of what follows them is
distributional clustering. Brown et al. 1992 built class-based n-grams from successor mutual
information; Pereira, Tishby & Lee 1993 clustered distributions. Gains are known to be largest
for rare contexts and on sparse data, and smaller on repetitive data like CHILDES. What would be
new is doing it online, in sleep, inside a tension-grown context graph.

**Design.**
0. *Instrument first (cheap, its own commit).* The observatory reports top-1 and bits by byte
   class (word-initial, inside, non-alphanumeric). It also reports accuracy split by the
   deepest cell's visits (< 3, 3–15, ≥ 16). These are the acceptance numbers for this step.
1. *Which cells get a class.* Cells with ≥ `cls_min_visits` (16) undecayed visits. Rare deep
   contexts are not classified directly. Their word-aligned suffix cell is: for "…the zebra␣"
   that is "zebra␣", which is in the same chain and has seen every occurrence of the word.
2. *Sketch and matching, in the sleep sweep only, O(1) per byte.*
   - A cell's sketch is its successor distribution truncated to the top 8 bytes and
     renormalized.
   - Candidate classes come from an LSH bucket keyed by the cell's sorted top-2 successors. It
     is a sorted map (deterministic, never iterated by hash).
   - The cell joins the nearest candidate class within Hellinger distance `cls_radius` (0.3).
     Otherwise it founds a new class while under `cls_budget` (4096).
   - A cell whose sketch drifts beyond the radius leaves on a later visit. A class with no
     members dies.
3. *Class evidence is pooled online, not in sleep.* At each arrival, the deepest classed cell
   in the chain has its class count x: one extra update per byte. Sleep only changes membership
   and never adds evidence, so negative result 1 still holds.
4. *Vote: back off to the class before the shorter context.* The class votes as an extra
   chain member at the depth of the cell it stands in for, with weight
   `cls_weight · T_c/(T_c+2) · 2^(d−1) / T_c`. It ranks below that cell and above its parent;
   `cls_weight` is swept over {0.25, 0.5, 1}.

**Acceptance.**
- Word-initial top-1 above today's 29.2 % (1 MB) and 32.4 % (68 MB, last MiB).
- Higher accuracy where the deepest cell has < 3 visits.
- Overall top-1 and bits not worse.
- Control: random class assignment with the same class sizes. The gain must come from
  similarity, not from extra voters.

**Built as designed** (`context/classes.rs`), with the vote at `2^(d−1) · cls_weight` (weight 1;
it trades top-1 for bits: weight 4 gives −0.053 bits at −0.2 points top-1 at 1 MB). Controls:
- `cls_mode = 2`, the random control: the same pipeline, but a fixed hash of the cell id picks
  one of 4096 classes.
- `lucy class-bound`, the offline upper bound. Pass 1 runs without classes. Its final eligible
  contexts are clustered in hindsight by Brown-style exchange, which maximizes
  `Σ N(class, x) ln N(class, x)/N(class)`, into 4096 classes. Pass 2 reruns with that fixed map
  through the same back-off path (`cls_mode = 3`). The cell structure is the same in every run
  (growth reads the deepest cell, not the vote), so only the class assignment differs.
- Exchange time: 1 MB clusters 25 614 contexts in 8.8 s; 68 MB clusters 516 161 in 195 s
  (rayon, deterministic).

**Results**, whole run (byte-weighted over all windows). "Rare" = the deepest chain cell had been
seen < 3 times: 15.3 % of bytes at 1 MB, 2.7 % at 68 MB.

| | 1 MB top-1 | bits | word-initial | rare | 68 MB top-1 | bits | word-initial | rare |
|---|---|---|---|---|---|---|---|---|
| baseline | 68.97 % | 1.632 | 29.18 % | 47.09 % | 73.25 % | 1.295 | 32.06 % | 53.36 % |
| random control | 68.99 % | 1.670 | 29.18 % | 47.29 % | 73.26 % | 1.432 | 32.06 % | 53.43 % |
| **online classes** | 69.04 % | 1.602 | 29.34 % | 47.39 % | 73.29 % | 1.278 | 32.15 % | 53.15 % |
| offline bound | 69.10 % | 1.589 | 29.37 % | 47.52 % | 73.39 % | 1.266 | 32.25 % | 53.46 % |

Share of the bound's gain over baseline that online classes capture:

| | 1 MB | 68 MB |
|---|---|---|
| bits | 70 % | 61 % |
| word-initial | 84 % | 47 % |
| top-1 | 54 % | 29 % |
| rare contexts | 70 % | negative (−0.21 points, against the bound's +0.10) |

Last 1 MB window at 68 MB: baseline 73.49 % / 1.262 / word-initial 32.21 %, online 73.60 % / 1.248 / 32.27 %,
bound 73.87 % / 1.221 / 32.69 %.

- **Similarity is what helps.** The random control gets no word-initial gain and makes bits
  worse (+0.04 at 1 MB, +0.14 at 68 MB), so the online gains are not just an extra voter.
- **The online classes capture most of the attainable gain at 1 MB and about half at 68 MB.**
- **But the attainable gain itself is small.** Even hindsight Brown-style classes through this
  back-off move word-initial accuracy by +0.2 points and bits by −0.03 to −0.04. Byte-level
  successor classes are not the lever for "which word comes next".
- The next lever is the fallback below: sketch the successor *chunk* (the next word), not the
  next byte.
- Classes stay off by default. The 68 MB wall times were taken while four runs and the
  exchange step shared the machine, so the cost of classes is not measured.

**Risks noted at design time** (both borne out):
- On CHILDES the gain may be small, because the data is repetitive and memorization already
  covers a lot.
- Byte-level successors (the next letter) may be too coarse a signal of word similarity. If so,
  the fallback is to sketch the successor *chunk* (the next word's first 2–3 bytes, read from
  the cell's descendants in the forward direction). That needs chunk tracking (step 4) and costs
  more.

## Performance

Profile (callgrind; `perf` needs `kernel.perf_event_paranoid ≤ 2`), CHILDES, before the work:
heap scheduler 24 % (moving 168-byte events), `dot` 19 % (SSE, 32 separate horizontal sums per
matvec), `expf` 6 % (≈ 5 per touch), `tanhf` 6 % (inherent), memcpy 10 %. Each step below kept the
state hash identical; single-thread events/s on CHILDES 200 k, 400 k ticks:

| Commit | Change | events/s |
|---|---|---|
| (before) | BinaryHeap, 168-byte events | 2.07 M |
| `1885484` | time-wheel buckets + per-tick state arena (messages ≈ 40 B) | 2.37 M |
| `76d7712` | decay factors from bit-identical tables, one exp each | 2.52 M |
| `c143d31` | `target-cpu=x86-64-v3` (AVX2; no FMA contraction, fixed sums) | 2.55 M |
| `7d5c6cb` | sort 8-byte delivery keys instead of messages | 2.66 M |
| `8e33bcb` | Fire phase as one batch per tick (+ opt-in parallel path) | 3.0 M |

After that (with plasticity on), per-phase instruction share: Fire 42 %, Deliver 25 %, Arrive
(input readout) 19 %, scheduler 8 %, consolidation 4 %. Instruction count fell 42 % but wall time
only ≈ 30 %: what remains is mostly memory traffic.

**Multi-core (`8e33bcb`).** Tick-synchronous: a tick's fires are computed in parallel (phase A,
pure, from the state at the start of the phase), then applied and spiked in cell order (phase B),
so messages read out-edge gains exactly as the sequential path. Opt-in (`--threads N`); the fused
sequential `fire()` stays the reference and a test plus `check-determinism --threads N`
(1 thread vs N) assert bit-identity. Gains are small because each tick has little work and a
barrier: 4 096 cells (≈ 20 fires/tick) 2.88 M → 2.85 M events/s at 8 threads (none); 32 768 cells
(≈ 164 fires/tick) 1.88 M → 2.07 M (+10 %). Amdahl ceiling with only the Fire phase parallel is
≈ 1.6×; the next steps would be parallel delivery by destination cell (deliveries to different
cells are independent) and a persistent spin-barrier pool instead of rayon fork-join per tick.

**Runs in parallel.** `lucy sweep` runs seeds × parameter grids as separate processes; this is
near-linear in cores and is the right tool for experiments.

**Womb corpus estimate.** ≈ 220 events per tick at 4 096 cells → ≈ 13.6 k ticks/s single-thread.
68 MB × K = 8 → 544 M ticks ≈ 11 h. K is a lever too: the measured hop depth is ≈ 5.4 of 8 ticks.

### Memory behaviour (perf, Ryzen 7 250: 8 cores / 16 threads, L2 1 MB/core, L3 16 MB shared)

`perf stat`, 100 k bytes of CHILDES, 4096 cells, competitive readout (`7b9fc45`):

| | IPC | L1D load miss | L1 misses hitting L2 | demand fills from L2 / L3 / DRAM | events/s |
|---|---|---|---|---|---|
| alone (other runs SIGSTOPped) | **0.99** | 6.8 % | 52 % | 47 % / 38 % / 15 % | 2.6 M |
| 17 processes on the machine | **0.11** | 7.1 % | 38 % | 37 % / 8 % / 55 % | 0.3 M |

- **Where the misses come from** (`perf record` on DRAM demand fills, alone): `num::dot` 39 %,
  the W·h and Q·h products of a fire; `step` (fire, inlined) 17 %; `spike` 13 % (out-edge loop
  over the edge SoA); `touch` 6 %; `learn_in_gains` 5 %; scheduler `pop` 4 %. L3 fills are led
  by `touch` (25 %) and `learn_in_gains` (12 %): per-cell fields spread over ≈ 10 columns, plus
  per-cell heap `credit` lists.
- **The dominant traffic is intrinsic.** Every fire reads its own W and Q, 2 × 32 × 32 f32 =
  8 KB. At 4096 cells that is 35 MB of matrices, more than the 16 MB L3. Renumbering cells for
  neighbour locality cannot shrink it; it only helps the small per-cell and edge fields, which at
  4096 cells already fit in L2 + L3.
- **Software prefetch of W and Q** is issued when a fire is scheduled during Deliver, so the rest
  of the Deliver phase hides the latency. Hash identical. Measured alone, interleaved ×2:
  −1 to −3 % cycles, −14 to −16 % DRAM demand fills, IPC 0.95–1.00 → 0.99–1.02.
- **Sweeps should run ≤ 8 jobs.** With 17 processes each run fell to 0.3 M events/s (IPC 0.11,
  L3 useless): about 5 M events/s in aggregate, less than 8 jobs gave earlier (≈ 13 M). With
  SMT and one shared L3, more jobs than physical cores lose throughput.
- **Not done.** Renumbering (BFS order from the input cells, with a stable logical id for the
  hash) and grouping hot per-cell scalars into one cache line together target the ≈ 30 % of
  misses outside the matrices; I expect ≤ 10 % at 4096 cells, more at ≥ 32 k cells. Halving the
  matrix traffic with f16/bf16 W and Q storage would change the dynamics (hash), so that needs
  your decision.

### GPU (proposal only, not built)

A single Lucy does not fit a GPU: ticks are sequential, a tick has ≈ 220 tiny irregular events,
and 3 kernel launches per tick (≈ 5–10 µs each) alone cap it near today's CPU tick rate. The case
where a GPU wins is **many independent Lucys in lockstep** (sweeps): batch B instances so each
tick processes B × ≈ 20 fires (each ≈ 3 D×D matvecs + 32 tanh) and B × ≈ 200 scattered messages per
launch. Memory bounds B: per instance ≈ 4 352 cells × (W + Q) × 4 KB ≈ 36 MB fp32, so ≈ 200 instances
fit the RTX 5060's 8 GB. At ≈ 30 µs per batched tick that is ≈ 6.6 M instance-ticks/s, against
≈ 16 cores × 13.6 k ≈ 220 k instance-ticks/s for `lucy sweep`: an estimated **≈ 20–30×** for sweeps,
none for a single gestation. Caveat: equivalence against the reference engine cannot be bit-exact
while the CPU uses glibc `expf`/`tanhf` and the GPU its own intrinsics. Bit-exactness would need
both to use one portable implementation of exp/tanh in the rules module (a one-time change of
reference results); otherwise the equivalence test must be statistical.

## Open findings

1. **W is never updated: hidden cells are a fixed random reservoir.** This violates the "nothing
   frozen" principle and is the main target of Phase 2 (W learns from the one-hop feedback during
   reverse replay). No change in Phase 1.
2. **Routing (resolved in part by plasticity).** Without plasticity, input-cell hotness carried no
   information (true cell hot 37.5 % vs 39.5 % for a random cell, first 200 k bytes). Co-activation
   growth lifts it to 69 % at 15 hot cells per byte, and accuracy to 33.7 %; the probe says 44.5 % is
   linearly available from the same firing, and the trigram reaches 57 %.
3. **Plasticity design choices to review.** Growth ranks by *excess* co-occurrence (not raw
   count), and edges into input cells earn vitality from reality instead of from use; both are
   explained in the plasticity section. The accuracy plateau (≈ 33 %, in-degree ≈ 6) is untuned.
4. **Surprise calibration (resolved).** Linear `softmax(score / tau)` was capped at ≈ 6.7 bits;
   log-domain calibration gives 3.83 bits (< unigram 4.68).
5. **Watch items (not fixed).** Theta (reservoir mean of `theta_eff`) falls from 0.42 to 0.041 by
   ≈ 300 k bytes and sits at ≈ 0.06 to 0.07 afterwards, also with plasticity. The rate stays at
   target, so fast adaptation does most of the gating. Reverb fraction plateaus at ≈ 32–35 %,
   below the 60 % alarm.
