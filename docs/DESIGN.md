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

1. Readout. Every input cell `c` scores "my byte comes next":
   `pred_c = Q_c u_c`, `score_c = prior_c + dot(pred_c, code_c) / D` (1 = mine, 0 = not mine).
   A cold cell has `u = 0`, so it scores exactly its prior. Guess = argmax score (ties → lowest
   byte); surprise from the scores (see Observatory).
2. Reality teaches every touched cell (hot cells and `b`): target `code_b` for `b`, `0` for hot
   wrong cells; `e = target - pred`;
   - `Q_c += eta_q / (nlms_eps + |u_c|^2) * e u_c^T` (normalized LMS, the hidden-cell rule);
   - in-gains: `toward = Q_c^T e` (the readout error mapped into the cell's input space), then
     `gain_i += eta_g * dot(toward, m_i) / D` and synaptic scaling;
   - `prior_c += eta_bias * (1[c = b] - score_c)`; `tau = ema(tau, |e|^2)`; clear the window.
   Cold wrong cells need no work: `Q u = 0` already, and their prior update (target 0, score =
   prior) is `prior *= 1 - eta_bias` per byte, applied lazily in closed form on next use.
   The priors therefore converge to byte frequencies (tested).
3. Force `b` to fire: `h_prev = h; h = code_b`; fire bookkeeping; send to out-edges at `t + 1`.

Decision 4 (amended): input cells have Q (their readout), still no W. Priors live in `Senses`.

## Structural plasticity (`plasticity.rs`)

First slice: co-activation edge growth into input cells. Every `consolidate_every` bytes the
engine replays its fire ring since the last pass (offline, sleep-like):

1. **Replay.** A byte window = hidden cells that fired between two byte arrivals (input-cell fires
   in the ring). Count `N(h, b)` (h fired in the window before byte b), `f_h`, `n_b`, windows `W`.
2. **Grow.** For each byte cell, up to `grow_per_cell` unconnected hidden cells with the largest
   *excess* co-occurrence `N(h,b) − f_h·n_b/W > 0` get an edge `h → b` (gain `grow_gain`,
   vitality = `N(h,b)`, what it already earned). Excess, not raw count: raw counts would route
   every new edge to the most frequent byte (space).
3. **Prune** each input cell's in-edges: decayed vitality < `prune_vitality` dies; over
   `input_in_cap`, lowest vitality dies first (ties → lower edge index).

**Vitality of edges into input cells is earned, not used:** it is credited (`+alpha_v`, lazily
decayed with `t_v`) when the edge carried a message into the cell whose byte then really arrived,
and not on transmit. With homeostatic firing rates, usage vitality is ≈ equal for every edge
and a new edge (vitality ≈ 1) would always lose to established ones (≈ 500); earned vitality is a
decayed co-occurrence count, the same statistic growth ranks by. Edges into hidden cells keep
usage vitality. No dynamics read vitality, so with consolidation off accuracy is unchanged.

Cost per pass ∝ fires since the last pass + the 256 input cells' in-edges. Ordered maps and index
tie-breaks keep it deterministic (tested).

## Constants (`Params`, overridable with `--params file.json`)

| Constant | Default | Rationale |
|---|---|---|
| `t_heat` | 4 ticks | Coincidence/trace window. Half of K, so a message landing early in a byte window still counts (e^-2) but the previous byte's prediction is mostly gone by the next. `u`, `heat`, `credit` share it to keep `u == Σ credit`. |
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
| `grow_per_cell` | 2 | New hidden→input edges per byte cell per pass: slow enough that pruning can keep up. |
| `grow_gain` | 0.1 | New edges start small; synaptic scaling then folds them into the cell's fixed gain budget. |
| `prune_vitality` | 0.5 | An in-edge of an input cell must keep earning: a single co-occurrence decays below 0.5 after ≈ 69 k ticks (≈ 8.7 k bytes). |
| `input_in_cap` | 32 | Homeostatic in-degree cap of input cells (2× the seed fan-in). Over it, the lowest-vitality edge dies. |
| `eta_bias` | 0.005 | Prior LMS step: averages byte frequency over ~200 bytes, so frequent bytes are tracked within a few utterances while one-off bytes barely move it. |

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

`snapshot` writes one JSON line per live cell (theta, rate, v, tau, heat, degrees, gain norms).

## Determinism

Seeded ChaCha8 for every random draw; no hash-map iteration anywhere; float summation orders fixed
(8-lane dot, event order, list order). `check-determinism` runs twice and compares the FNV-1a hash
of every metrics line (minus wall clock) plus the full final state (cells, edges, ring, pending
events). libm `exp`/`tanh` make runs bit-identical for the same binary on the same CPU
architecture, so `rust-toolchain.toml` pins the compiler exactly (1.99.1) and `flake.nix` provides it.

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
trigram 2.12). Accuracy reaches ≈ 32 % within 12.5 k bytes and plateaus; in-degree settles at ≈ 6,
far under the cap, so the vitality floor and growth rate (not the cap) limit it. Not tuned yet.

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
