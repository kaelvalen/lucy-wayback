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
e      = u - Q h_prev                                   // tension
Q     += eta_q / (nlms_eps + |h_prev|^2) * e h_prev^T   // normalized LMS (fix 3)
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

- `surprise_bits`, `accuracy`: per byte in the window.
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

## Open findings (Phase 1 run, need a decision)

1. **The input readout cannot learn as specified.** Guess/surprise read `heat = Σ|gain|·|h|`,
   but the rule `gain += eta_g·dot(code − u, m)/D` trains the vector `u` toward `code`. That moves
   |gain| with the sign of `dot(code, h)`, which is random. Reading the trained quantity instead
   (`dot(u_c, code_c)`) does not help either: scalar gains on hidden states that are random relative
   to the codes cannot steer `u_c` toward `code_c`. Measured (first 50 k bytes): heat, alignment,
   with/without input scaling are all at chance (accuracy ≈ 0.2 %). Giving each input cell a linear
   readout using its own Q (prediction `Q_c u_c`, NLMS toward `code_b` or 0, the hidden-cell rule
   reused) lifts accuracy to ≈ 10 %.
2. **σ-normalised softmax surprise is miscalibrated.** Most of the 256 cells are cold, so σ is
   small relative to the hot ones, the softmax is overconfident, and misses cost far more than
   8 bits. Even the 10 %-accuracy readout scores ≈ 13.5 bits. Proposed: calibrate the temperature
   online in the observatory (one scalar, fitted by log-loss, never fed back into the dynamics),
   or mix with a uniform floor.
3. **Prediction lag.** The spec computes `e = u − Q·h_prev` *before* `h_prev = h`, so Q predicts
   input from the state two fires back. Implemented as written; switching to the current `h` is a
   one-line change in `kernel::fire`.
