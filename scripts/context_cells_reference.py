"""Reference prototype (Python, prequential). Tension-grown context cells: a cell = a context (suffix of the history).

- SPREAD: at each byte the active chain is every existing cell whose context is a suffix of the
  recent history (orders 0..K). Only those cells are touched.
- TRACE: each cell keeps decayed counts of what followed it (earned vitality of its out-edges);
  the prediction is a confidence-weighted vote of the active chain.
- TENSION/GROWTH: when the deepest active cell mispredicts and has enough evidence, it grows a
  child one byte deeper (the clone that disambiguates its context). Cells grow only where wrong.
- DECAY: counts decay lazily; cells that stop being visited are pruned (not needed at 1 MB).
Prequential top-1 vs online uni/bi/trigram counts on the same stream.
"""
import sys, collections, time
path, N = sys.argv[1], int(sys.argv[2])
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
GROW_MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
data = open(path, 'rb').read()[:N]

cells = {b'': collections.Counter()}     # context -> next-byte counts
tot = {b'': 0.0}
uni = collections.Counter(); bi = collections.defaultdict(collections.Counter)
tri = collections.defaultdict(collections.Counter)
hits = collections.Counter(); n = 0; t0 = time.time()
bits = collections.Counter()
for i, x in enumerate(data):
    hist = data[max(0, i - K):i]
    chain = [b'']
    for k in range(1, len(hist) + 1):
        c = hist[-k:]
        if c in cells: chain.append(c)
        else: break
    # vote: deeper, more confident cells dominate (weight = total/(total+2) * 2^order)
    score = collections.defaultdict(float)
    for c in chain:
        T = tot[c]
        if T <= 0: continue
        w = (T / (T + 2.0)) * (2.0 ** len(c)) / T
        for b, v in cells[c].items():
            score[b] += w * v
    guess = max(score, key=score.get) if score else 32
    Z = sum(score.values()); eps = 1e-3
    p = ((score.get(x, 0.0) / Z) if Z else 0.0) * (1 - eps) + eps / 256
    import math; bits['ctx'] += -math.log2(p)
    # baselines
    g_uni = uni.most_common(1)[0][0] if uni else 32
    p1 = data[i-1] if i else None; p2 = data[i-2:i] if i >= 2 else None
    g_bi = bi[p1].most_common(1)[0][0] if p1 is not None and bi[p1] else g_uni
    g_tri = tri[p2].most_common(1)[0][0] if p2 is not None and tri[p2] else g_bi
    n += 1; hits['ctx'] += guess == x; hits['uni'] += g_uni == x
    hits['bi'] += g_bi == x; hits['tri'] += g_tri == x
    # learn: every active cell earns; deepest mispredicting cell grows a child
    deep = chain[-1]
    deep_guess = cells[deep].most_common(1)[0][0] if cells[deep] else None
    for c in chain:
        cells[c][x] += 1.0; tot[c] += 1.0
    if deep_guess != x and len(deep) < len(hist) and tot[deep] >= GROW_MIN:
        child = hist[-(len(deep) + 1):]
        cells[child] = collections.Counter({x: 1.0}); tot[child] = 1.0
    uni[x] += 1
    if p1 is not None: bi[p1][x] += 1
    if p2 is not None: tri[p2][x] += 1
    if (i + 1) % 100_000 == 0:
        print(f"{i+1:>8}  ctx {hits['ctx']/n:.3f}  uni {hits['uni']/n:.3f}  bi {hits['bi']/n:.3f}  "
              f"tri {hits['tri']/n:.3f}  bits {bits['ctx']/n:.3f}  cells {len(cells)}  {time.time()-t0:.0f}s", flush=True)
        hits.clear(); bits.clear(); n = 0
