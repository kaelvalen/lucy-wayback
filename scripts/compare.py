#!/usr/bin/env python3
"""Compare runs on the headline metrics, means over their probe checkpoints.

usage: python3 scripts/compare.py run1.jsonl [run2.jsonl ...]
Columns: routing (Lucy readout on probe test bytes), edge-restricted probe, full probe (tissue),
forgetting probe (first 20 k bytes through a frozen clone: first -> last), Lucy's own surprise
and health over the last 20 % of windows.
"""
import json
import statistics as st
import sys


def rows(path):
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def main():
    print(f"{'run':52} {'n':>2} {'routing':>7} {'edge':>6} {'tissue':>6} {'bigram':>6} {'old first->last':>15} {'bits':>5} {'sat':>5} {'reverb':>6} {'theta':>6} {'in/inp':>6}")
    for path in sys.argv[1:]:
        rs = list(rows(path))
        pr = [r for r in rs if r.get("kind") == "probe" and r["bytes_seen"] % 100000 == 0]
        m = [r for r in rs if r.get("kind") == "metrics" and r.get("accuracy") is not None]
        late = m[int(len(m) * 0.8):]
        mean = lambda xs: st.mean(xs) if xs else float("nan")
        old = [p["old_probe_accuracy"] for p in pr if p.get("old_probe_accuracy") is not None]
        g = lambda k: mean([r[k] for r in late if r.get(k) is not None])
        print(f"{path[-52:]:52} {len(pr):>2} {mean([p['readout_accuracy'] for p in pr]):7.4f} "
              f"{mean([p['edge_probe_accuracy'] for p in pr if p.get('edge_probe_accuracy')]):6.4f} "
              f"{mean([p['probe_accuracy'] for p in pr]):6.4f} {mean([p['baseline_accuracy'][1] for p in pr]):6.4f} "
              f"{(f'{old[0]:.3f}->{old[-1]:.3f}' if old else '-'):>15} {g('readout_bits'):5.2f} {g('saturation'):5.3f} "
              f"{g('reverb_fraction'):6.3f} {st.median(r['reservoir']['theta_mean'] for r in late):6.3f} {late[-1]['input_in_degree']:6.2f}")


if __name__ == "__main__":
    main()
