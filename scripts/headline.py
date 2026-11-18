#!/usr/bin/env python3
"""Print a run's headline series: readout accuracy, edges/cell, coverage, and probe checkpoints.

usage: python3 scripts/headline.py metrics.jsonl [every_nth_window]
"""
import json
import sys


def rows(path):
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:  # a run still writing: partial last line
                pass


def main():
    path = sys.argv[1]
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rs = list(rows(path))
    m = [r for r in rs if r.get("kind") == "metrics" and r.get("accuracy") is not None]
    print(f"{'bytes':>9} {'acc':>6} {'bigram':>6} {'in/inp':>6} {'in/hid':>6} {'grown':>6} {'pruned':>6} {'truthhot':>8} {'f/tick':>6} {'maxf':>5} {'reverb':>6} {'branch':>6}")
    for r in m[::step] + m[-1:]:
        print(f"{r['bytes_seen']:>9} {r['accuracy']:>6.3f} {r['baseline_accuracy'][1]:>6.3f} {r['input_in_degree']:>6.2f} "
              f"{r.get('hidden_in_degree', float('nan')):>6.2f} {r['edges_grown']:>6} {r['edges_pruned']:>6} "
              f"{r['truth_hot_fraction']:>8.3f} {r['fires_per_tick']:>6.1f} {r['max_fires_per_tick']:>5} "
              f"{r['reverb_fraction']:>6.3f} {r.get('branching_ratio') or float('nan'):>6.3f}")
    print(f"\n{'probe@':>9} {'routing':>8} {'edgeprb':>8} {'tissue':>8} {'control':>8} {'uni':>6} {'bi':>6} {'tri':>6} {'bits':>6} {'oldprb':>7} {'oldrd':>6}")
    for p in (r for r in rs if r.get("kind") == "probe"):
        b = p["baseline_accuracy"]
        print(f"{p['bytes_seen']:>9} {p.get('readout_accuracy', float('nan')):>8.4f} {p.get('edge_probe_accuracy') or float('nan'):>8.4f} {p['probe_accuracy']:>8.4f} "
              f"{p['control_accuracy']:>8.4f} {b[0]:>6.3f} {b[1]:>6.3f} {b[2]:>6.3f} {p['probe_bits']:>6.3f} {p.get('old_probe_accuracy') or float('nan'):>7.4f} {p.get('old_readout_accuracy') or float('nan'):>6.3f}")


if __name__ == "__main__":
    main()
