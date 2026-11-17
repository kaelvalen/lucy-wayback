#!/usr/bin/env python3
"""Plot a Lucy metrics.jsonl: learning (surprise, accuracy vs unigram baselines), activity, homeostasis.

usage: python3 scripts/plot.py runs/metrics.jsonl [out.png]
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    header, rows = {}, []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("kind") == "header":
                header = d
            elif d.get("kind") == "metrics":
                rows.append(d)
    return header, rows


def series(rows, *keys):
    xs, ys = [], []
    for r in rows:
        v = r
        for k in keys:
            v = v.get(k) if isinstance(v, dict) else None
        if v is not None:
            xs.append(r["bytes_seen"])
            ys.append(v)
    return xs, ys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path.rsplit(".", 1)[0] + ".png"
    header, rows = load(path)
    if not rows:
        sys.exit(f"no metrics rows in {path}")

    fig, ax = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    fig.suptitle(f"Lucy — {header.get('corpus', path)}")

    a = ax[0][0]
    a.plot(*series(rows, "surprise_bits"), label="surprise (bits/byte)")
    if "unigram_entropy_bits" in header:
        a.axhline(header["unigram_entropy_bits"], color="k", ls="--", label="unigram entropy")
    a.axhline(8.0, color="grey", ls=":", label="uniform (8 bits)")
    a.set_title("Surprise per byte")
    a.legend()

    a = ax[0][1]
    a.plot(*series(rows, "accuracy"), label="next-byte accuracy")
    if "unigram_top1_accuracy" in header:
        a.axhline(header["unigram_top1_accuracy"], color="k", ls="--", label="always most-frequent byte")
    a.axhline(1 / 256, color="grey", ls=":", label="chance")
    a.set_title("Guess accuracy")
    a.legend()

    a = ax[1][0]
    a.plot(*series(rows, "fires_per_tick"), label="mean fires/tick")
    a.plot(*series(rows, "max_fires_per_tick"), label="max fires in one tick", alpha=0.6)
    a.set_yscale("log")
    a.set_title("Activity (dead network / avalanche watch)")
    a.legend()

    a = ax[1][1]
    a.plot(*series(rows, "hop_depth"), label="hop depth per byte window")
    if "seed_config" in header:
        a.axhline(header["seed_config"]["ticks_per_byte"], color="k", ls="--", label="K (ticks per byte)")
    b = a.twinx()
    b.plot(*series(rows, "reverb_fraction"), color="C3", label="reverb fraction")
    b.set_ylim(0, 1)
    a.set_title("Wave reach and self-sustained activity")
    a.legend(loc="upper left")
    b.legend(loc="upper right")

    a = ax[2][0]
    a.plot(*series(rows, "tension"), label="tension: mean |e|^2 of hidden fires")
    a.plot(*series(rows, "reservoir", "tau", "p50"), label="tau p50 (reservoir)")
    a.plot(*series(rows, "reservoir", "tau", "p90"), label="tau p90 (reservoir)")
    a.set_title("Tension")
    a.legend()

    a = ax[2][1]
    a.plot(*series(rows, "reservoir", "theta_mean"), label="theta (reservoir mean)")
    a.plot(*series(rows, "reservoir", "v_mean"), label="vitality (reservoir mean)")
    a.set_yscale("log")  # vitality (~alpha_v * rate * t_v) dwarfs theta otherwise
    b = a.twinx()
    b.plot(*series(rows, "reservoir", "rate_mean"), color="C3", label="rate (reservoir mean)")
    if "params" in header:
        b.axhline(header["params"]["target_rate"], color="C3", ls="--", label="target rate")
    a.set_title("Homeostasis")
    a.legend(loc="upper left")
    b.legend(loc="upper right")

    for row in ax:
        for a in row:
            a.set_xlabel("bytes seen")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
