#!/usr/bin/env python3
"""Linear probe: does recent reservoir firing carry next-byte context? (offline, numpy only)

Input: files written by `cargo run --release --example probe_dump -- CORPUS SEED START N PREFIX`.
For each byte arrival: the hidden cells that fired in the K ticks before it, and the true byte.

Temporal split: first 80 % train, last 20 % test. Reports test top-1 and cross-entropy for
  - count baselines (unigram / bigram / trigram with back-off), fitted on train only;
  - a control probe on a one-hot of the previous byte (must land near the bigram: proves the probe
    trainer can extract context when it is there);
  - the reservoir probe (softmax regression, minibatch SGD, L2-normalized binary rows).

usage: python3 scripts/probe.py PREFIX [PREFIX ...]   (writes probe_report.json)
"""
import json
import sys

import numpy as np

CLASSES = 256


def load(prefix):
    indptr = np.fromfile(prefix + ".indptr", dtype="<u4").astype(np.int64)
    indices = np.fromfile(prefix + ".indices", dtype="<u2").astype(np.int64)
    labels = np.fromfile(prefix + ".labels", dtype=np.uint8).astype(np.int64)
    with open(prefix + ".meta.json") as f:
        meta = json.load(f)
    first = meta["first_hidden"]
    return indptr, indices - first, labels, meta["cells"] - first, meta


def count_baselines(labels, cut):
    """Order-0/1/2 count models fitted on labels[:cut], evaluated on labels[cut:] (with back-off)."""
    tables = [dict() for _ in range(3)]
    for i in range(2, cut):
        for k in range(3):
            row = tables[k].setdefault(tuple(labels[i - k : i]), np.zeros(CLASSES))
            row[labels[i]] += 1
    out = {}
    for k in range(3):
        correct, bits = 0, 0.0
        for i in range(cut, len(labels)):
            row = None
            for kk in range(k, -1, -1):  # back off to shorter contexts when unseen
                row = tables[kk].get(tuple(labels[i - kk : i]))
                if row is not None:
                    break
            correct += int(np.argmax(row) == labels[i])
            bits -= np.log2((row[labels[i]] + 0.5) / (row.sum() + 0.5 * CLASSES))
        n = len(labels) - cut
        out[("unigram", "bigram", "trigram")[k]] = (correct / n, float(bits) / n)
    return out


def gather(indptr, indices, rows):
    """Flattened features of `rows`: (feature idx, row position, value = 1/sqrt(nnz), nnz)."""
    starts, ends = indptr[rows], indptr[rows + 1]
    lens = ends - starts
    pos = np.repeat(np.arange(len(rows)), lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    feat = indices[np.repeat(starts, lens) + offs]
    val = np.repeat(1.0 / np.sqrt(np.maximum(lens, 1)), lens).astype(np.float32)
    return feat, pos, val, lens


def logits_of(W, b, indptr, indices, rows):
    feat, pos, val, lens = gather(indptr, indices, rows)
    z = np.zeros((len(rows), CLASSES), dtype=np.float32)
    nz = lens > 0
    if nz.any():
        offsets = (np.cumsum(lens) - lens)[nz]
        z[nz] = np.add.reduceat(W[feat] * val[:, None], offsets, axis=0)
    return z + b, feat, pos, val


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def evaluate(W, b, indptr, indices, labels, rows):
    correct, bits = 0, 0.0
    for s in range(0, len(rows), 4096):
        r = rows[s : s + 4096]
        p = softmax(logits_of(W, b, indptr, indices, r)[0])
        correct += int((p.argmax(axis=1) == labels[r]).sum())
        bits -= np.log2(np.maximum(p[np.arange(len(r)), labels[r]], 1e-12)).sum()
    return correct / len(rows), float(bits) / len(rows)


def train_probe(indptr, indices, labels, n_feat, train, test, epochs=6, lr=0.2, batch=256, seed=0):
    """Plain SGD on softmax cross-entropy, per-sample steps applied in minibatches (weight gradient
    summed over the batch: rows are L2-normalized, so one sample moves its logits by ~lr; the
    bias takes the batch mean). Returns test (top-1, bits) after each epoch."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_feat, CLASSES), dtype=np.float32)
    b = np.zeros(CLASSES, dtype=np.float32)
    history = []
    for ep in range(epochs):
        step = lr / (1 + ep)
        perm = rng.permutation(train)
        for s in range(0, len(perm), batch):
            rows = perm[s : s + batch]
            z, feat, pos, val = logits_of(W, b, indptr, indices, rows)
            g = softmax(z)
            g[np.arange(len(rows)), labels[rows]] -= 1.0
            if len(feat):
                order = np.argsort(feat, kind="stable")
                uniq, first = np.unique(feat[order], return_index=True)
                W[uniq] -= step * np.add.reduceat((g[pos] * val[:, None])[order], first, axis=0)
            b -= step * g.mean(axis=0)
        history.append(evaluate(W, b, indptr, indices, labels, test))
    return history


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    report = []
    for prefix in sys.argv[1:]:
        indptr, indices, labels, n_feat, meta = load(prefix)
        n = len(labels)
        cut = int(n * 0.8)
        rows = np.arange(2, n)
        train, test = rows[rows < cut], rows[rows >= cut]
        base = count_baselines(labels, cut)
        c_indptr = np.arange(n + 1, dtype=np.int64)  # control: one-hot previous byte
        c_indices = np.concatenate([[0], labels[:-1]])
        control = train_probe(c_indptr, c_indices, labels, CLASSES, train, test)
        probe = train_probe(indptr, indices, labels, n_feat, train, test)
        active = (indptr[1:] - indptr[:-1]).mean()
        print(f"\n== {prefix}  (start byte {meta['start_byte']}, n={n}, {active:.1f} active cells/sample)")
        for name, (acc, bits) in base.items():
            print(f"  {name:28s} top-1 {acc:7.2%}  {bits:6.3f} bits")
        print(f"  {'control probe (prev byte)':28s} top-1 {control[-1][0]:7.2%}  {control[-1][1]:6.3f} bits")
        print(f"  {'reservoir probe':28s} top-1 {probe[-1][0]:7.2%}  {probe[-1][1]:6.3f} bits")
        print("  reservoir probe by epoch:", "  ".join(f"{a:.2%}/{b:.2f}" for a, b in probe))
        report.append(
            {"prefix": prefix, "baselines": base, "control": control[-1], "probe": probe[-1], "probe_history": probe}
        )
    with open("probe_report.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
