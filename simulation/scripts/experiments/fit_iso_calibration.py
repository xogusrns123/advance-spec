#!/usr/bin/env python3
"""Fit offline isotonic calibration maps from extension_isofit_* samples.

Input : one or more ``<prefix>.b<budget>.<pid>.part`` files written by the
        simulator's ``_CalibSampleCollector`` (env SIM_ISO_COLLECT_OUT).
        Each line: {"meta": {...}, "samples": {"eagle": [[p, depth, y], ...],
        "suffix": [...]}}. Edge probs are already Jeffreys count-shrunk for
        suffix when SIM_SUFFIX_SHRINK was active during collection (default).
Output: JSON consumed by ``_FrozenIsoCalibrator`` (env SIM_ISO_CALIB):
        {"meta": {...}, "groups": {grp: {"x": [bin left edges asc],
                                          "y": [isotonic accept rates]}}}

Method: per group, quantile-bin the (p, y) samples (--bins equal-count
bins), take each bin's mean accept rate, then run pool-adjacent-violators
over the bins (weights = bin counts) → a monotone non-decreasing step
function p → P(accept).

Usage:
    python3 simulation/scripts/experiments/fit_iso_calibration.py \
        --samples 'simulation/results/iso_calib/bfcl_v4_samples.jsonl.*.part' \
        --output simulation/results/iso_calib/bfcl_v4_iso_map.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys

import numpy as np


def pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: weighted isotonic (non-decreasing) fit."""
    n = len(y)
    # Blocks as (value, weight, count) stacks.
    vals: list[float] = []
    wts: list[float] = []
    cnts: list[int] = []
    for i in range(n):
        vals.append(float(y[i]))
        wts.append(float(w[i]))
        cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            wt = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wt if wt > 0 else 0.0)
            wts.append(wt)
            cnts.append(c1 + c2)
    out = np.empty(n)
    pos = 0
    for v, c in zip(vals, cnts):
        out[pos:pos + c] = v
        pos += c
    return out


def fit_group(p: np.ndarray, y: np.ndarray, n_bins: int) -> dict:
    order = np.argsort(p, kind="stable")
    p, y = p[order], y[order]
    n = len(p)
    n_bins = min(n_bins, max(1, n // 50))  # ≥50 samples per bin
    edges_idx = np.linspace(0, n, n_bins + 1).astype(int)
    bx, by, bw = [], [], []
    for i in range(n_bins):
        lo, hi = edges_idx[i], edges_idx[i + 1]
        if hi <= lo:
            continue
        bx.append(float(p[lo]))         # bin LEFT edge (lookup is right-1)
        by.append(float(y[lo:hi].mean()))
        bw.append(float(hi - lo))
    by_iso = pav(np.asarray(by), np.asarray(bw))
    # Merge adjacent equal-value steps for a compact lookup table.
    xs, ys = [0.0], [float(by_iso[0])]
    for i in range(1, len(bx)):
        if by_iso[i] != ys[-1]:
            xs.append(bx[i])
            ys.append(float(by_iso[i]))
    return {"x": xs, "y": ys,
            "n_samples": int(n), "n_bins": len(bx),
            "accept_rate": float(y.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True,
                    help="glob pattern of .part files from SIM_ISO_COLLECT_OUT")
    ap.add_argument("--output", required=True)
    ap.add_argument("--bins", type=int, default=512)
    ap.add_argument("--shrink", choices=["jeffreys", "none"], default="jeffreys",
                    help="MUST match the SIM_SUFFIX_SHRINK setting used during "
                         "collection; stored in meta so _FrozenIsoCalibrator "
                         "mirrors it at eval time")
    args = ap.parse_args()

    files = sorted(glob.glob(args.samples))
    if not files:
        sys.exit(f"no sample files match {args.samples!r}")

    data: dict[str, list] = {"eagle": [], "suffix": []}
    metas = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                blob = json.loads(line)
                metas.append(blob.get("meta", {}))
                for grp, rows in blob.get("samples", {}).items():
                    data.setdefault(grp, []).extend(rows)

    groups = {}
    for grp, rows in data.items():
        if not rows:
            print(f"  {grp}: 0 samples — skipped", file=sys.stderr)
            continue
        arr = np.asarray(rows, dtype=np.float64)
        p, y = arr[:, 0], arr[:, 2]
        groups[grp] = fit_group(p, y, args.bins)
        g = groups[grp]
        print(f"  {grp}: n={g['n_samples']} accept_rate={g['accept_rate']:.4f} "
              f"steps={len(g['x'])}", file=sys.stderr)
        # Reliability snapshot at a few raw probs.
        probe = [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]
        xs = np.asarray(g["x"])
        ys = np.asarray(g["y"])
        cal = [float(ys[max(0, int(np.searchsorted(xs, q, side='right')) - 1)])
               for q in probe]
        print("    p_raw→P(accept): "
              + ", ".join(f"{q:g}→{c:.3f}" for q, c in zip(probe, cal)),
              file=sys.stderr)

    if not groups:
        sys.exit("no samples in any group")

    out = {
        "meta": {
            "shrink": None if args.shrink == "none" else args.shrink,
            "source_files": files,
            "source_methods": sorted({m.get("method", "?") for m in metas}),
            "source_budgets": sorted({m.get("budget", 0) for m in metas}),
        },
        "groups": groups,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
