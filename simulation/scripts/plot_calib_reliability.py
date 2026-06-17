#!/usr/bin/env python3
"""Reliability diagrams for the chain-hybrid calibration maps.

Per model run dir: 3 panels (eagle | suffix raw c/n | suffix Jeffreys-shrunk).
  bars   = empirical accept rate per fixed-width prob bin, accumulated from
           the SAME train decision log the maps were fit on
  line   = the deployed frozen isotonic step map (what serving looks up)
  dashed = the PREVIOUS run's map for the same group (run-to-run drift)
  bottom = per-bin sample counts (log y) — sparse bins explain unstable fits

Usage:
    python3 simulation/scripts/plot_calib_reliability.py \
        --dir simulation/results/chain_hybrid_v2_14b \
        [--prev-dir simulation/results/chain_hybrid_tail] \
        [--decisions decisions_select1_train.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_chain_hybrid_calib import load_samples  # noqa: E402

BIN_W = 0.05


def binned(p: np.ndarray, y: np.ndarray):
    edges = np.arange(0.0, 1.0 + BIN_W, BIN_W)
    centers, rates, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        centers.append((lo + hi) / 2)
        counts.append(int(m.sum()))
        rates.append(float(y[m].mean()) if m.any() else np.nan)
    return np.asarray(centers), np.asarray(rates), np.asarray(counts)


def step_xy(map_path: Path, group: str):
    try:
        m = json.load(open(map_path))["groups"][group]
    except (FileNotFoundError, KeyError):
        return None
    xs = list(m["x"]) + [1.0]
    ys = list(m["y"]) + [m["y"][-1]]
    return np.asarray(xs), np.asarray(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prev-dir", default=None,
                    help="overlay this run's maps as dashed lines")
    ap.add_argument("--decisions", default="decisions_select1_train.jsonl")
    ap.add_argument("--tag", default=None, help="title tag (default: dir name)")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    tag = args.tag or out_dir.name
    s = load_samples(str(out_dir / args.decisions))

    ep = np.asarray([r[0] for r in s["eagle"]])
    ey = np.asarray([r[2] for r in s["eagle"]])
    sp = np.asarray([r[0] for r in s["suffix"]])
    sy = np.asarray([r[2] for r in s["suffix"]])
    have_counts = all(r[3] is not None and r[4] for r in s["suffix"])
    if have_counts:
        c = np.asarray([float(r[3]) for r in s["suffix"]])
        n = np.asarray([float(r[4]) for r in s["suffix"]])
        spj = (c + 0.5) / (n + 1.0)
    else:
        spj = sp.copy()

    panels = [
        ("eagle (draft softmax p)", ep, ey, "calib_noshrink.json", "eagle"),
        ("suffix raw c/n  +  no-shrink map", sp, sy,
         "calib_noshrink.json", "suffix"),
        ("suffix Jeffreys (c+.5)/(n+1)  +  Jeffreys map", spj, sy,
         "calib_jeffreys.json", "suffix"),
    ]

    fig, axes = plt.subplots(
        2, 3, figsize=(15, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    for col, (title, p, y, map_file, group) in enumerate(panels):
        ax, axc = axes[0][col], axes[1][col]
        centers, rates, counts = binned(p, y)
        ax.bar(centers, rates, width=BIN_W * 0.9, color="#9ecae1",
               edgecolor="#6baed6", label="empirical accept (bin)")
        cur = step_xy(out_dir / map_file, group)
        if cur:
            ax.step(cur[0], cur[1], where="post", color="crimson", lw=2,
                    label="deployed iso map (this run)")
        if args.prev_dir:
            prev = step_xy(Path(args.prev_dir) / map_file, group)
            if prev:
                ax.step(prev[0], prev[1], where="post", color="black",
                        lw=1.4, ls="--", label="previous run's map")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":", alpha=0.7)
        ax.set_title(f"{title}\n(n={len(p)}, base accept={y.mean():.3f})",
                     fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("P(accept)")
            ax.legend(fontsize=7, loc="upper left")
        axc.bar(centers, np.maximum(counts, 0.5), width=BIN_W * 0.9,
                color="#bdbdbd")
        axc.set_yscale("log")
        axc.grid(alpha=0.25, which="both")
        axc.set_xlabel("probability bin")
        if col == 0:
            axc.set_ylabel("n samples")
    fig.suptitle(f"Calibration reliability — {tag} "
                 f"(bars: train-log empirical; line: deployed map"
                 + ("; dashed: previous run" if args.prev_dir else "") + ")",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / "figures" / "calib_reliability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
