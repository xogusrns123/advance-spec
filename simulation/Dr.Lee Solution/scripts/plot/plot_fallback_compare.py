#!/usr/bin/env python3
"""Supplementary comparison: SuffixDecoding-paper hybrid (round-level score-
threshold fallback, model side = DFlash) vs our calibrated compose, singles and
oracle. 5-bar MAT per workload; fallback at its BEST swept tau (generous to the
baseline), with the full swept tau grid noted in the corner.

Inputs: replay_logs/mat_{ds}_4way_{grp}{sfx}.replay.txt + fallback_sweep json.
  python3 scripts/plot_fallback_compare.py --log-suffix _pre \
      --sweep /workspace/simulation/results/pipeline_4way/fallback_sweep_pre.json \
      --note "..."
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_mat_4way as p4

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "readable_outputs" / "figures" / "mat"

PROPS = ["dflash", "suffix", "fallback", "calib", "oracle"]
LABELS = {"dflash": "DFlash (single)", "suffix": "Suffix (single)",
          "fallback": "SD-paper hybrid\n(fallback, best τ)",
          "calib": "Compose (calibrated)", "oracle": "Oracle (best handoff)"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "fallback": "#9467BD",
          "calib": "#54A24B", "oracle": "#E45756"}
DS_LABEL = {"specbench": "Spec-Bench", "bfcl": "BFCL v4",
            "swebench": "SWE-bench", "spider": "Spider2-DBT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-suffix", default="")
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--outname", default="MAT_fallback_compare.png")
    args = ap.parse_args()

    p4.LOG_SUFFIX = args.log_suffix
    K, _ = p4.load()                       # {ds: {prop: (K, rounds)}}
    sweep = json.load(open(args.sweep))
    best = {}
    for ds, taus in sweep.items():
        t, v = max(taus.items(), key=lambda kv: kv[1]["K"])
        best[ds] = (float(t), v["K"], v["suffix_share"])

    wls = [ds for ds in ["specbench", "bfcl", "swebench", "spider"]
           if K.get(ds) and ds in best]
    fig, ax = plt.subplots(figsize=(2.6 + 2.5 * len(wls), 5.6))
    n, g = len(PROPS), 0.80
    bw = g / n
    ymax = 0.0
    for pi, p in enumerate(PROPS):
        xs = [ti - g / 2 + bw * (pi + 0.5) for ti in range(len(wls))]
        if p == "fallback":
            ys = [best[ds][1] for ds in wls]
        else:
            ys = [K[ds].get(p, (0, 0))[0] for ds in wls]
        ymax = max(ymax, max(ys))
        ax.bar(xs, ys, width=bw * 0.9, color=COLORS[p], label=LABELS[p])
        for x, y in zip(xs, ys):
            if y > 0:
                ax.text(x, y + 0.04, f"{y:.2f}", ha="center", va="bottom",
                        fontsize=10.5, color="#444444")
    for ti, ds in enumerate(wls):
        x = ti - g / 2 + bw * (PROPS.index("fallback") + 0.5)
        ax.text(x, best[ds][1] + ymax * 0.065, f"τ*={best[ds][0]:g}", ha="center",
                fontsize=11.5, fontweight="bold", color="#9467BD")
    ax.set_ylim(0, ymax * 1.20)
    ax.set_xticks(range(len(wls)))
    ax.set_xticklabels([DS_LABEL[d] for d in wls], fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("mean accept length  (tokens)", fontsize=14)
    ax.legend(fontsize=11.5, frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("SD-paper hybrid (score<τ → DFlash fallback) vs compose"
                 + (f"   [{args.note}]" if args.note else ""), fontsize=15)
    grid = sorted({float(t) for taus in sweep.values() for t in taus})
    ax.text(0.99, 0.99,
            "τ swept: {" + ", ".join(f"{t:g}" for t in grid) + "}\n"
            "bar = best τ per workload",
            transform=ax.transAxes, ha="right", va="top", fontsize=11.5,
            color="#555555")

    fig.tight_layout()
    fp = OUT / args.outname
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp}")
    for ds in wls:
        b = best[ds]
        print(f"  [{ds}] fallback τ*={b[0]:g} K={b[1]:.2f} (suffix {b[2]:.0%}) | "
              + "  ".join(f"{p}={K[ds].get(p, (0, 0))[0]:.2f}"
                          for p in ("dflash", "suffix", "calib", "oracle")))


if __name__ == "__main__":
    main()
