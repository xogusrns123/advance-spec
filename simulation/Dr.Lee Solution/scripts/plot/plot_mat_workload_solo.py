#!/usr/bin/env python3
"""ONE standalone image per workload: the 5-arm pooled MAT (NOT split by subtask).
Same arms/colors/format as MAT_per_workload_4way, but each workload on its own
figure -> MAT_workload_{ds}.png.

  python3 scripts/plot_mat_workload_solo.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_mat_4way as p4

OUT = p4.OUT
SEG = "/workspace/simulation/results/pipeline_4way/segments"
# harness + task subtitle (kept identical to the other MAT figures)
WL_INFO = {"specbench": "480 tasks (all) · 6 subtasks",
           "bfcl": "bfcl_eval (prompt-mode FC) · 753 tasks · 5 categories",
           "swebench": "mini-swe-agent · 19 of 500 tasks (self-terminated, 250 steps) · 7 repos",
           "spider": "spider-agent-dbt · 68 tasks (all) · 68 databases",
           "tau2": "tau2 official sim · 64 tasks · 3 domains"}
DS_LABEL = {"specbench": "Spec-Bench", "bfcl": "BFCL v4",
            "swebench": "SWE-bench Verified", "spider": "Spider2-DBT",
            "tau2": "τ²-bench"}


def main():
    p4.SWEEP_DIR = SEG
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-suffix", default="",
                    help="read mat_{ds}_4way_{grp}{sfx}.replay.txt (e.g. _raw)")
    ap.add_argument("--compose-suffix", default="",
                    help="override ONLY compose(calib) group, e.g. _raw (no calib)")
    ap.add_argument("--out-suffix", default="", help="appended to filename, e.g. _nocalib")
    ap.add_argument("--out-dir", default="",
                    help="override output dir (e.g. the mat(deployable) folder)")
    ap.add_argument("--sweep-dir", default="",
                    help="override dir with fallback_sweep_fresh_{ds}.json")
    args = ap.parse_args()
    p4.LOG_SUFFIX = args.log_suffix
    p4.COMPOSE_SUFFIX = args.compose_suffix
    if args.sweep_dir:
        p4.SWEEP_DIR = args.sweep_dir
    if args.compose_suffix == "_raw":
        p4.LABELS["calib"] = "Compose (raw, no calib)"
    OUT_SUFFIX = args.out_suffix
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    K, _ = p4.load()
    out_dir.mkdir(parents=True, exist_ok=True)
    corner = ""
    if p4.FB_GRID:
        corner = "τ swept: {" + ", ".join(f"{t:g}" for t in p4.FB_GRID) + "}\n" + \
                 ("τ* picked on calib half; bar = test half"
                  if p4.FB_SPLIT else "τ* on SD-paper hybrid bar = best")

    for ds in p4.DS:
        if not K.get(ds):
            continue
        arms = [a for a in p4.PROPS if a in K[ds] and K[ds][a][0] > 0]
        vals = [K[ds][a][0] for a in arms]
        ymax = max(vals)

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        xs = list(range(len(arms)))
        for x, a in zip(xs, arms):
            ax.bar(x, K[ds][a][0], width=0.72, color=p4.COLORS[a], label=p4.LABELS[a])
            ax.text(x, K[ds][a][0] + ymax * 0.012, f"{K[ds][a][0]:.2f}",
                    ha="center", va="bottom", fontsize=11, color="#444444")
        # τ* on the fallback bar
        if "fallback" in arms and ds in p4.FB_TAUS:
            xf = arms.index("fallback")
            ax.text(xf, K[ds]["fallback"][0] + ymax * 0.07, f"τ*={p4.FB_TAUS[ds]:g}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=p4.COLORS["fallback"])
        ax.set_ylim(0, ymax * 1.20)
        ax.set_xticks([])                       # arms shown via legend
        ax.set_ylabel("mean accept length  (tokens)", fontsize=12)
        ax.set_title(f"{DS_LABEL[ds]} — MAT\n{WL_INFO.get(ds, '')}", fontsize=13)
        ax.legend(fontsize=10, frameon=False, loc="upper left", ncol=1)
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if corner:
            ax.text(0.99, 0.99, corner, transform=ax.transAxes, ha="right",
                    va="top", fontsize=9, color="#666666")
        fig.tight_layout()
        fp = out_dir / f"MAT_workload_{ds}{OUT_SUFFIX}.png"
        fig.savefig(fp, dpi=150); plt.close(fig)
        print(f"saved {fp}  " + " ".join(f"{a}={K[ds][a][0]:.2f}" for a in arms))


if __name__ == "__main__":
    main()
