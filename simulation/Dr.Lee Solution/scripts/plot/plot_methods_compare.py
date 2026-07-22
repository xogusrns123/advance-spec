#!/usr/bin/env python3
"""Methodology comparison in the CALIB_effect format (deployable split, test
half): four compose arms per workload with the SD-paper hybrid drawn as the
purple bar-to-beat line. No percentage-improvement annotations.

  raw          compose on raw DFlash conf + raw arctic score (no correction)
  calib        online windowed calibration: logistic head + isotonic tail
  smoothing    raw head + succession-rescored tail (Laplace (k+1)/(n+2))
  weight       fixed weights: head x1.4, tail x0.125

Reads the *_split replay logs + the deployable fallback sweep. Emits
readable_outputs/figures/mat(deployable)/METHODS_compare.png

  python3 scripts/plot/plot_methods_compare.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/workspace/simulation/Dr.Lee Solution")
RLOG = BASE / "readable_outputs" / "figures" / "replay_logs"
SWEEP = Path("/workspace/simulation/results/pipeline_deployable/segments")
OUT = BASE / "readable_outputs" / "figures" / "mat(deployable)"

WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench\nVerified",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
# (log tag, label, color) — distinct qualitative colors
ARMS = [("calib_raw_split", "raw (no correction)", "#bdbdbd"),
        ("calib_onl8000_split", "calib (online, window 8k)", "#4C78A8"),
        ("calib_succrawhead_split", "smoothing", "#54A24B"),
        ("calib_h14tw125_split", "weight (head ×1.4, tail ×0.125)", "#E45756")]
C_HYB = "#9467BD"


def compose_k(ds, tag):
    txt = (RLOG / f"mat_{ds}_4way_{tag}.replay.txt").read_text()
    return float(re.search(r"calib: K=([0-9.]+)", txt).group(1))


def hybrid_k(ds):
    d = next(iter(json.load(open(SWEEP / f"fallback_sweep_fresh_{ds}.json")).values()))
    if set(d) == {"calib", "test"}:            # deployable split sweep
        best = max(d["calib"], key=lambda t: d["calib"][t]["K"])
        return d["test"][best]["K"]
    return max(v["K"] for v in d.values())


K = {ds: {tag: compose_k(ds, tag) for tag, _, _ in ARMS} for ds in WLS}
hyb = {ds: hybrid_k(ds) for ds in WLS}
order = WLS                                    # fixed order (no lift sorting)

x = np.arange(len(order))
na = len(ARMS)
w = 0.18
half = (na - 1) / 2.0
fig, ax = plt.subplots(figsize=(14.5, 6.8))
for i, (tag, lab, col) in enumerate(ARMS):
    vals = [K[d][tag] for d in order]
    b = ax.bar(x + (i - half) * w, vals, w, label=lab, color=col,
               edgecolor="k", linewidth=0.5, zorder=3)
    for bar in b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold")

for i, d in enumerate(order):
    ax.hlines(hyb[d], i - (half + 0.5) * w, i + (half + 0.5) * w, color=C_HYB,
              lw=2.4, zorder=4)
    ax.text(i + (half + 0.5) * w + 0.03, hyb[d], f"{hyb[d]:.2f}", color=C_HYB,
            fontsize=8.5, va="center", fontweight="bold")

ax.hlines([], [], [], color=C_HYB, lw=2.4, label="SD-paper hybrid (per-step switch, best τ)")
ax.set_xticks(x)
ax.set_xticklabels([WL_NAME[d] for d in order], fontsize=11)
ax.set_ylabel("compose MAT  (mean accepted tokens / verify step)", fontsize=11)
ax.set_title("Workload-agnostic score correction — three methodologies vs the "
             "SD-paper hybrid\n(deployable split: fit on the calib half, MAT on the "
             "disjoint test half; purple line = SD-paper hybrid baseline)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9.5, loc="upper right", frameon=True, ncol=1)
ax.grid(axis="y", alpha=0.2, zorder=0)
ax.set_ylim(0, max(max(max(K[d].values()) for d in WLS), max(hyb.values())) * 1.18)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fp = OUT / "METHODS_compare.png"
fig.savefig(fp, dpi=150, bbox_inches="tight")
print("saved ->", fp)
print(f"{'wl':<10}{'raw':>7}{'calib':>8}{'smooth':>8}{'weight':>8}{'hybrid':>8}")
for d in order:
    print(f"{d:<10}" + "".join(f"{K[d][t]:>8.2f}" for t, _, _ in ARMS)
          + f"{hyb[d]:>8.2f}")
