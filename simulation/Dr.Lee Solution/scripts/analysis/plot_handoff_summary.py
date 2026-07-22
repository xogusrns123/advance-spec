#!/usr/bin/env python3
"""Per-workload single-bar summaries of handoff-selection quality.

Reads results/bias_variance/handoff_{wl}_{calib}.json (analyze_handoff_selection.py)
and emits to readable_outputs/figures/0716_regenerated/:
  handoff_missrate_by_workload_{calib}.png  miss rate (= early+late = 1-optimal), %
  handoff_matloss_by_workload_{calib}.png   total MAT loss (= early+late loss = gap)

  python3 scripts/analysis/plot_handoff_summary.py [raw|online]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/workspace/simulation/Dr.Lee Solution")
BV = BASE / "results" / "bias_variance"
OUT = BASE / "readable_outputs" / "figures" / "0716_regenerated"
WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "tau2-bench"}
CALIB = sys.argv[1] if len(sys.argv) > 1 else "raw"
CLAB = "raw prob" if CALIB == "raw" else "online calib"
C_MISS, C_LOSS = "#D62728", "#8B0000"   # miss rate = red, MAT loss = dark red

data = {}
for wl in WLS:
    fp = BV / f"handoff_{wl}_{CALIB}.json"
    if fp.exists():
        data[wl] = json.load(open(fp))
wls = [w for w in WLS if w in data]
x = np.arange(len(wls))
names = [WL_NAME[w] for w in wls]
OUT.mkdir(parents=True, exist_ok=True)


def bar_fig(vals, ylabel, title, fname, color, fmt, pad):
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    b = ax.bar(x, vals, color=color, edgecolor="k", linewidth=0.5, width=0.6)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01,
                fmt.format(v), ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel(ylabel); ax.set_ylim(0, max(vals) * pad)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved ->", OUT / fname)


# 1) miss rate = early + late = 1 - optimal
miss = [(data[w]["frac"]["early"] + data[w]["frac"]["late"]) * 100 for w in wls]
bar_fig(miss, "handoff miss rate (%)",
        f"Handoff selection miss rate per workload ({CLAB} signals)",
        f"handoff_missrate_by_workload_{CALIB}.png", C_MISS, "{:.1f}%", 1.15)

# 2) total MAT loss = early_loss + late_loss = oracle gap
loss = [data[w]["early_loss"] + data[w]["late_loss"] for w in wls]
bar_fig(loss, "MAT loss vs decision oracle (accepted tokens/round)",
        f"Handoff MAT loss per workload ({CLAB} signals)",
        f"handoff_matloss_by_workload_{CALIB}.png", C_LOSS, "{:.2f}", 1.15)

print("\nwl".ljust(12) + "miss%   MATloss")
for w in wls:
    dd = data[w]
    print(f"{w:<11}{(dd['frac']['early']+dd['frac']['late'])*100:>6.1f}"
          f"{dd['early_loss']+dd['late_loss']:>9.3f}")
