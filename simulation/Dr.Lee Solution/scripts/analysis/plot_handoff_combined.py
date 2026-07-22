#!/usr/bin/env python3
"""Cross-workload summary of handoff-position selection regret.

Reads results/bias_variance/handoff_{wl}.json (analyze_handoff_selection.py) for
all 5 workloads and emits to readable_outputs/figures/0716_regenerated/:
  handoff_position_ratio_all.png   stacked early/optimal/late share per workload
  handoff_matloss_decomp_all.png   stacked early/late MAT loss (= oracle gap) per wl

  python3 scripts/analysis/plot_handoff_combined.py
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
C_EARLY, C_OPT, C_LATE = "#d1603d", "#4C9F70", "#3E6E9E"
CALIB = sys.argv[1] if len(sys.argv) > 1 else "raw"
CLAB = "raw prob" if CALIB == "raw" else "online calib"

data = {}
for wl in WLS:
    fp = BV / f"handoff_{wl}_{CALIB}.json"
    if fp.exists():
        data[wl] = json.load(open(fp))
wls = [w for w in WLS if w in data]
x = np.arange(len(wls))
names = [WL_NAME[w] for w in wls]

# ---- ratio (stacked %) ----
early = np.array([data[w]["frac"]["early"] * 100 for w in wls])
opt = np.array([data[w]["frac"]["optimal"] * 100 for w in wls])
late = np.array([data[w]["frac"]["late"] * 100 for w in wls])
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.bar(x, early, 0.6, color=C_EARLY, label="early handoff")
ax.bar(x, opt, 0.6, bottom=early, color=C_OPT, label="optimal")
ax.bar(x, late, 0.6, bottom=early + opt, color=C_LATE, label="late handoff")
for i in range(len(wls)):
    ax.text(x[i], early[i] / 2, f"{early[i]:.1f}", ha="center", va="center", fontsize=9, color="white")
    ax.text(x[i], early[i] + opt[i] / 2, f"{opt[i]:.1f}", ha="center", va="center", fontsize=9, color="white")
    ax.text(x[i], early[i] + opt[i] + late[i] / 2, f"{late[i]:.1f}", ha="center", va="center", fontsize=9, color="white")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("share of handoff decisions (%)"); ax.set_ylim(0, 100)
ax.set_title(f"Handoff-position selection vs decision oracle ({CLAB} signals, all workloads)", fontweight="bold")
ax.legend(loc="lower right", ncol=3, framealpha=0.95); ax.grid(axis="y", alpha=0.3, ls=":")
fig.tight_layout(); fig.savefig(OUT / f"handoff_position_ratio_all_{CALIB}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved ->", OUT / f"handoff_position_ratio_all_{CALIB}.png")

# ---- MAT loss (stacked, = oracle gap) ----
el = np.array([data[w]["early_loss"] for w in wls])
ll = np.array([data[w]["late_loss"] for w in wls])
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.bar(x, el, 0.6, color=C_EARLY, label="early handoff loss")
ax.bar(x, ll, 0.6, bottom=el, color=C_LATE, label="late handoff loss")
for i in range(len(wls)):
    if el[i] > 0.02:
        ax.text(x[i], el[i] / 2, f"{el[i]:.2f}", ha="center", va="center", fontsize=9, color="white")
    if ll[i] > 0.02:
        ax.text(x[i], el[i] + ll[i] / 2, f"{ll[i]:.2f}", ha="center", va="center", fontsize=9, color="white")
    ax.text(x[i], el[i] + ll[i] + 0.01, f"gap {el[i] + ll[i]:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("MAT loss vs decision oracle (accepted tokens/round)")
ax.set_ylim(0, max(el + ll) * 1.18)
ax.set_title(f"Handoff MAT-loss decomposition: early vs late ({CLAB} signals, all workloads)", fontweight="bold")
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3, ls=":")
fig.tight_layout(); fig.savefig(OUT / f"handoff_matloss_decomp_all_{CALIB}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved ->", OUT / f"handoff_matloss_decomp_all_{CALIB}.png")

# ---- text table ----
print("\nwl".ljust(12) + "early%  opt%   late%   e-loss  l-loss  gap    oracle  compose")
for w in wls:
    dd = data[w]
    print(f"{w:<11}{dd['frac']['early']*100:>6.1f}{dd['frac']['optimal']*100:>7.1f}"
          f"{dd['frac']['late']*100:>7.1f}{dd['early_loss']:>8.3f}{dd['late_loss']:>8.3f}"
          f"{dd['early_loss']+dd['late_loss']:>7.3f}{dd['oracle_mat']:>8.2f}{dd['compose_mat']:>8.2f}")
