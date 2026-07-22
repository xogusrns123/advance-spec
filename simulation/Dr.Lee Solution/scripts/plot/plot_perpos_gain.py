#!/usr/bin/env python3
"""RAW-composition mispricing diagnostic (motivation figure): per handoff
position k, stacked bars of REALIZED accept (head + tail) with the RAW EXPECTED
gains the compose controller actually optimizes overlaid as lines. The gap
between the red/blue lines and the bar segments is the mispricing that mis-routes
raw-prob composition. Pooled over all workloads.

  python3 scripts/plot/plot_perpos_gain.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/workspace/simulation/Dr.Lee Solution/readable_outputs/figures/motivation")
D = json.load(open("/workspace/tmp/perpos_gain_raw.json"))
ks = sorted(int(k) for k in D if D[k]["n"] > 0)
rh = np.array([D[str(k)]["rh"] for k in ks])
rt = np.array([D[str(k)]["rt"] for k in ks])
gh = np.array([D[str(k)]["gh"] for k in ks])
gt = np.array([D[str(k)]["gt"] for k in ks])

C_HEAD, C_TAIL = "#9ecae1", "#fdd0a2"          # realized bars (light)
L_HEAD, L_TAIL = "#08519c", "#d94801"          # expected lines (dark)

fig, ax = plt.subplots(figsize=(10.5, 6.4))
ax.bar(ks, rh, color=C_HEAD, edgecolor="k", linewidth=0.4, zorder=2,
       label="realized head accept  E[min(k, a_head)]")
ax.bar(ks, rt, bottom=rh, color=C_TAIL, edgecolor="k", linewidth=0.4, zorder=2,
       label="realized tail accept  (grafted at k)")
ax.plot(ks, gh, color=L_HEAD, lw=2.4, marker="o", ms=4, zorder=4,
        label="expected head gain  G_k   (RAW)")
ax.plot(ks, gh + gt, color=L_TAIL, lw=2.4, marker="s", ms=4, zorder=4,
        label="expected total gain  G_k + S_k·T_k   (RAW; gap = expected tail)")

ax.set_xlabel("handoff position k  (head length; k=0 = pure suffix)", fontsize=11)
ax.set_ylabel("accepted tokens", fontsize=11)
ax.set_xticks(ks)
ax.set_title("RAW-composition mispricing — realized accept (bars) vs the "
             "expected gains the controller optimizes (lines)\n"
             "pooled over all workloads · no calibration · "
             "controller picks argmax_k [1 + G_k + S_k·T_k]",
             fontsize=11.5, fontweight="bold")
ax.legend(fontsize=9.5, loc="upper right", frameon=True)
ax.grid(axis="y", alpha=0.25, zorder=0)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fp = OUT / "perpos_gain_raw.png"
fig.savefig(fp, dpi=150)
print("saved ->", fp)
print(f"{'k':>3}{'real_head':>11}{'real_tail':>11}{'exp_head':>11}{'exp_tail':>11}")
for i, k in enumerate(ks):
    print(f"{k:>3}{rh[i]:>11.2f}{rt[i]:>11.2f}{gh[i]:>11.2f}{gt[i]:>11.2f}")
