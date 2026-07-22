#!/usr/bin/env python3
"""Per-position gain under online calibration — one figure per head×tail combo.
Shared realized head/tail bars (calibration-independent) with the CALIBRATED
expected gain lines overlaid, so you can see the lines now track the bars
(compare to the raw figure where expected tail sat 4-5x above realized).

  python3 scripts/plot/plot_perpos_gain_calib.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/workspace/simulation/Dr.Lee Solution/readable_outputs/figures/motivation")
D = json.load(open("/workspace/tmp/perpos_gain_calib.json"))
R = D["realized"]
ks = sorted(int(k) for k in R)
rh = np.array([R[str(k)]["rh"] for k in ks])
rt = np.array([R[str(k)]["rt"] for k in ks])

C_HEAD, C_TAIL = "#9ecae1", "#fdd0a2"
L_HEAD, L_TAIL = "#08519c", "#d94801"

for combo, cg in D["combos"].items():
    gh = np.array([cg[str(k)]["gh"] for k in ks])
    gt = np.array([cg[str(k)]["gt"] for k in ks])
    h, t = combo.split("+")
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.bar(ks, rh, color=C_HEAD, edgecolor="k", linewidth=0.4, zorder=2,
           label="realized head accept  E[min(k, a_head)]")
    ax.bar(ks, rt, bottom=rh, color=C_TAIL, edgecolor="k", linewidth=0.4, zorder=2,
           label="realized tail accept  (grafted at k)")
    ax.plot(ks, gh, color=L_HEAD, lw=2.4, marker="o", ms=4, zorder=4,
            label=f"expected head gain  ({h} head, calibrated)")
    ax.plot(ks, gh + gt, color=L_TAIL, lw=2.4, marker="s", ms=4, zorder=4,
            label=f"expected total gain  (calibrated; gap = expected tail, {t})")
    ax.set_xlabel("handoff position k  (head length; k=0 = pure suffix)", fontsize=11)
    ax.set_ylabel("accepted tokens", fontsize=11)
    ax.set_xticks(ks)
    ax.set_ylim(0, max(3.5, (rh + rt).max() * 1.15))
    ax.set_title(f"Online-calibrated per-position gains — {h} head + {t} tail (window 8k)\n"
                 "pooled over all workloads · calibrated expected lines now track "
                 "the realized bars",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"perpos_gain_calib_{h}_{t}.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    print("saved ->", fp)
