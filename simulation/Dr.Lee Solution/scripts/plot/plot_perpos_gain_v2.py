#!/usr/bin/env python3
"""Regenerate motivation per-position figures from the PROPER budget-aware
measurement (perpos2_{ds}.json). Shared realized bars (pooled, deployed log+iso
trajectory, test split) + expected head / expected total lines. raw + 6 calib
combos. Same total-line format as before.
"""
from pathlib import Path
import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/workspace/simulation/Dr.Lee Solution/readable_outputs/figures/motivation")
files = sorted(glob.glob("/workspace/tmp/perpos2_*.json"))
agg = None
for f in files:
    d = json.load(open(f))
    if agg is None:
        agg = {k: (np.array(v) if k in ("rh", "rt", "rn", "raw") else v) for k, v in d.items()}
        agg["raw"] = np.array(d["raw"]); agg["combos"] = {c: np.array(v) for c, v in d["combos"].items()}
        continue
    for k in ("rh", "rt", "rn"):
        agg[k] = agg[k] + np.array(d[k])
    agg["raw"] = agg["raw"] + np.array(d["raw"])
    for c, v in d["combos"].items():
        agg["combos"][c] = agg["combos"][c] + np.array(v)
n = agg["rn"]; n = np.where(n == 0, 1, n)
rh = agg["rh"] / n; rt = agg["rt"] / n
ks = list(range(len(rh)))
C_HEAD, C_TAIL, L_HEAD, L_TAIL = "#9ecae1", "#fdd0a2", "#08519c", "#d94801"


def draw(gh, gt, title, fname):
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.bar(ks, rh, color=C_HEAD, edgecolor="k", linewidth=0.4, zorder=2,
           label="realized head accept  E[min(k, a_head)]")
    ax.bar(ks, rt, bottom=rh, color=C_TAIL, edgecolor="k", linewidth=0.4, zorder=2,
           label="realized tail accept  (grafted at k)")
    ax.plot(ks, gh, color=L_HEAD, lw=2.4, marker="o", ms=4, zorder=4,
            label="expected head gain  G_k")
    ax.plot(ks, gh + gt, color=L_TAIL, lw=2.4, marker="s", ms=4, zorder=4,
            label="expected total gain  G_k + S_k·T_k  (gap = expected tail)")
    ax.set_xlabel("handoff position k  (head length; k=0 = pure suffix)", fontsize=11)
    ax.set_ylabel("accepted tokens", fontsize=11)
    ax.set_xticks(ks)
    ax.set_ylim(0, max(3.6, float((rh + rt).max()) * 1.15,
                       float((gh + gt).max()) * 1.05))
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper right" if "RAW" in title else "lower right",
              frameon=True)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=150); plt.close(fig)
    print("saved ->", fname)


NOTE = "pooled test split · budget-aware calibrators (calib-split fit)"
gr = agg["raw"] / n
draw(gr[0], gr[1],
     "RAW per-position gains (no calibration)\nexpected total ≫ realized "
     "(tail 4-5× over) → mis-routes to low k\n" + NOTE, "perpos_gain_raw.png")
for c, v in agg["combos"].items():
    h, t = c.split("+")
    g = v / n
    draw(g[0], g[1],
         f"Calibrated per-position gains — {h} head + {t} tail\n"
         "expected tracks realized across all k (incl. deep)\n" + NOTE,
         f"perpos_gain_calib_{h}_{t}.png")
