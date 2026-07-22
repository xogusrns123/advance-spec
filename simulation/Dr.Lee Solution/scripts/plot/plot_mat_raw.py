#!/usr/bin/env python3
"""MAT bar: baseline (Suffix-Decoding-paper hybrid) vs Ours (DFlash+Suffix compose),
RAW probabilities (NO calibration -- team decided to gate calib behind an ablation).

Reads results/mat_raw/report_{wl}.json (produced with --no-calib):
  switch_real -> SD hybrid (per-step single proposer, raw score select)
  compose     -> Ours (grafted head+tail, raw score)
Emits results/interp_validation/figures/mat_raw_hybrid_vs_ours.png
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WL = ["spider", "swebench", "bfcl", "specbench"]
RAW = "results/mat_raw"
FIGDIR = Path("results/interp_validation/figures")

rows = []
for wl in WL:
    r = json.load(open(f"{RAW}/report_{wl}.json"))
    a = r["arms"]
    rows.append(dict(wl=wl, hybrid=a["switch_real"]["K"], ours=a["compose"]["K"]))

x = np.arange(len(WL)); w = 0.36
hyb = [r["hybrid"] for r in rows]
our = [r["ours"] for r in rows]
C_HYB, C_OURS = "#9aa7b3", "#F58518"

fig, ax = plt.subplots(figsize=(9.2, 5.6))
b1 = ax.bar(x - w / 2, hyb, w, label="Suffix-Decoding hybrid (baseline)",
            color=C_HYB, edgecolor="k", linewidth=0.5, zorder=3)
b2 = ax.bar(x + w / 2, our, w, label="Ours: DFlash+Suffix compose",
            color=C_OURS, edgecolor="k", linewidth=0.5, zorder=3)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                f"{b.get_height():.2f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")
for i, r in enumerate(rows):
    d = 100.0 * (r["ours"] / r["hybrid"] - 1) if r["hybrid"] else 0.0
    ax.text(i, max(r["hybrid"], r["ours"]) + 0.28, f"{d:+.0f}%",
            ha="center", va="bottom", fontsize=10,
            color=("#1a7d1a" if d >= 0 else "#b00"), fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(WL, fontsize=11)
ax.set_ylabel("MAT  (mean accepted tokens per verify step)", fontsize=11)
ax.set_title("MAT: Suffix-Decoding hybrid vs Ours (compose)  —  RAW prob, no calibration",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=10, loc="upper right", frameon=True)
ax.grid(axis="y", alpha=0.2, zorder=0)
ax.set_ylim(0, max(max(hyb), max(our)) * 1.18)
fig.tight_layout()
FIGDIR.mkdir(parents=True, exist_ok=True)
fp = FIGDIR / "mat_raw_hybrid_vs_ours.png"
fig.savefig(fp, dpi=150, bbox_inches="tight")
print("saved ->", fp)
for r in rows:
    print(f"{r['wl']:<10} hybrid={r['hybrid']:.3f}  ours={r['ours']:.3f}  "
          f"delta={100*(r['ours']/r['hybrid']-1):+.1f}%")
