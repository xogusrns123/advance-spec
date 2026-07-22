#!/usr/bin/env python3
"""MAT per workload, 3-way: SD-paper hybrid / Compose (no calibration, raw) /
Oracle. Values lifted directly from the already-rendered MAT_per_workload_4way.png
(raw deck). x-axis = workload names only. -> mat_bars/MAT_per_workload_3way.png
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WLS = ["Spec-Bench", "BFCL v4", "SWE-bench", "Spider2-DBT", "τ²-bench"]
# (hybrid, compose_raw, oracle) — read off MAT_per_workload_4way.png (raw version)
# all-eval (no warm split): HYB=SD-paper hybrid@τ*, COM=compose raw, ORC=oracle
HYB = [4.08, 3.68, 2.94, 3.62, 3.14]
COM = [2.75, 3.24, 3.32, 3.91, 3.44]
ORC = [4.82, 5.04, 4.31, 5.42, 4.60]
ARMS = [("SD-paper hybrid", HYB, "#9467BD"),
        ("Compose", COM, "#54A24B"),
        ("Oracle (best handoff)", ORC, "#E45756")]

x = np.arange(len(WLS)); w = 0.26
fig, ax = plt.subplots(figsize=(10.5, 5.8))
for i, (lab, vals, col) in enumerate(ARMS):
    b = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col,
               edgecolor="k", linewidth=0.5, zorder=3)
    for bar in b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(WLS, fontsize=11)
ax.set_ylabel("MAT  (mean accepted tokens / verify step)", fontsize=11)
ax.set_title("MAT per workload — SD-paper hybrid vs Compose vs Oracle", fontsize=13,
             fontweight="bold")
ax.legend(fontsize=10, loc="upper right", frameon=True)
ax.grid(axis="y", alpha=0.2, zorder=0)
ax.set_ylim(0, max(max(ORC), max(HYB)) * 1.16)
fig.tight_layout()
fp = Path("readable_outputs/figures/mat/mat_bars/MAT_per_workload_3way.png")
fig.savefig(fp, dpi=150, bbox_inches="tight")
print("saved ->", fp)
