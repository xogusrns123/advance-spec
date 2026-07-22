#!/usr/bin/env python3
"""Single plot (not subplots): the compose vs SD-paper-hybrid gap under warming
OFF vs ON, grouped by workload. Per workload group, four bars
  [hybrid OFF, compose OFF | hybrid ON, compose ON]
(OFF = hatched, ON = solid; hybrid = purple, compose = green). The compose-hybrid
gap is annotated above each warming pair so you can see how warming changes it.
Reads warming_4way.json (produced by warming_effect_4way.py --plot).

-> readable_outputs/figures/mat/warming/WARMING_gap_by_workload.png
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("readable_outputs/figures/mat/warming")
WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
C_HYB, C_COM = "#9467BD", "#54A24B"

data = json.load(open(OUT / "warming_4way.json"))
x = np.arange(len(WLS))
w = 0.19
# offsets: OFF pair (left), ON pair (right), small gap in the middle
off = {"hyb_off": -1.6 * w, "com_off": -0.6 * w, "hyb_on": 0.6 * w, "com_on": 1.6 * w}

fig, ax = plt.subplots(figsize=(12.5, 6.2))


def bars(key, arm, color, hatch):
    vals = [data[wl]["arms"][arm][("on" if "on" in key else "off")] for wl in WLS]
    b = ax.bar(x + off[key], vals, w, color=color, edgecolor="k", linewidth=0.6,
               hatch=hatch, zorder=3)
    for bar in b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    return vals


ho = bars("hyb_off", "fallback", C_HYB, "//")
co = bars("com_off", "calib", C_COM, "//")
hn = bars("hyb_on", "fallback", C_HYB, "")
cn = bars("com_on", "calib", C_COM, "")

# compose - hybrid gap annotation above each warming pair
for i in range(len(WLS)):
    g_off, g_on = co[i] - ho[i], cn[i] - hn[i]
    yoff = max(ho[i], co[i])
    yon = max(hn[i], cn[i])
    ax.annotate(f"gap {g_off:+.2f}", (i + (off["hyb_off"] + off["com_off"]) / 2, yoff + 0.22),
                ha="center", fontsize=8.5, fontweight="bold",
                color="#1a7d1a" if g_off >= 0 else "#c0392b")
    ax.annotate(f"gap {g_on:+.2f}", (i + (off["hyb_on"] + off["com_on"]) / 2, yon + 0.22),
                ha="center", fontsize=8.5, fontweight="bold",
                color="#1a7d1a" if g_on >= 0 else "#c0392b")

ax.set_xticks(x)
ax.set_xticklabels([WL_NAME[wl] for wl in WLS], fontsize=11)
ax.set_ylabel("MAT  (mean accepted tokens / round)", fontsize=11)
ax.set_title("Compose vs SD-paper hybrid — gap under warming OFF / ON, per workload",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.2, zorder=0)
allv = ho + co + hn + cn
ax.set_ylim(0, max(allv) * 1.20)

handles = [
    plt.Rectangle((0, 0), 1, 1, fc=C_HYB, ec="k", hatch="//", label="SD-hybrid — warming OFF"),
    plt.Rectangle((0, 0), 1, 1, fc=C_COM, ec="k", hatch="//", label="Compose — warming OFF"),
    plt.Rectangle((0, 0), 1, 1, fc=C_HYB, ec="k", label="SD-hybrid — warming ON"),
    plt.Rectangle((0, 0), 1, 1, fc=C_COM, ec="k", label="Compose — warming ON"),
]
ax.legend(handles=handles, fontsize=9.5, loc="upper right", ncol=2, framealpha=0.92)
fig.tight_layout()
fp = OUT / "WARMING_gap_by_workload.png"
fig.savefig(fp, dpi=150, bbox_inches="tight")
print("saved ->", fp)
