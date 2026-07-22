#!/usr/bin/env python3
"""Calibration-ladder figure for the compose arm: three head/tail calibration
settings on the SAME controller / records, per workload, with the SD-paper hybrid
baseline overlaid as the bar-to-beat.

  raw     identity (raw DFlash conf + raw arctic score; no calibration)
  affine  Dr.Lee original: head = 0.69*conf+0.29, tail identity
  beta+iso  beta head hazard + isotonic tail (deployed calibration)

Reads deck replay logs: mat_{ds}_4way_calib{,_affine,_raw}.replay.txt + fallback
sweep for the hybrid. Emits readable_outputs/figures/mat/CALIB_effect.png.

  docker exec sglang-bench bash -lc \
    "cd '/workspace/simulation/Dr.Lee Solution' && PYTHONPATH=/workspace python3 scripts/plot_calib_effect.py"
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RLOG = Path("readable_outputs/figures/replay_logs")

_ap = argparse.ArgumentParser()
_ap.add_argument("--log-suffix", default="",
                 help="read mat_{ds}_4way_calib{variant}<sfx>.replay.txt (e.g. _split)")
_ap.add_argument("--sweep-dir",
                 default="/workspace/simulation/results/pipeline_4way/segments",
                 help="dir with fallback_sweep_fresh_{ds}.json for the hybrid line")
_ap.add_argument("--out-dir", default="readable_outputs/figures/mat/calibration")
_ap.add_argument("--note", default="", help="extra title line (e.g. deployable split)")
_args = _ap.parse_args()

SEG = Path(_args.sweep_dir)
OUT = Path(_args.out_dir)
LOGSFX = _args.log_suffix

WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench\nVerified",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
ARMS = [("_raw", "no calib", "#cfe8c8"),
        ("_H-logistic", "head calib. (logistic)", "#93cd8a"),
        ("_isotail", "tail calib. (isotonic)", "#57a457"),
        ("_logiso", "head&tail calib.", "#2f7a2f")]
C_HYB = "#9467BD"


def compose_k(ds, sfx):
    txt = (RLOG / f"mat_{ds}_4way_calib{sfx}{LOGSFX}.replay.txt").read_text()
    return float(re.search(r"calib: K=([0-9.]+)", txt).group(1))


def hybrid_k(ds):
    d = next(iter(json.load(open(SEG / f"fallback_sweep_fresh_{ds}.json")).values()))
    if set(d) == {"calib", "test"}:       # deployable split: tau* on calib, K on test
        best_tau = max(d["calib"], key=lambda t: d["calib"][t]["K"])
        return d["test"][best_tau]["K"]
    return max(v["K"] for v in d.values())


K = {ds: {sfx: compose_k(ds, sfx) for sfx, _, _ in ARMS} for ds in WLS}
hyb = {ds: hybrid_k(ds) for ds in WLS}
# sort by calibration lift (beta over raw)
order = sorted(WLS, key=lambda d: -(K[d]["_logiso"] / K[d]["_raw"]))

x = np.arange(len(order))
na = len(ARMS)
w = 0.16
half = (na - 1) / 2.0
fig, ax = plt.subplots(figsize=(14.5, 6.8))
for i, (sfx, lab, col) in enumerate(ARMS):
    vals = [K[d][sfx] for d in order]
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
    # raw->beta lift %
    lift = 100.0 * (K[d]["_logiso"] / K[d]["_raw"] - 1)
    top = max(K[d]["_logiso"], hyb[d])
    ax.annotate(f"raw→calib +{lift:.0f}%", (i, top + 0.34), ha="center",
                fontsize=9.5, fontweight="bold", color="#1a7d1a")

ax.hlines([], [], [], color=C_HYB, lw=2.4,
          label="SD-paper hybrid (per-step switch, best τ)")
ax.set_xticks(x)
ax.set_xticklabels([WL_NAME[d] for d in order], fontsize=11)
ax.set_ylabel("compose MAT  (mean accepted tokens / verify step)", fontsize=11)
ax.set_title("Calibration ladder on the compose arm — no calib vs head calib. (logistic) "
             "vs tail calib. (isotonic) vs head&tail\n(same controller, same records; purple "
             "line = SD-paper hybrid, the baseline compose must beat)"
             + (f"\n{_args.note}" if _args.note else ""),
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9.5, loc="upper right", frameon=True, ncol=1)
ax.grid(axis="y", alpha=0.2, zorder=0)
ax.set_ylim(0, max(max(K[d]["_logiso"] for d in WLS), max(hyb.values())) * 1.22)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fp = OUT / "CALIB_effect.png"
fig.savefig(fp, dpi=150, bbox_inches="tight")
print("saved ->", fp)
print(f"{'wl':<10}{'nocal':>7}{'head':>8}{'tail':>9}{'h&t':>9}{'hybrid':>8}")
for d in order:
    print(f"{d:<10}{K[d]['_raw']:>7.2f}{K[d]['_H-logistic']:>8.2f}{K[d]['_isotail']:>9.2f}"
          f"{K[d]['_logiso']:>9.2f}{hyb[d]:>8.2f}")
