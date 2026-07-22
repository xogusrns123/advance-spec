#!/usr/bin/env python3
"""CALIB_comparison: per workload, the MAT of calibrating ONLY one proposer's
signal, four methods each (option B — each side's natural calibrators; the other
side stays raw).

  DFlash head only (tail = raw):   raw · affine(0.69·raw+0.29) · logistic · beta   (blue)
  Suffix tail only (head = raw):   raw · scaled · linear · isotonic                (orange)

Two grouped-bar panels (head / tail), workloads on x. raw is the shared left bar
in both (head raw == tail raw == full-raw compose). SD-paper hybrid drawn as a
per-workload reference line. Reads the deck replay logs. Emits
readable_outputs/figures/mat/calibration/CALIB_comparison.png.
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
_ap.add_argument("--note", default="", help="extra suptitle line (e.g. deployable split)")
_args = _ap.parse_args()

SEG = Path(_args.sweep_dir)
OUT = Path(_args.out_dir)
LOGSFX = _args.log_suffix
WLS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench\nVerified",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
# (tag suffix in log filename, label, color)
HEAD = [("_raw", "raw", "#c6dbef"),
        ("_affine", "fixed 0.69·raw+0.29 (Dr.Lee)", "#addd8e"),
        ("_H-linear", "linear (a·raw+b, fitted)", "#9ecae1"),
        ("_H-logistic", "logistic", "#6baed6"), ("_betahead", "beta", "#2f6fb0")]
TAIL = [("_raw", "raw", "#fdd0a2"),
        ("_T-linear", "linear", "#fd8d3c"), ("_isotail", "isotonic", "#d94801")]
C_HYB = "#9467BD"


def ck(ds, sfx):
    fp = RLOG / f"mat_{ds}_4way_calib{sfx}{LOGSFX}.replay.txt"
    return float(re.search(r"calib: K=([0-9.]+)", fp.read_text()).group(1))


def hyb(ds):
    d = next(iter(json.load(open(SEG / f"fallback_sweep_fresh_{ds}.json")).values()))
    if set(d) == {"calib", "test"}:       # deployable split: tau* on calib, K on test
        best_tau = max(d["calib"], key=lambda t: d["calib"][t]["K"])
        return d["test"][best_tau]["K"]
    return max(v["K"] for v in d.values())


H = {ds: [ck(ds, s) for s, _, _ in HEAD] for ds in WLS}
T = {ds: [ck(ds, s) for s, _, _ in TAIL] for ds in WLS}
HYB = {ds: hyb(ds) for ds in WLS}

fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
x = np.arange(len(WLS))
for ax, data, spec, title in (
        (axes[0], H, HEAD, "DFlash HEAD calibration only  (suffix tail = raw)"),
        (axes[1], T, TAIL, "Suffix TAIL calibration only  (dflash head = raw)")):
    n = len(spec); w = 0.8 / n
    for j, (sfx, lab, col) in enumerate(spec):
        vals = [data[ds][j] for ds in WLS]
        b = ax.bar(x + (j - (n - 1) / 2) * w, vals, w, label=lab, color=col,
                   edgecolor="k", linewidth=0.4, zorder=3)
        for bar in b:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    for i, ds in enumerate(WLS):
        ax.hlines(HYB[ds], i - 0.45, i + 0.45, color=C_HYB, lw=2.2, zorder=4)
    ax.hlines([], [], [], color=C_HYB, lw=2.2, label="SD-paper hybrid")
    ax.set_xticks(x); ax.set_xticklabels([WL_NAME[d] for d in WLS], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=1, frameon=True)
    ax.grid(axis="y", alpha=0.2, zorder=0)
axes[0].set_ylabel("compose MAT  (mean accepted tokens / verify step)", fontsize=11)
ymax = max(max(max(H[d]), max(T[d]), HYB[d]) for d in WLS)
axes[0].set_ylim(0, ymax * 1.16)
fig.suptitle("Isolating each calibrator — head-only (blue) vs tail-only (orange); "
             "raw is the shared baseline, purple line = SD-paper hybrid"
             + (f"\n{_args.note}" if _args.note else ""),
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fp = OUT / "CALIB_comparison.png"
fig.savefig(fp, dpi=150, bbox_inches="tight")
print("saved ->", fp)
print(f"{'wl':<10} HEAD[raw/affine/log/beta]        TAIL[raw/scaled/lin/iso]      hybrid")
for ds in WLS:
    print(f"{ds:<10} {['%.2f' % v for v in H[ds]]}  {['%.2f' % v for v in T[ds]]}  {HYB[ds]:.2f}")
