#!/usr/bin/env python3
"""One MAT-per-workload ladder figure PER final methodology (deployable split,
same format as MAT_per_workload_4way/6way): the two single proposers, the
SD-paper hybrid, compose raw, compose corrected by THE method, and the oracle.

  online     ONLINE windowed calibration (logistic head + isotonic tail, win 8k)
  twoscalar  fixed weights: head min(1, 1.4*conf), tail 0.125*score
  succession raw head + succession-rescored tail (derived, no fitting)

Emits readable_outputs/figures/mat(deployable)/MAT_per_workload_{method}.png

  python3 scripts/plot/plot_mat_ladders.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_mat_4way as p4

BASE = Path(__file__).resolve().parents[2]
LOGS = BASE / "readable_outputs" / "figures" / "replay_logs"
OUT = BASE / "readable_outputs" / "figures" / "mat(deployable)"
SWEEP = "/workspace/simulation/results/pipeline_deployable/segments"

NOTE = ("deployable split: every fitted quantity from the CALIB half (even "
        "within-label convs), MAT measured on the disjoint TEST half (odd)")

# method key -> (compose log group, bar label, title line)
METHODS = {
    "online": ("calib_onl8000_split",
               "Compose (online calib: logistic head + isotonic tail)",
               "online windowed calibration (window 8k verify labels, no offline fit)"),
    "twoscalar": ("calib_h14tw125_split",
                  "Compose (fixed weights: min(1, 1.4·conf), 0.125·score)",
                  "fixed-weight compose (two universal constants)"),
    "succession": ("calib_succrawhead_split",
                   "Compose (succession rescore)",
                   "raw head + succession-rescored tail (derived, no fitting)"),
}

PROPS = ["dflash", "suffix", "fallback", "calib_raw", "method", "oracle"]
LABELS_BASE = {"dflash": "DFlash (single)", "suffix": "Suffix (single)",
               "fallback": "SD-paper hybrid (fallback)",
               "calib_raw": "Compose (raw)",
               "oracle": "Oracle (best handoff)"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "fallback": "#9467BD",
          "calib_raw": "#cfe8c8", "method": "#2f7a2f", "oracle": "#E45756"}


def load(method_grp):
    K = {}
    for ds in p4.DS:
        K[ds] = {}
        o, _ = p4.parse_log(LOGS / f"mat_{ds}_4way_singles_split.replay.txt")
        K[ds].update(o)
        o, _ = p4.parse_log(LOGS / f"mat_{ds}_4way_oracle_split.replay.txt")
        K[ds].update(o)
        o, _ = p4.parse_log(LOGS / f"mat_{ds}_4way_calib_raw_split.replay.txt")
        if "calib" in o:
            K[ds]["calib_raw"] = o["calib"]
        o, _ = p4.parse_log(LOGS / f"mat_{ds}_4way_{method_grp}.replay.txt")
        if "calib" in o:
            K[ds]["method"] = o["calib"]
    p4.SWEEP_DIR = SWEEP
    for ds, (tau, entry) in p4._load_fallback().items():
        p4.FB_TAUS[ds] = tau
        K[ds]["fallback"] = (entry["K"], entry.get("rounds", 0))
    return K


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for method, (grp, label, subtitle) in METHODS.items():
        p4.PROPS = PROPS + ["calib"]     # parse_log filters by this; "calib:" lines
        p4.LABELS = dict(LABELS_BASE, method=label)
        p4.COLORS = COLORS
        K = load(grp)
        p4.PROPS = PROPS                 # drawing order (no raw "calib" bar)
        corner = ""
        if p4.FB_GRID:
            corner = ("τ swept: {" + ", ".join(f"{t:g}" for t in p4.FB_GRID) + "}\n"
                      + ("τ* picked on the calib half; bar = MAT on the test half"
                         if p4.FB_SPLIT else "τ* = best per column"))
        wls = [ds for ds in p4.DS if K.get(ds)]
        fig, ax = plt.subplots(figsize=(2.4 + 2.7 * len(wls), 5.6))
        p4.bars(ax, wls, lambda ds: K[ds], tau_of=lambda ds: p4.FB_TAUS.get(ds),
                fs_val=7.5)
        ax.set_xticks(range(len(wls)))
        ax.set_xticklabels([p4.DS_NAME[ds] for ds in wls], fontsize=8.5)
        ax.set_ylabel("mean accept length  (tokens)", fontsize=11)
        ax.set_title(f"Dr.Lee extension — MAT per workload · {subtitle}\n{NOTE}",
                     fontsize=12)
        ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=2)
        if corner:
            ax.text(0.99, 0.99, corner, transform=ax.transAxes, ha="right",
                    va="top", fontsize=9, color="#555555")
        fig.tight_layout()
        fp = OUT / f"MAT_per_workload_{method}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f"saved {fp}")
        for ds in wls:
            print(f"  [{ds}] " + "  ".join(
                f"{p}={K[ds].get(p, (0, 0))[0]:.2f}" for p in PROPS))


if __name__ == "__main__":
    main()
