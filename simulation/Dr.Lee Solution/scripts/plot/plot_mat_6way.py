#!/usr/bin/env python3
"""MAT per workload, 6-way extension of MAT_per_workload_4way (deployable
split): the two single proposers, the SD-paper hybrid, and the compose ladder
raw -> calib (beta head + isotonic tail) -> succession rescore (0-param), plus
the handoff oracle. Same format/colors as plot_mat_4way; compose family drawn
as a light->dark green ladder.

Reads the *_split replay logs + the split fallback sweep (tau* picked on the
calib half, K reported on the test half). Emits
readable_outputs/figures/mat(deployable)/MAT_per_workload_6way.png.

  python3 scripts/plot/plot_mat_6way.py
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

PROPS = ["dflash", "suffix", "fallback", "calib_raw", "calib", "calib_succ", "oracle"]
LABELS = {"dflash": "DFlash (single)", "suffix": "Suffix (single)",
          "fallback": "SD-paper hybrid (fallback)",
          "calib_raw": "Compose (raw)",
          "calib": "Compose (calib: isotonic tail)",
          "calib_succ": "Compose (succession rescore)",
          "oracle": "Oracle (best handoff)"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "fallback": "#9467BD",
          "calib_raw": "#cfe8c8", "calib": "#54A24B", "calib_succ": "#1b5e1b",
          "oracle": "#E45756"}
# group log -> which prop each parsed "calib:"/"dflash:"... line feeds
GROUPS = [("singles_split", None),          # dflash + suffix
          ("calib_raw_split", "calib_raw"),  # compose raw   ("calib:" line)
          # calib arm = ISOTONIC TAIL ONLY (identity head): same raw DFlash head
          # as the succession arm, so the two differ only in tail treatment
          # (per-workload fitted isotonic vs derived succession)
          ("calib_isotail_split", "calib"),
          # succession arm: RAW dflash head (the fixed 0.69/0.29 affine card is
          # discarded) + succession-rescored tail
          ("calib_succrawhead_split", "calib_succ"),
          ("oracle_split", None)]            # oracle

NOTE = ("deployable split: compose calibrators + hybrid τ* fit on the CALIB half "
        "(even within-label convs), MAT measured on the disjoint TEST half (odd)")


def load():
    K = {}
    for ds in p4.DS:
        K[ds] = {}
        for grp, rename in GROUPS:
            o, _ = p4.parse_log(LOGS / f"mat_{ds}_4way_{grp}.replay.txt")
            if rename is not None:
                if "calib" in o:
                    K[ds][rename] = o["calib"]
            else:
                K[ds].update(o)
    p4.SWEEP_DIR = SWEEP
    for ds, (tau, entry) in p4._load_fallback().items():
        p4.FB_TAUS[ds] = tau
        K[ds]["fallback"] = (entry["K"], entry.get("rounds", 0))
    return K


def main():
    # reuse p4.bars() with the 7-arm spec
    p4.PROPS, p4.LABELS, p4.COLORS = PROPS, LABELS, COLORS
    K = load()
    OUT.mkdir(parents=True, exist_ok=True)
    corner = ""
    if p4.FB_GRID:
        corner = ("τ swept: {" + ", ".join(f"{t:g}" for t in p4.FB_GRID) + "}\n"
                  + ("τ* picked on the calib half; bar = MAT on the test half"
                     if p4.FB_SPLIT else "τ* = best per column"))

    wls = [ds for ds in p4.DS if K.get(ds)]
    fig, ax = plt.subplots(figsize=(2.4 + 3.0 * len(wls), 5.8))
    p4.bars(ax, wls, lambda ds: K[ds], tau_of=lambda ds: p4.FB_TAUS.get(ds),
            fs_val=7)
    ax.set_xticks(range(len(wls)))
    ax.set_xticklabels([p4.DS_NAME[ds] for ds in wls], fontsize=8.5)
    ax.set_ylabel("mean accept length  (tokens)", fontsize=11)
    ax.set_title("Dr.Lee extension — MAT per workload (6-way)\n" + NOTE, fontsize=12)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    if corner:
        ax.text(0.99, 0.99, corner, transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color="#555555")
    fig.tight_layout()
    fp = OUT / "MAT_per_workload_6way.png"
    fig.savefig(fp, dpi=150)
    print(f"saved {fp}")
    for ds in wls:
        print(f"  [{ds}] " + "  ".join(
            f"{p}={K[ds].get(p, (0, 0))[0]:.2f}({K[ds].get(p, (0, 0))[1]})"
            for p in PROPS))


if __name__ == "__main__":
    main()
