#!/usr/bin/env python3
"""Render the offline iso-calibration reliability map + PAV overlay.

confidence_calibration.ipynb-style reliability diagram, one panel per
edge source (eagle3 / suffix), built from the extension_isofit_* sample
dumps (train split). Bars = fixed-width 0.05 binned accept rate
(EAGLE-2 Fig.6 analog); red line = the deployed PAV isotonic step map
(the exact curve _FrozenIsoCalibrator uses at eval). Second figure =
per-bin sample counts (log y).

Usage:
    python3 simulation/notebooks/_render_iso_calib.py \
        [--samples GLOB] [--map JSON] [--tag bfcl_v4_s8k16]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # works on host and in container
FIG_DIR = ROOT / "simulation" / "notebooks" / "figures"

EAGLE3_COLOR = "#1f77b4"   # tab:blue  (matches confidence_calibration.ipynb)
SUFFIX_COLOR = "#ff7f0e"   # tab:orange
COLORS = {"eagle": EAGLE3_COLOR, "suffix": SUFFIX_COLOR}
TITLES_SHRUNK = {"eagle": "eagle3 (draft-head softmax)",
                 "suffix": "suffix (trie count ratio, Jeffreys-shrunk)"}
XLABELS_SHRUNK = {"eagle": "per-edge draft prob",
                  "suffix": "per-edge prob (count-shrunk)"}
TITLES_RAW = {"eagle": "eagle3 (draft-head softmax)",
              "suffix": "suffix (trie count ratio, RAW)"}
XLABELS_RAW = {"eagle": "per-edge draft prob",
               "suffix": "per-edge prob (raw c/n)"}


def load_samples(pattern: str) -> dict[str, np.ndarray]:
    data: dict[str, list] = {"eagle": [], "suffix": []}
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no sample files match {pattern!r}")
    for fp in files:
        with open(fp) as f:
            for line in f:
                blob = json.loads(line)
                for grp, rows in blob.get("samples", {}).items():
                    data.setdefault(grp, []).extend(rows)
    return {g: np.asarray(rows, dtype=np.float64)
            for g, rows in data.items() if rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples",
                    default=str(ROOT / "simulation/results/iso_calib/"
                                       "bfcl_v4_s8k16_samples.jsonl.*.part"))
    ap.add_argument("--map",
                    default=str(ROOT / "simulation/results/iso_calib/"
                                       "bfcl_v4_s8k16_iso_map.json"))
    ap.add_argument("--tag", default="bfcl_v4_s8k16")
    ap.add_argument("--raw", action="store_true",
                    help="samples were collected WITHOUT Jeffreys shrinkage "
                         "(SIM_SUFFIX_SHRINK=0) — adjusts labels only")
    args = ap.parse_args()

    titles = TITLES_RAW if args.raw else TITLES_SHRUNK
    xlabels = XLABELS_RAW if args.raw else XLABELS_SHRUNK

    samples = load_samples(args.samples)
    with open(args.map) as f:
        iso_map = json.load(f)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    groups = [g for g in ("eagle", "suffix") if g in samples]
    edges = np.arange(0.0, 1.0001, 0.05)
    centers = (edges[:-1] + edges[1:]) / 2
    width = 0.05 * 0.92

    # --- Figure 1: reliability bars + PAV overlay ---
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5),
                             sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, grp in zip(axes, groups):
        arr = samples[grp]
        p, y = arr[:, 0], arr[:, 2]
        idx = np.clip(np.digitize(p, edges) - 1, 0, len(centers) - 1)
        trials = np.bincount(idx, minlength=len(centers)).astype(float)
        accepts = np.bincount(idx, weights=y, minlength=len(centers))
        rates = np.where(trials > 0, accepts / np.maximum(trials, 1), np.nan)

        ax.bar(centers, rates, width=width, color=COLORS[grp],
               edgecolor="white", linewidth=0.4, alpha=0.85,
               label="binned accept rate (0.05 bins)")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2,
                alpha=0.8, label="y = x (perfectly calibrated)")

        # Deployed PAV isotonic step map (exact eval-time lookup curve).
        gm = iso_map["groups"].get(grp)
        if gm is not None:
            xs = list(map(float, gm["x"])) + [1.0]
            ys = list(map(float, gm["y"]))
            ys = ys + [ys[-1]]
            ax.step(xs, ys, where="post", color="red", linewidth=2.0,
                    label="PAV isotonic (deployed map)")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.set_xticks(np.arange(0, 1.01, 0.1))
        ax.grid(True, alpha=0.3)
        ax.set_title(titles[grp], fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabels[grp])
        ax.text(0.02, 0.96, f"N = {len(p):,}", transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="gray"))
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9,
                  bbox_to_anchor=(0.02, 0.90))
    axes[0].set_ylabel("Accept rate", fontsize=11)
    fig.suptitle(f"Per-edge accept-rate calibration — {args.tag} "
                 f"(train split)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out1 = FIG_DIR / f"iso_calib_reliability_{args.tag}.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: per-bin sample counts (log y) ---
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 3.2),
                             sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, grp in zip(axes, groups):
        arr = samples[grp]
        p = arr[:, 0]
        idx = np.clip(np.digitize(p, edges) - 1, 0, len(centers) - 1)
        trials = np.bincount(idx, minlength=len(centers)).astype(float)
        ax.bar(centers, trials, width=width, color=COLORS[grp],
               edgecolor="white", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(titles[grp], fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabels[grp])
    axes[0].set_ylabel("samples / bin (log)", fontsize=10)
    fig.suptitle(f"Sample counts — {args.tag} (train split)",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    out2 = FIG_DIR / f"iso_calib_counts_{args.tag}.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(out1)
    print(out2)


if __name__ == "__main__":
    main()
