#!/usr/bin/env python3
"""Per-k MAT bar charts (single panel each) from the offline replay.
Bars: single(DFlash), single(Suffix), composition(chain), oracle.
MAT = accepted tokens / round. CPU only. Run in docker:
  python3 scripts/plot_mat_bars.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import os
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/workspace/simulation/Dr.Lee Solution")
REC_DIR = os.environ.get("RECORD_DIR", "results/perpos")   # e.g. results/perpos_inf
FIG_SUFFIX = os.environ.get("FIG_SUFFIX", "")               # e.g. _inf
KS = [0, 1, 2, 4, 8]
PROPS = ["dflash", "suffix", "select", "calib", "chain", "oracle"]
LABELS = {"dflash": "single\n(DFlash)", "suffix": "single\n(Suffix)",
          "select": "per-depth\nselect (raw)", "calib": "per-depth\nselect (calib)",
          "chain": "composition", "oracle": "oracle"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "select": "#E45756",
          "calib": "#B94A8C", "chain": "#54A24B", "oracle": "#B279A2"}


def get_K(k):
    out = subprocess.run(
        ["python3", "scripts/replay_extension.py", "--record",
         f"{REC_DIR}/multislot_k{k}.jsonl", "--props", *PROPS,
         "--three-way", "--group-mode", "rid"],  # NO LOO: disjoint prompt split (calib on even rids)
        cwd=BASE, capture_output=True, text=True).stdout
    K = {}
    for line in out.splitlines():
        m = re.search(r"(\w+):\s*K=([0-9.]+)", line)
        if m and m.group(1) in PROPS:
            K[m.group(1)] = float(m.group(2))
    return K


def main():
    outdir = BASE / "readable_outputs" / "figures" / "multislot"
    outdir.mkdir(parents=True, exist_ok=True)
    allK = {}
    for k in KS:
        K = get_K(k)
        allK[k] = K
        vals = [K[p] for p in PROPS]
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        bars = ax.bar(range(len(PROPS)), vals, width=0.62,
                      color=[COLORS[p] for p in PROPS])
        for _op in ("sel_oracle", "oracle"):               # both oracles = ceilings
            if _op in PROPS:
                _bi = PROPS.index(_op)
                bars[_bi].set_hatch("//"); bars[_bi].set_edgecolor("white")
        for x, v in enumerate(vals):
            ax.text(x, v + max(vals) * 0.015, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(PROPS)))
        ax.set_xticklabels([LABELS[p] for p in PROPS], fontsize=9.5)
        ax.set_ylabel("MAT  (accepted tokens / round)", fontsize=10)
        nov = {0: "0 (exact repeat)", 1: "1", 2: "2", 4: "4", 8: "8 (all novel)"}[k]
        ax.set_title(f"MAT — multislot  k={k}   (novel slots: {nov})\n"
                     f"Qwen3.5-27B + DFlash + Suffix", fontsize=10.5)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="y", alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fp = outdir / f"mat_k{k}{FIG_SUFFIX}.png"
        fig.savefig(fp, dpi=150); plt.close(fig)
        print(f"saved {fp.name}: " + "  ".join(f"{p}={K[p]:.1f}" for p in PROPS))
    print(f"\nfigures -> {outdir}")


if __name__ == "__main__":
    main()
