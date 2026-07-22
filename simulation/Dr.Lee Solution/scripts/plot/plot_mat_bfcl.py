#!/usr/bin/env python3
"""Single-panel MAT bars for BFCLv4 CLEAN (conversation-disjoint split, NO leak).

Record = results/perpos_bfcl_conv/bfcl_v4.jsonl (57 eval / 59 warm calls), the clean
split where warm/eval come from DISJOINT conversations (even->warm, odd->eval). The
old results/perpos_bfcl/bfcl_v4.jsonl is a LEAKY row-level split — do not use.

Reproduces (previously an ad-hoc python -c) the mat_bfcl_clean.png figure, now with
the full selection ladder. BFCL clean is thinking-ON (capture includes <think>),
unlike thinking-OFF multislot/specbench — a different axis, noted in the title.
Bars: single(DFlash), single(Suffix), select(raw), select(calib), composition, oracle.

Run inside sglang-bench (figures dir is root-owned):
  cd "/workspace/simulation/Dr.Lee Solution" && python3 scripts/plot_mat_bfcl.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/workspace/simulation/Dr.Lee Solution")
RECORD = "results/perpos_bfcl_conv/bfcl_v4.jsonl"
PROPS = ["dflash", "suffix", "select", "calib", "chain", "oracle"]
LABELS = {"dflash": "single\n(DFlash)", "suffix": "single\n(Suffix)",
          "select": "per-depth\nselect (raw)", "calib": "per-depth\nselect (calib)",
          "chain": "composition", "oracle": "oracle"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "select": "#E45756",
          "calib": "#B94A8C", "chain": "#54A24B", "oracle": "#B279A2"}


def get_K():
    out = subprocess.run(
        ["python3", "scripts/replay_extension.py", "--record", RECORD, "--props", *PROPS,
         "--three-way", "--group-mode", "lenreset"],  # NO LOO: disjoint conversation split
        cwd=BASE, capture_output=True, text=True).stdout
    K = {}
    for line in out.splitlines():
        m = re.search(r"(\w+):\s*K=([0-9.]+)", line)
        if m and m.group(1) in PROPS:
            K[m.group(1)] = float(m.group(2))
    return K


def main():
    K = get_K()
    vals = [K[p] for p in PROPS]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(range(len(PROPS)), vals, width=0.62, color=[COLORS[p] for p in PROPS])
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
    ax.set_title("MAT — BFCLv4 clean, thinking-ON  (3-way disjoint, NO LOO)\n"
                 "Qwen3.5-27B + DFlash + Suffix   "
                 "tree=warm 59 / calibrate 25 / test 32 calls", fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fp = BASE / "readable_outputs" / "figures" / "ladders" / "mat_bfcl_clean.png"
    fig.savefig(fp, dpi=150); plt.close(fig)
    print(f"saved {fp.name}: " + "  ".join(f"{p}={K[p]:.2f}" for p in PROPS))


if __name__ == "__main__":
    main()
