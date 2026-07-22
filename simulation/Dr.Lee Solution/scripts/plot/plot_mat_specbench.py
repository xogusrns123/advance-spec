#!/usr/bin/env python3
"""SpecBench per-subtask MAT — grouped bars (DFlash/Suffix/composition/oracle).
SpecBench is single-turn with NO warm corpus -> Suffix is COLD (the theory's cold
regime endpoint: composition.md slide 11, cold T->0 => standalone==oracle). CPU.
  python3 scripts/plot_mat_specbench.py
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
import numpy as np

BASE = Path("/workspace/simulation/Dr.Lee Solution")
PROPS = ["dflash", "suffix", "chain", "oracle"]
LEGEND = {"dflash": "single (DFlash)", "suffix": "single (Suffix, cold)",
          "chain": "composition", "oracle": "oracle"}
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "chain": "#54A24B", "oracle": "#B279A2"}


def per_task():
    out = subprocess.run(
        ["python3", "scripts/replay_extension.py", "--record",
         "results/perpos/specbench.jsonl", "--props", *PROPS],
        cwd=BASE, capture_output=True, text=True).stdout
    data = {}
    for line in out.splitlines():
        m = re.match(r"\s*\[(\w+)\]\s+(.*)", line)
        if not m:
            continue
        task, rest = m.group(1), m.group(2)
        vals = dict(re.findall(r"(\w+)=([0-9.]+)", rest))
        if all(p in vals for p in PROPS):
            data[task] = {p: float(vals[p]) for p in PROPS}
    return data


def main():
    data = per_task()
    # order subtasks by DFlash MAT (structured -> free)
    tasks = sorted(data, key=lambda t: -data[t]["dflash"])
    x = np.arange(len(tasks))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for i, p in enumerate(PROPS):
        vals = [data[t][p] for t in tasks]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=LEGEND[p], color=COLORS[p])
        if p == "oracle":
            for b in bars:
                b.set_hatch("//"); b.set_edgecolor("white")
        for xi, v in zip(x + (i - 1.5) * w, vals):
            ax.text(xi, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9, rotation=15)
    ax.set_ylabel("MAT  (accepted tokens / round)")
    ax.set_title("SpecBench per-subtask MAT — Qwen3.5-27B + DFlash + Suffix (cold)\n"
                 "single-turn = cold regime: Suffix≈0, composition≈DFlash≈oracle "
                 "(no repetition to memorize)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(0, max(data[t]["oracle"] for t in tasks) * 1.2)
    fig.tight_layout()
    out = BASE / "readable_outputs" / "figures" / "legacy" / "mat_specbench.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"saved {out}")
    for t in tasks:
        print(f"  {t:16s} " + "  ".join(f"{p}={data[t][p]:.2f}" for p in PROPS))


if __name__ == "__main__":
    main()
