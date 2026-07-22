#!/usr/bin/env python3
"""Picture5-1: the MIRROR of Picture5 — the DFlash → suffix boundary.

Same boundary-aligned format as regen_picture5.py, but aligned on the START of
an R (>=4-gram repeat) region instead of its end. offset o = p - region_start
(o=-1 last novel/G token, o=0 first repeated/R token):
    suffix (retrieval): 1[s(p) >= 1]  — jumps UP the instant the repeat begins
    DFlash  (model)   : 1[a(p) >= 1]  — flat across the boundary

Boundaries = STARTS of R regions under the canonical n-gram-repeat partition
(_gr.rg_cover(s, 4)), i.e. G->R transitions, pooled over the agentic workloads
(swebench/spider/bfcl/tau2). No gap/island smoothing.

  PYTHONPATH=/workspace python3 scripts/plot/regen_picture5_1.py
"""
from __future__ import annotations
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gr import rg_cover  # noqa: E402

DATA = BASE / "results" / "interp_validation"
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"
ORANGE, BLUE, NAVY = "#D85A30", "#2A78D6", "#1F2A44"
POOL = ["swebench", "spider", "bfcl", "tau2"]    # same pool as Picture5
RUN_LEN, WIN = 4, 8


def load(w):
    with gzip.open(DATA / f"curves_{w}.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f]


def main():
    suf, dfl = defaultdict(list), defaultdict(list)
    for w in POOL:
        for r in load(w):
            s, a = r["s"], r["a"]
            L = len(s)
            if L < 2:
                continue
            cov = rg_cover(s, RUN_LEN)          # True = R (>=4-gram repeat)
            # boundary = G->R transition; b = first repeated token (o=0)
            starts = [i for i in range(1, L) if not cov[i - 1] and cov[i]]
            for b in starts:
                for off in range(-WIN, WIN + 1):
                    p = b + off             # boundary = START of the repeat run
                    if 0 <= p < L:
                        suf[off].append(1 if s[p] >= 1 else 0)
                        if a[p] >= 0:
                            dfl[off].append(1 if a[p] >= 1 else 0)
    offs = list(range(-WIN, WIN + 1))
    sy = [sum(suf[o]) / len(suf[o]) for o in offs]
    dy = [sum(dfl[o]) / len(dfl[o]) for o in offs]

    fig, ax = plt.subplots(figsize=(8.7, 4.5))
    ax.axvline(-0.5, color=NAVY, ls=":", lw=1.5)
    ax.text(-0.35, 1.0, "boundary (G → R: start of repeat)", fontsize=11.5,
            color=NAVY, va="top", ha="left")
    ax.plot(offs, sy, color=ORANGE, lw=2.8, label="suffix (retrieval)")
    ax.plot(offs, dy, color=BLUE, lw=2.8, ls="--", label="DFlash (model)")
    ax.set_xlabel("token offset relative to the boundary", fontsize=13)
    ax.set_ylabel("conditional acceptance  $a_i$", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-WIN, WIN)
    ax.set_xticks(range(-WIN, WIN + 1, 2))
    ax.legend(loc="center right", frameon=False, fontsize=12.5)
    ax.tick_params(labelsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "Picture5-1.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")
    print(f"  suffix: o=-1 {sy[WIN-1]:.2f} -> o=0 {sy[WIN]:.2f}  (jump)")
    print(f"  dflash: o=-1 {dy[WIN-1]:.2f} -> o=0 {dy[WIN]:.2f}")


if __name__ == "__main__":
    main()
