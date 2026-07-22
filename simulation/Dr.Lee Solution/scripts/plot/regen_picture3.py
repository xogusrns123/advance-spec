#!/usr/bin/env python3
"""Regenerate synthetic Picture3 with REAL data — same CCDF format.

Provenance partition (Picture2/3), canonical n-gram-repeat definition (_gr):
a token is R (repeated) iff covered by a >=4-gram repeat, rg_cover(s, 4); else G.
On the arm-independent interp_validation s(p) curves:
  |R| = length of a maximal repeated run
  |G| = length of an interior novel gap (between two repeated runs)
pooled over the agentic workloads (swebench / spider / bfcl / tau2).

Two complementary-CDF (survival) curves S(x) = P(length > x), log-x, plus the
"bridgeable" band x<=3 and the measured P(|G| <= 3).

  PYTHONPATH=/workspace python3 scripts/plot/regen_picture3.py
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _gr import rg_cover

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "results" / "interp_validation"
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"

ORANGE, BLUE, NAVY = "#D85A30", "#2A78D6", "#1F2A44"
LIGHTBLUE = "#C3D9F3"
POOL = ["swebench", "spider", "bfcl", "tau2"]
BRIDGE = 3                      # model-head reach (tokens)


def load(w):
    with gzip.open(DATA / f"curves_{w}.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f]


def runs_gaps(curves):
    R, G = [], []
    for r in curves:
        s = r["s"]
        L = len(s)
        if L < 2:
            continue
        w = rg_cover(s)                 # True = R (>=4-gram repeat), False = G
        runs, i = [], 0
        while i < L:
            j = i
            while j < L and w[j] == w[i]:
                j += 1
            runs.append((i, j, w[i]))
            i = j
        for idx, (b, e, iw) in enumerate(runs):
            if iw:
                R.append(e - b)
            elif 0 < idx < len(runs) - 1:
                G.append(e - b)
    return R, G


def survival(lengths, xmax):
    """S(x) = P(L > x) for x = 0..xmax, as step arrays."""
    n = len(lengths)
    xs = list(range(0, xmax + 1))
    ys = [sum(1 for v in lengths if v > x) / n for x in xs]
    return xs, ys


def median(x):
    x = sorted(x)
    return x[len(x) // 2]


def main():
    R, G = [], []
    for w in POOL:
        a, b = runs_gaps(load(w))
        R += a; G += b
    xmax = max(max(R), max(G))
    Rx, Ry = survival(R, xmax)
    Gx, Gy = survival(G, xmax)
    p_bridge = sum(1 for g in G if g <= BRIDGE) / len(G)
    mR, mG = median(R), median(G)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.axvspan(1, BRIDGE, color=LIGHTBLUE, alpha=0.45, lw=0, zorder=0)
    ax.axvline(BRIDGE, color=NAVY, ls="--", lw=1.3, zorder=1)
    ax.step(Rx, Ry, where="post", color=ORANGE, lw=2.6,
            label=f"repeated-span length |R|  (median {mR})")
    ax.step(Gx, Gy, where="post", color=BLUE, lw=2.6,
            label=f"novel-gap length |G|  (median {mG})")

    ax.annotate(f"P(|G| ≤ {BRIDGE}) ≈ {p_bridge:.2f}\n"
                f"bridgeable with a {BRIDGE}-token model head",
                xy=(BRIDGE, [y for x, y in zip(Gx, Gy) if x == BRIDGE][0]),
                xytext=(BRIDGE + 1.2, 0.60), fontsize=12.5, color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.3))

    ax.set_xscale("log")
    ax.set_xlim(1, xmax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("segment length (tokens, log scale)", fontsize=13)
    ax.set_ylabel("P(length > x)", fontsize=13)
    ax.legend(loc="upper right", frameon=False, fontsize=12.5)
    ax.tick_params(labelsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "Picture3.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")
    print(f"  |R| n={len(R)} median={mR} mean={sum(R)/len(R):.1f}")
    print(f"  |G| n={len(G)} median={mG} mean={sum(G)/len(G):.1f}  "
          f"P(|G|<={BRIDGE})={p_bridge:.2f}")


if __name__ == "__main__":
    main()
