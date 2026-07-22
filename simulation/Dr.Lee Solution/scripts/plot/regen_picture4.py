#!/usr/bin/env python3
"""Regenerate synthetic Picture4 with REAL data — same 3-line survival format.

  S_i = P(first i draft tokens accepted), three proposers:
    DFlash (model)                 : accept-length a(p) over all positions
    suffix — warm (inside copy run): suffix accept-length s(p) | p in warm region
    suffix — cold (no match)       : suffix accept-length s(p) | p in cold region

warm/cold = the plot_traj_warmcold region decomposition applied to the s(p)
copy-length series (run_cover run-len>=4, close gaps<=12, drop islands<16),
pooled over the agentic workloads. Signals from interp_validation curves.

  PYTHONPATH=/workspace python3 scripts/plot/regen_picture4.py
"""
from __future__ import annotations
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from plot_traj_warmcold import run_cover, segments_from_hits  # noqa: E402

DATA = BASE / "results" / "interp_validation"
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"
ORANGE, BLUE, GRAY = "#D85A30", "#2A78D6", "#9A988F"
POOL = ["swebench", "spider", "bfcl", "tau2"]   # agentic workloads only
RUN_LEN, GAP, MIN_SEG, MAX_DEPTH = 4, 12, 16, 15


def load(w):
    with gzip.open(DATA / f"curves_{w}.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f]


def collect():
    dfl, sw, sc = [], [], []
    for w in POOL:
        for r in load(w):
            s, a = r["s"], r["a"]
            L = len(s)
            if L < 2:
                continue
            warm = set()
            for b, e in segments_from_hits(run_cover(s, RUN_LEN), GAP, MIN_SEG):
                warm.update(range(b, e))
            for p in range(L):
                if a[p] >= 0:
                    dfl.append(a[p])
                (sw if p in warm else sc).append(s[p])
    return {"dfl": dfl, "warm": sw, "cold": sc}


def survival(lengths):
    n = len(lengths)
    return [sum(1 for v in lengths if v >= i) / n for i in range(1, MAX_DEPTH + 1)]


def main():
    data = collect()
    depths = list(range(1, MAX_DEPTH + 1))
    series = [
        ("DFlash (model)", "dfl", BLUE, "o", "-"),
        ("suffix — warm (inside copy run)", "warm", ORANGE, "s", "-"),
        ("suffix — cold (no match)", "cold", GRAY, "^", "-"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for name, key, col, mk, ls in series:
        ax.plot(depths, survival(data[key]), ls=ls, marker=mk, ms=6, lw=2.2,
                color=col, label=name)
    ax.set_xlabel("draft depth $i$", fontsize=13)
    ax.set_ylabel("survival  $S_i$ = P(first $i$ accepted)", fontsize=12.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0.6, MAX_DEPTH + 0.4)
    ax.set_xticks(range(1, MAX_DEPTH + 1, 2))
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    ax.tick_params(labelsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "Picture4.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}")
    for name, key, *_ in series:
        S = survival(data[key])
        print(f"  {name:<34} n={len(data[key]):>7}  S1={S[0]:.2f} S8={S[7]:.2f} S15={S[14]:.2f}")


if __name__ == "__main__":
    main()
