#!/usr/bin/env python3
"""Per-depth ORACLE outcome breakdown — who UNIQUELY matches GT, by depth.

The o4_selection.png "EAGLE3-chosen fraction" is misleading for the oracle arm:
chosen defaults to eagle on both/none/eagle-right, so at deep depths (where
"none" = neither proposer matches GT dominates, the chain is dead) it trends to
~1 regardless of which proposer is actually better. This plots the real outcome
from oracle_hit:

  left  — fractions of ALL decisions at depth d that are eagle-only / suffix-only
          / both / none. The "purely EAGLE3 won" curve (eagle-only) is the user's
          ask; it drops faster than suffix-only -> they cross over.
  right — among DECISIVE decisions (exactly one proposer right, eagle xor suffix),
          eagle-right vs suffix-right fraction by depth (normalized split). suffix
          overtakes with depth, matching suffix's higher deep survival.

Usage (container, root):
  python3 simulation/scripts/plot_o4_oracle_winner.py \
      --dir simulation/results/o4_perdepth/qwen3_14b_ocalib --label EAGLE3
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", default="EAGLE3")
    ap.add_argument("--oracle-log", default="decisions_select1_oracle.jsonl")
    args = ap.parse_args()
    D = Path(args.dir)
    C = defaultdict(lambda: defaultdict(int))
    with open(D / args.oracle_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "decision" or r.get("tail"):
                continue
            d = int(r["depth"])
            C[d]["n"] += 1
            h = r.get("oracle_hit")
            C[d][h] = C[d].get(h, 0) + 1
    depths = sorted(C)
    n = np.array([C[d]["n"] for d in depths], float)
    eag = np.array([C[d].get("eagle", 0) for d in depths], float)
    suf = np.array([C[d].get("suffix", 0) for d in depths], float)
    both = np.array([C[d].get("both", 0) for d in depths], float)
    none = np.array([C[d].get("none", 0) for d in depths], float)
    dec = eag + suf
    lab = args.label

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    # left: fractions of all decisions
    ax1.plot(depths, eag / n, "-o", color="#1f77b4", ms=4,
             label=f"{lab} ONLY won (pure)")
    ax1.plot(depths, suf / n, "-o", color="#ff7f0e", ms=4,
             label="suffix ONLY won (pure)")
    ax1.plot(depths, both / n, "-s", color="#2ca02c", ms=3, alpha=0.7,
             label="both matched")
    ax1.plot(depths, none / n, "-^", color="#999999", ms=3, alpha=0.7,
             label="neither (chain dead)")
    ax1.set_xlabel("depth d"); ax1.set_ylabel("fraction of ALL decisions")
    ax1.set_title(f"Per-depth oracle outcome ({lab} vs suffix)\n"
                  "'neither' dominates deep -> chosen defaults to model "
                  "(why EAGLE3-chosen looked ~1)", fontsize=9)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8); ax1.set_ylim(0, 1.0)

    # right: decisive split (eagle-right vs suffix-right among eagle xor suffix)
    safe = dec > 0
    dd = np.array(depths)[safe]
    ax2.plot(dd, (eag / dec)[safe], "-o", color="#1f77b4", ms=4,
             label=f"{lab} wins | decisive")
    ax2.plot(dd, (suf / dec)[safe], "-o", color="#ff7f0e", ms=4,
             label="suffix wins | decisive")
    ax2.axhline(0.5, color="k", ls=":", lw=0.8)
    ax2.set_xlabel("depth d")
    ax2.set_ylabel("fraction of DECISIVE (exactly one right)")
    ax2.set_title("Among unique-winner decisions: suffix overtakes with depth\n"
                  "(matches suffix's higher deep survival)", fontsize=9)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8); ax2.set_ylim(0, 1.0)
    fig.suptitle(f"O4 oracle per-position winner — {lab} 14B "
                 f"(from oracle_hit, not the eagle-default 'chosen')", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = D / "figures" / "o4_oracle_winner.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")
    print("depth  pureEAGLE%  pureSUFFIX%  both%  none%  | decisive suffix-right%")
    for i, d in enumerate(depths):
        ds = (suf[i] / dec[i] * 100) if dec[i] else float("nan")
        print("  %2d   %6.1f   %6.1f   %5.1f  %5.1f  |  %5.1f"
              % (d, 100*eag[i]/n[i], 100*suf[i]/n[i], 100*both[i]/n[i],
                 100*none[i]/n[i], ds))


if __name__ == "__main__":
    main()
