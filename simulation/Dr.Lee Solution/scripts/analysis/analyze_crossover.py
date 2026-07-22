#!/usr/bin/env python3
"""Slide 9 / §9 — partial-warm crossover analysis from pw.json.

Reads run_partialwarm_tree.py's pw.json and reports, per novel-slot count k:
the accepted-tokens-per-round K for each proposer (dflash / suffix / chain /
tree) + measured coverage, and the crossover point where the composition
(chain/tree) overtakes both standalones. Emits a K-vs-k figure (MAT/K graph —
the standard per the project's no-speedup-graph rule; NO speedup-vs-vanilla, NO
cross-machine comparison).

  python3 scripts/analyze_crossover.py --pw results/partialwarm_tree/pw.json \
      --fig results/partialwarm_tree/figures/crossover_K.png
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw", required=True)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    d = json.load(open(args.pw))
    KS = d["KS"]
    props = list(d["K"].keys())
    cov = d.get("coverage", [None] * len(KS))

    # ---- table ----
    print(f"model={d.get('model')}  proposers={props}")
    hdr = "  k   cov   " + "  ".join(f"{p:>8}" for p in props)
    print(hdr)
    print("-" * len(hdr))
    for i, k in enumerate(KS):
        cv = f"{cov[i]:.2f}" if cov[i] is not None else "  ? "
        row = f"{k:>3}  {cv:>4}   " + "  ".join(f"{d['K'][p][i]:>8.2f}" for p in props)
        print(row)

    # ---- crossover: first k where a composition beats BOTH standalones ----
    singles = [p for p in ("dflash", "suffix") if p in d["K"]]
    comps = [p for p in ("chain", "tree") if p in d["K"]]
    if singles and comps:
        print("\ncrossover (composition > best standalone):")
        for i, k in enumerate(KS):
            best_single = max(d["K"][p][i] for p in singles)
            for c in comps:
                Kc = d["K"][c][i]
                gap = (Kc / best_single - 1.0) * 100 if best_single else float("inf")
                flag = "  <== overtakes" if Kc > best_single else ""
                print(f"  k={k}: {c} K={Kc:.2f} vs best-standalone K={best_single:.2f} "
                      f"({gap:+.0f}%){flag}")

    if not args.fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)

    style = {"dflash": ("#1f77b4", "o", "DFlash (Predictor)"),
             "suffix": ("#ff7f0e", "s", "Suffix (Memorizer)"),
             "chain":  ("#2ca02c", "^", "Chain (gated)"),
             "tree":   ("#d62728", "D", "Tree (ungated)")}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for p in props:
        c, m, lab = style.get(p, ("#555", "x", p))
        ax.plot(KS, d["K"][p], marker=m, color=c, label=lab, lw=2, ms=6)
    ax.set_xlabel("novel slots  k  (partial-warm; larger = more head breaks)")
    ax.set_ylabel("K  (accepted draft tokens / round)")
    ax.set_title(f"Partial-warm crossover — {d.get('model')}")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(KS, cov, ls="--", color="#888", lw=1, marker=".", label="coverage")
    ax2.set_ylabel("coverage (eval n-grams in warm)", color="#888")
    ax2.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=140)
    print(f"\nsaved figure -> {args.fig}")


if __name__ == "__main__":
    main()
