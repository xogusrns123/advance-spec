#!/usr/bin/env python3
"""Raw, fit-free measured relationship: DFlash head raw prob -> accept length.

For every drafted round we take:
  x = raw prob   = dflash_conf[0]  (softmax max-prob of the first drafted token)
  y = accept len = leading run of 1s in dflash_match (realized accepted tokens)

pooled over all 5 workloads. NO fit of any kind is drawn — only the measured
data: a hexbin density of the raw (prob, accept-length) pairs, plus binned
EMPIRICAL means (markers ± 95% CI, which are measurements, not a fitted curve).

  docker exec sglang-bench bash -lc \
    "cd '/workspace/simulation/Dr.Lee Solution' && python3 scripts/plot/regen_raw_prob_accept.py"
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent
RES = BASE / "results"
OUT = BASE / "readable_outputs" / "figures" / "0710_regenerated"
NAVY, ORANGE = "#1F2A44", "#D85A30"

RECS = {  # display -> per-round record (carries dflash_conf / dflash_match)
    "AgenticSQL": "perpos_spider_alleval/spider_4way.jsonl",
    "SWE-Bench":  "perpos_swebench_alleval/swebench_4way.jsonl",
    "τ²-bench":   "perpos_tau2_alleval/tau2_4way.jsonl",
    "BFCLv4":     "perpos_bfcl_full/bfcl_v4_full.jsonl",
    "Spec-Bench": "perpos_specbench_full/specbench.jsonl",
}
NBINS = 20                       # prob bins on [0, 1]


def accept_len(match):
    n = 0
    for m in match:
        if m:
            n += 1
        else:
            break
    return n


def main():
    xs, ys = [], []
    for wl, rel in RECS.items():
        p = RES / rel
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                conf = r.get("dflash_conf")
                match = r.get("dflash_match")
                if not conf or match is None:
                    continue
                xs.append(float(conf[0]))
                ys.append(accept_len(match))

    n = len(xs)
    yhi = sorted(ys)[int(0.99 * n)]              # clip y-view to the 99th pct

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    hb = ax.hexbin(xs, ys, gridsize=(46, 26), bins="log", cmap="Blues",
                   mincnt=1, extent=(0, 1, 0, yhi + 1))
    cb = fig.colorbar(hb, ax=ax, pad=0.015)
    cb.set_label("rounds per cell (log)", fontsize=11)

    # binned EMPIRICAL means (measurement, not a fit) — markers + 95% CI, no line
    edges = [i / NBINS for i in range(NBINS + 1)]
    cx, cy, ce = [], [], []
    for b in range(NBINS):
        lo, hi = edges[b], edges[b + 1]
        vals = [y for x, y in zip(xs, ys) if lo <= x < hi]
        if len(vals) < 30:
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        cx.append((lo + hi) / 2)
        cy.append(m)
        ce.append(1.96 * (var / len(vals)) ** 0.5)
    ax.errorbar(cx, cy, yerr=ce, fmt="o", ms=7, color=ORANGE, ecolor=ORANGE,
                elinewidth=1.5, capsize=3, mec="white", mew=1.0, ls="none",
                zorder=5, label="binned empirical mean ± 95% CI (measured)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, yhi + 1)
    ax.set_xlabel("DFlash head raw prob  (softmax max-prob of first drafted token)",
                  fontsize=12)
    ax.set_ylabel("realized accept length  (tokens)", fontsize=12)
    ax.set_title(f"Raw prob → accept length (measured, no fit)   n={n:,} rounds",
                 fontsize=13, color=NAVY)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.tick_params(labelsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "raw_prob_accept_length.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"wrote {fp}  n={n} y99={yhi}")
    for x, y in zip(cx, cy):
        print(f"  prob~{x:.2f}  mean accept={y:.2f}")


if __name__ == "__main__":
    main()
