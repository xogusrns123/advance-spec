#!/usr/bin/env python3
"""Effect of WARM-SET warming on the suffix proposer. Always tested on the eval set.

  warming ON  : fill the suffix GLOBAL tree with the warm set, then evaluate eval.
  warming OFF : evaluate eval directly (global tree starts empty; it still grows
                within each eval trajectory via add_response, so only the CROSS-
                trajectory warm memorization is removed).

Suffix-only (no dflash / no calib) so it is light and does not touch the compose
pipeline. Reports mean accepted tokens/round (suffix MAT) per workload, on vs off,
and writes an independent bar figure.

  # original (pre-1:1) split — read warm from the .orig traces:
  PYTHONPATH=/workspace python3 scripts/warming_effect.py --split orig --out-tag orig
  # 1:1 split — read warm from the current traces:
  PYTHONPATH=/workspace python3 scripts/warming_effect.py --split cur  --out-tag 1to1
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import argparse, json, os
from pathlib import Path

import sys
sys.path.insert(0, "scripts")
from replay_extension import ArcticSuffix

WL = {"specbench": "results/perpos_specbench_alleval/specbench_4way",
      "bfcl": "results/perpos_bfcl_alleval/bfcl_4way",
      "swebench": "results/perpos_swebench_alleval/swebench_4way",
      "spider": "results/perpos_spider_alleval/spider_4way",
      "tau2": "results/perpos_tau2_alleval/tau2_4way"}
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench\nVerified",
           "spider": "Spider2-DBT", "tau2": "τ²-bench"}
MAXR = 4096


def suffix_mat(traces, warm_on, num_spec):
    warm = traces["warm_traces"] if warm_on else []
    suffix = ArcticSuffix(); suffix.fit(warm)
    tot, rounds = 0, 0
    for t in traces["eval_traces"]:
        gt, pids = t["output_ids"], t["prompt_ids"]
        suffix.new_eval(pids)
        m, r = 0, 0
        while m < len(gt) and r < MAXR:
            root = gt[m]
            ctx = pids + gt[:m] + [root]
            tk = suffix.speculate(ctx, num_spec)
            acc = 0
            for x, g in zip(tk, gt[m + 1:]):
                if x == g:
                    acc += 1
                else:
                    break
            tot += acc; rounds += 1
            nxt = [root] + gt[m + 1:m + 1 + acc]
            if m + 1 + acc < len(gt):
                nxt.append(gt[m + 1 + acc])
            suffix.add_response(nxt)
            m += 1 + acc + 1; r += 1
    return tot / rounds if rounds else 0.0, len(traces["warm_traces"]) if warm_on else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["orig", "cur"], default="cur")
    ap.add_argument("--out-tag", default="cur")
    args = ap.parse_args()
    rows = []
    for wl, stem in WL.items():
        tp = stem + ".traces.json"
        if args.split == "orig" and os.path.exists(tp + ".orig"):
            tp = tp + ".orig"
        traces = json.load(open(tp))
        ns = traces.get("num_spec", 32)
        on, nw = suffix_mat(traces, True, ns)
        off, _ = suffix_mat(traces, False, ns)
        rows.append(dict(wl=wl, on=on, off=off, warm=nw, eval=len(traces["eval_traces"])))
        print(f"{wl:<10} warm={nw:<5} eval={len(traces['eval_traces']):<5} "
              f"suffix MAT  on={on:.2f}  off={off:.2f}  gain={on - off:+.2f}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    OUT = Path("readable_outputs/figures/mat/warming"); OUT.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rows)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    b1 = ax.bar(x - w / 2, [r["off"] for r in rows], w, label="warming OFF (empty global tree)",
                color="#bdbdbd", edgecolor="k", linewidth=0.5, zorder=3)
    b2 = ax.bar(x + w / 2, [r["on"] for r in rows], w, label="warming ON (warm-set global tree)",
                color="#F58518", edgecolor="k", linewidth=0.5, zorder=3)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for i, r in enumerate(rows):
        ax.annotate(f"+{r['on'] - r['off']:.2f}", (i, max(r['on'], r['off']) + 0.16),
                    ha="center", fontsize=10, fontweight="bold", color="#1a7d1a")
    ax.set_xticks(x); ax.set_xticklabels([WL_NAME[r["wl"]] for r in rows], fontsize=10)
    ax.set_ylabel("suffix MAT  (mean accepted tokens / round)", fontsize=11)
    split = "original split" if args.split == "orig" else "1:1 split"
    ax.set_title(f"Warming effect on the suffix proposer — global tree warmed by the warm set\n"
                 f"(tested on eval set; {split}; warm:eval per bar-group)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_ylim(0, max(max(r["on"], r["off"]) for r in rows) * 1.2)
    fig.tight_layout()
    fp = OUT / f"WARMING_effect_{args.out_tag}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    json.dump(rows, open(OUT / f"warming_{args.out_tag}.json", "w"), indent=1)
    print("saved ->", fp)


if __name__ == "__main__":
    main()
