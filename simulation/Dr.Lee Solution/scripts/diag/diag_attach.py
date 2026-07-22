#!/usr/bin/env python3
"""Diagnostic: did the controller attach the Suffix tail at the right depth?
Per composition round (chain trajectory), compare the chosen head length k* to
H = the DFlash head's correct-run length (leading gt-matches). Categorize:
  ① too_shallow  (k* < H): switched to suffix before DFlash broke -> DFlash under-used
  ② right        (k* == H): attached exactly at DFlash's break point
  ③ dead_tail    (k* > H): head broke before k* -> gated tail never reached (died)
CPU only. Emits fractions per group + a grouped bar chart.
  python3 scripts/diag_attach.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
from fusion_tree import build_extension_chain, adaptive_nhead  # noqa: E402
from measure_k_fusion import ArcticSuffix  # noqa: E402
from simulation.evaluation.tree_knapsack import greedy_tree_walk_path  # noqa: E402

BASE = Path("/workspace/simulation/Dr.Lee Solution")
GROUPS = {
    "multislot": [BASE / f"results/perpos_inf/multislot_k{k}.jsonl" for k in (0, 1, 2, 4, 8)],
    "specbench": [BASE / "results/perpos/specbench.jsonl"],
}
CATS = ["no_suffix", "too_shallow", "right", "dead_tail"]


def head_run(match):
    h = 0
    for m in match:
        if m == 1:
            h += 1
        else:
            break
    return h


def diag_record(record_path):
    """Run the chain trajectory over one record file; return per-round categories."""
    traces = json.load(open(Path(record_path).with_suffix(".traces.json")))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)
    recs = defaultdict(dict)
    for l in open(record_path):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r

    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    cats = []
    for rid, rby in recs.items():
        tr = eval_traces.get(rid)
        if not tr:
            continue
        gt, prompt_ids = tr["output_ids"], tr["prompt_ids"]
        W = next(iter(rby.values()))["W"]
        suffix.new_eval(prompt_ids)
        m = 0
        for _ in range(2000):
            rec = rby.get(m + 1)
            if rec is None or m >= len(gt):
                break
            block_full = rec["dflash_tok"]; conf = rec["dflash_conf"]
            H = head_run(rec["dflash_match"])
            root = gt[m]
            ctx = prompt_ids + gt[:m] + [root]
            _, T = suffix.probe(ctx, num_spec)
            k = adaptive_nhead(conf, T=T, num_spec=W)
            # categorize this round's attach position (no_suffix = controller maxed
            # the head, declining a tail; the 3-way is over tail-intended rounds).
            if k >= W:
                cats.append("no_suffix")
            elif k < H:
                cats.append("too_shallow")
            elif k == H:
                cats.append("right")
            else:
                cats.append("dead_tail")
            # advance the chain trajectory (faithful to replay_extension)
            tail = suffix.speculate(ctx + block_full[:k], num_spec)
            tree = build_extension_chain(block_full[:k], tail[:max(0, num_spec - k)])
            path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt[m + 1:])
            acc = len(path)
            nxt = [root] + [tree.tokens[i] for i in path]
            bi = m + 1 + acc
            if bi < len(gt):
                nxt.append(gt[bi])
            suffix.add_response(nxt)
            m += 1 + acc + 1
    return cats


def main():
    frac = {}
    for grp, files in GROUPS.items():
        allc = []
        for f in files:
            if f.exists():
                allc += diag_record(f)
        n = len(allc) or 1
        counts = {c: allc.count(c) for c in CATS}
        frac[grp] = {c: counts[c] / n for c in CATS}
        print(f"[{grp}] rounds={len(allc)}  " +
              "  ".join(f"{c}={counts[c]}({frac[grp][c]*100:.0f}%)" for c in CATS))

    # grouped bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    groups = list(frac)
    x = np.arange(len(groups)); w = 0.26
    COL = {"no_suffix": "#4C78A8", "too_shallow": "#F58518",
           "right": "#54A24B", "dead_tail": "#E45756"}
    LAB = {"no_suffix": "⓪ no suffix\n(k*=W, pure DFlash)",
           "too_shallow": "① too shallow\n(k*<H, DFlash under-used)",
           "right": "② right\n(k*=H)",
           "dead_tail": "③ dead tail\n(k*>H, gated tail died)"}
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    w = 0.2
    for i, c in enumerate(CATS):
        vals = [frac[g][c] * 100 for g in groups]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=LAB[c], color=COL[c])
        for xi, v in zip(x + (i - 1.5) * w, vals):
            ax.text(xi, v + 0.6, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("% of composition rounds")
    ax.set_title("Suffix-tail attach position vs DFlash break point (H)\n"
                 "controller k* : too shallow / right / dead (Qwen3.5-27B, argmax)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(0, max(frac[g][c] * 100 for g in groups for c in CATS) * 1.2)
    fig.tight_layout()
    out = BASE / "readable_outputs" / "figures" / "attach_diagnosis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
