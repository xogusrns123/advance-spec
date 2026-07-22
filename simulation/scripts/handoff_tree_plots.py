#!/usr/bin/env python3
"""TREE-structure hand-off curves (both DFlash and suffix are trees).

Same experiment as handoff_uncapped_plots.py (fixed-threshold a* sweep -> curve,
score-based variable threshold -> point; MAT + verify-held-constant speedup; cold
& warm), but BOTH proposers use their TREE structure instead of a linear chain:

  * DFlash TREE  = DDTree top-k per block depth. A_d_tree = first depth where the
    gt token's rank in the depth distribution is >= k_tree (gt not among the
    top-k candidates). k_tree=1 recovers the linear chain accept. Requires the
    per-depth gt_rank emitted by dflash_offline.py (dflash_proposals_tree.jsonl).
  * suffix TREE  = the suffix draft's own tree-walk accept = the `orc` column
    (col 1) of the dense table, vs the linear `grd` column (col 2) used before.
    Needs the table built with --full-handoff-depths so orc[j] exists for every
    hand-off depth j up to the full block (A_d_tree can exceed the chain accept).

Everything downstream (val gate, threshold policy, renewal walk, latency model
with verify HELD CONSTANT) is reused verbatim from handoff_uncapped_plots by
building the same per-position dict with ae:=A_d_tree and asv:=orc.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.scripts.handoff_uncapped_plots import compute, plot  # noqa: E402


def load_haz_tree(path):
    """(rid, decode_step) -> list of (dflash_p, gt_rank) ordered by depth."""
    tmp = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            tmp[(r["rid"], r["decode_step"])][r["depth"]] = (
                r["dflash_p"], r["gt_rank"])
    return {k: [dm[d] for d in sorted(dm)] for k, dm in tmp.items()}


def ad_tree(ranks, k_tree):
    """A_d_tree = leading depths whose gt token is within the top-k_tree."""
    m = 0
    for _, gr in ranks:
        if gr < k_tree:
            m += 1
        else:
            break
    return m


def load_seqs_tree(table_path, tree_haz, k_tree):
    """Build the per-position dict handoff_uncapped_plots.compute expects, but
    with ae := A_d_tree (DFlash tree) and asv := orc (suffix tree)."""
    seqs = defaultdict(dict)
    n = nohaz = 0
    with gzip.open(table_path, "rt") as f:
        for line in f:
            r = json.loads(line)
            sfx = r["sfx"]
            asv = {s[0]: s[1] for s in sfx}          # col 1 = orc (tree accept)
            gclj = {s[0]: s[4] for s in sfx}
            gcl0 = sfx[0][4] if sfx else 0
            score0 = sfx[0][6] if sfx else 0.0
            ranks = tree_haz.get((r["rid"], r["pos"]), [])
            if not ranks:
                nohaz += 1
            haz = [p for p, _ in ranks]
            seqs[(r["rid"], r["ci"])][r["pos"]] = dict(
                ae=ad_tree(ranks, k_tree), asv=asv, gclj=gclj, gcl0=gcl0,
                score0=score0, haz=haz)
            n += 1
    print(f"positions={n} no_tree_haz={nohaz}", file=sys.stderr)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="dense table (built with --full-handoff-depths)")
    ap.add_argument("--dflash-tree", required=True, help="dflash_proposals_tree.jsonl (has gt_rank)")
    ap.add_argument("--k-tree", type=int, default=4, help="DFlash tree top-k per depth")
    ap.add_argument("--regime", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    tree_haz = load_haz_tree(args.dflash_tree)
    seqs = load_seqs_tree(args.table, tree_haz, args.k_tree)
    res = compute(seqs)

    if args.k_tree == 1:
        # top-1 == the linear DFlash chain; only the suffix is a tree
        head = "DFlash-chain"
        struct = "Chain-head + suffix-tree hand-off"
        tag = "dchain_sfxtree"
    else:
        head = f"DFlash-tree(top-{args.k_tree})"
        struct = "Tree hand-off"
        tag = f"tree_k{args.k_tree}"
    model = f"Qwen3.5-27B  ({head} head + suffix-tree tail)"
    print(f"\n=== {args.regime} ({head} + suffix-tree, k_tree={args.k_tree}) ===")
    for k in ("dflash", "suffix", "o3", "score"):
        r = res[k]
        extra = f" a*_med={r['astar_med']:.3f}" if "astar_med" in r else ""
        print(f"  {k:8s}: MAT={r['mat']:.3f} speedup={r['speedup']:.2f}x{extra}")
    best = max(res["curve"], key=lambda c: c[1])
    print(f"  best fixed a*={best[0]:.2f}: MAT={best[1]:.3f} speedup={best[2]:.2f}x")

    plot(res, args.regime, model, args.outdir, tag=tag, struct_label=struct,
         curve_label=f"{head} head + suffix-tree tail")
    if args.out_json:
        json.dump(res, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
