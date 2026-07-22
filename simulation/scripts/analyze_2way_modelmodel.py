"""2-WAY MODEL-vs-MODEL selection ladder (block-anchored, gt-path).

Question: per-depth calibration failed to beat raw for the MODEL-vs-SUFFIX pairs
(EAGLE3 vs suffix, MTP vs suffix) -- the two scores lived in incomparable regimes
and the joint was information-capped. Does calibration behave better between two
MODEL-based proposers, whose probs come from comparable softmax heads?

  8b : DFlash(main, served as eagle_*) vs EAGLE3(aux, e3_*)
  27b: MTP(main, eagle_*) vs DFlash(dflash_*, merged offline forward)

Same unified select-1 rule as analyze_3way_ladder: each proposer estimates
P(proposer == gt); select-1 = argmax. With 2 proposers, `max_other` is simply the
other proposer's prob, i.e. the classic 2-way boundary. All held-out via
GroupKFold-by-rid OOF. Reuses the 3-way machinery so the two studies are identical
apart from the proposer set.

  python3 analyze_2way_modelmodel.py --cell 8b  --record-dir <dir> --exclude-loopy [--fig]
  python3 analyze_2way_modelmodel.py --cell 27b --record-dir <dir> --merge-dflash <dflash_proposals.jsonl> [--fig]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_3way_ladder import (  # noqa: E402
    CELLS, CALIBS, load_chains, loopy_rids, merge_dflash, row_info,
    collect_samples, oof_estimates, selacc_mat, pick_raw, pick_est, pick_oracle,
)

# the two MODEL-based proposers per cell (drop suffix)
MODEL_PAIRS = {"8b": ("dflash", "e3"), "27b": ("mtp", "dflash")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["8b", "27b"], required=True)
    ap.add_argument("--record-dir", required=True)
    ap.add_argument("--decisions-file", default="decisions_select1_oracle.jsonl")
    ap.add_argument("--merge-dflash", default=None)
    ap.add_argument("--proposers", default=None,
                    help="comma pair overriding the default model-vs-model pair")
    ap.add_argument("--exclude-loopy", action="store_true")
    ap.add_argument("--fig", action="store_true")
    args = ap.parse_args()

    c = CELLS[args.cell]
    names = tuple(args.proposers.split(",")) if args.proposers else MODEL_PAIRS[args.cell]
    assert len(names) == 2, "this study is 2-way"
    for P in names:
        assert P in c["names"], f"{P} not a proposer of cell {args.cell} ({c['names']})"
    tok = {P: c["tok"][P] for P in names}
    pk = {P: c["p"][P] for P in names}
    a, b = names

    chains = load_chains(Path(args.record_dir) / args.decisions_file)
    if args.exclude_loopy:
        bad = loopy_rids(args.record_dir, args.decisions_file)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        print(f"excluded {len(bad)} loopy reqs")
    if args.merge_dflash:
        # merge_dflash writes dflash_token/dflash_p onto matched rows regardless of cell key
        merge_dflash(chains, args.merge_dflash)

    # keep only blocks where BOTH model proposers are present somewhere in the block
    def has_both(rs):
        return any(r.get(tok[a]) is not None for r in rs) and \
               any(r.get(tok[b]) is not None for r in rs)
    chains = {k: v for k, v in chains.items() if has_both(v)}

    for rs in chains.values():
        for r in rs:
            r["_ri"] = row_info(r, names, tok, pk)
    print(f"cell={args.cell}  MODEL-vs-MODEL  proposers={names}  blocks={len(chains)}")

    S = collect_samples(chains, names, tok, pk)
    for P in names:
        y = S[P]["y"]
        print(f"  decisive samples[{P}] = {len(y)} (hit-rate "
              f"{np.mean(y) if y else float('nan'):.3f})")
    methods = ("raw",) + CALIBS + ("mono", "bayes")
    for m in methods:
        oof_estimates(S, names, m)

    rows = []

    def add(lab, fn, sub):
        sa, mt = selacc_mat(chains, fn, names, tok, pk, sub)
        rows.append((lab, sa, mt))
        print(f"{lab:26s} selacc={sa:.4f}  MAT={mt:.4f}")

    print()
    add(f"{a}-only", pick_raw, (a,))
    add(f"{b}-only", pick_raw, (b,))
    print("  --- 2-way (model vs model) ---")
    add("2way raw", pick_raw, names)
    for m in CALIBS:
        add(f"2way calib:{m}", pick_est(m), names)
    add("2way mono(ceiling)", pick_est("mono"), names)
    add("2way bayes", pick_est("bayes"), names)
    add("2way oracle", pick_oracle(names), names)

    # gap-recovery summary vs raw->oracle
    d = {lab: (sa, mt) for lab, sa, mt in rows}
    raw_sa = d["2way raw"][0]
    orc_sa = d["2way oracle"][0]
    print(f"\n  raw->oracle selacc gap = {orc_sa - raw_sa:+.4f}")
    if orc_sa > raw_sa:
        print("  gap recovery (selacc):")
        for lab in [f"2way calib:{m}" for m in CALIBS] + ["2way mono(ceiling)", "2way bayes"]:
            v = d[lab][0]
            print(f"    {lab:22s} {100 * (v - raw_sa) / (orc_sa - raw_sa):+6.1f}%")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        from _ladder_style import ladder_bar
        figdir = Path(args.record_dir) / "figures"
        figdir.mkdir(exist_ok=True)
        labs = [r[0] for r in rows]
        C = {"raw": "#7f7f7f", "calib": "#e377c2", "mono": "darkorange",
             "bayes": "lime", "oracle": "#d62728"}
        cols = []
        for l in labs:
            if "oracle" in l: cols.append(C["oracle"])
            elif "calib" in l: cols.append(C["calib"])
            elif "mono" in l: cols.append(C["mono"])
            elif "bayes" in l: cols.append(C["bayes"])
            elif "only" in l: cols.append("#1f77b4")
            else: cols.append(C["raw"])
        nm = "+".join(names)
        ladder_bar([r[1] for r in rows], labs, "decisive selection accuracy (alive-cond)",
                   f"2-way model-vs-model ({nm}, {c['model']}, gt-path, held-out OOF)",
                   f"{figdir}/ladder_selacc_2wayMM_{args.cell}.png", fmt="{:.3f}", colors=cols)
        ladder_bar([r[2] for r in rows], labs, "MAT (block-anchored)",
                   f"2-way model-vs-model MAT ({nm}, {c['model']}, gt-path, held-out OOF)",
                   f"{figdir}/ladder_mat_2wayMM_{args.cell}.png", fmt="{:.3f}", colors=cols)
        print(f"\nfigures -> {figdir}/ladder_{{selacc,mat}}_2wayMM_{args.cell}.png")


if __name__ == "__main__":
    main()
