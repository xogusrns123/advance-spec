#!/usr/bin/env python3
"""SuffixDecoding-paper HYBRID baseline (Arctic Inference style), replayed under
the same protocol as the 4-arm figures: per ROUND, probe the suffix tree at the
current position; if its score >= tau propose the suffix draft (budget
num_spec), otherwise FALL BACK to the model drafter — the DFlash block chain
(the paper falls back to EAGLE; DFlash is our model-side drafter). Binary
round-level switch: no head+tail grafting, no handoff point — this is the
"select whole round" ancestor of our compose.

Same node budget (chain, num_spec), same trajectory advance (root + accepted +
bonus), same warm-set-only tree, teacher-forced greedy accept. tau is swept
(the arctic score is an UNCALIBRATED expected-accept estimate that overshoots
realized accept 3-6x, so the useful range is wide).

  python3 scripts/replay_fallback_sweep.py \
      --records specbench=results/perpos_specbench_full/specbench.jsonl ... \
      --taus 0.5 1 2 4 8 16 32 --out results/pipeline_4way/fallback_sweep.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
from measure_k_fusion import ArcticSuffix  # noqa: E402
from fusion_tree import build_extension_chain  # noqa: E402
from replay_extension import split_parity  # noqa: E402
from simulation.evaluation.tree_knapsack import greedy_tree_walk_path  # noqa: E402


def load_record(record):
    traces = json.load(open(Path(record).with_suffix(".traces.json")))
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    recs = defaultdict(dict)
    for l in open(record):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r
    return traces["warm_traces"], eval_traces, recs, traces.get("num_spec", 32)


def replay_fallback(warm_traces, eval_traces, recs, num_spec, tau, max_rounds=4096,
                    only_rids=None):
    """-> (Ks, per_task Ks, suffix_round_share). only_rids: replay just these
    requests (three-way split half); skipped requests do NOT warm the tree,
    matching replay_extension's --three-way semantics."""
    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    Ks, per_task, n_suf, n_rounds = [], defaultdict(list), 0, 0
    for rid, rby in recs.items():
        if only_rids is not None and rid not in only_rids:
            continue
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        gt, pids, task = tr["output_ids"], tr["prompt_ids"], tr.get("task", "all")
        suffix.new_eval(pids)
        m = 0
        for _ in range(max_rounds):
            rec = rby.get(m + 1)
            if rec is None or m >= len(gt):
                break
            root = gt[m]
            ctx_list = pids + gt[:m] + [root]
            suf, T = suffix.probe(ctx_list, num_spec)
            if T >= tau:                          # suffix draft
                tree = build_extension_chain([], suf[:num_spec])
                n_suf += 1
            else:                                 # fallback: DFlash block chain
                tree = build_extension_chain(rec["dflash_tok"][:num_spec], [])
            n_rounds += 1
            path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt[m + 1:])
            acc = len(path)
            Ks.append(acc); per_task[task].append(acc)
            accepted_toks = [tree.tokens[i] for i in path]
            nxt = [root] + accepted_toks
            if m + 1 + acc < len(gt):
                nxt.append(gt[m + 1 + acc])
            suffix.add_response(nxt)
            m += 1 + acc + 1
            if m >= len(gt):
                break
    return Ks, per_task, (n_suf / n_rounds if n_rounds else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True, help="name=record.jsonl")
    ap.add_argument("--taus", nargs="+", type=float,
                    default=[0.5, 1, 2, 4, 8, 16, 32])
    ap.add_argument("--max-rounds", type=int, default=4096)
    ap.add_argument("--three-way", action="store_true",
                    help="deployable split: replay each tau separately on the "
                         "calibrate half (even groups — pick tau* there) and the "
                         "test half (odd groups — report K there). Output entries "
                         "become {'calib': {tau: ...}, 'test': {tau: ...}}.")
    ap.add_argument("--group-mode", default="conv",
                    choices=["rid", "lenreset", "conv", "convlabel"],
                    help="split unit for --three-way (see replay_extension)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = {}
    for spec in args.records:
        name, rec = spec.split("=", 1)
        warm, ev, recs, num_spec = load_record(rec)
        print(f"\n== {name} (num_spec={num_spec})")
        if args.three_way:
            _, par = split_parity(ev, args.group_mode)
            halves = {"calib": {rid for rid in recs if par.get(rid, 0) == 0},
                      "test": {rid for rid in recs if par.get(rid, 0) == 1}}
            print(f"three-way({args.group_mode}): calib={len(halves['calib'])} "
                  f"test={len(halves['test'])} calls")
            out[name] = {"calib": {}, "test": {}}
            for tau in args.taus:
                for half, rids in halves.items():
                    Ks, per_task, share = replay_fallback(
                        warm, ev, recs, num_spec, tau, args.max_rounds,
                        only_rids=rids)
                    mK = sum(Ks) / len(Ks) if Ks else 0.0
                    out[name][half][str(tau)] = dict(
                        K=mK, rounds=len(Ks), suffix_share=share,
                        by_task={t: (sum(k) / len(k), len(k))
                                 for t, k in per_task.items()})
                    print(f"  tau={tau:<5g} [{half:>5}] K={mK:.2f}  "
                          f"(rounds={len(Ks)}, suffix rounds {share:.0%})",
                          flush=True)
            continue
        out[name] = {}
        for tau in args.taus:
            Ks, per_task, share = replay_fallback(warm, ev, recs, num_spec, tau,
                                                  args.max_rounds)
            mK = sum(Ks) / len(Ks) if Ks else 0.0
            out[name][str(tau)] = dict(
                K=mK, rounds=len(Ks), suffix_share=share,
                by_task={t: (sum(k) / len(k), len(k)) for t, k in per_task.items()})
            print(f"  tau={tau:<5g} K={mK:.2f}  (rounds={len(Ks)}, "
                  f"suffix rounds {share:.0%})", flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
