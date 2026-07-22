#!/usr/bin/env python3
"""Extract the ORDERED calibration-target pair streams for the bias/variance study.

Two raw signals the online calibrators correct:
  HEAD  DFlash conf   -> did the block token match gt at that depth (0/1)
        => the conditional accept RATE as a function of DFlash prob.
  TAIL  arctic suffix score -> realized suffix-tail accept LENGTH (int)
        => accept length as a function of Suffix score.

Both are collected ACCEPT-CONDITIONED (candidate handoffs k<=a_d, the head still
alive so the label is realizable) along the deployed chain policy, in SERVING
ORDER on the TEST half of the three-way convlabel split (same eval subset as the
window ablation). No re-serving: DFlash blocks come from the record, the suffix
side is the model-free warm tree. The tail probe (per candidate k, per round) is
the only real cost; head pairs are free from the record.

Emits results/bias_variance/pairs_{ds}.npz with ordered arrays:
  head_conf, head_match, head_round     (one row per accept-conditioned depth)
  tail_score, tail_acc, tail_round      (one row per candidate handoff k)
`*_round` is a global monotonically-increasing round index (serving order) so the
window simulation can reproduce the online sliding window / refit cadence.

  python3 scripts/analysis/extract_bias_variance_pairs.py --workload tau2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from replay_extension import (ArcticSuffix, adaptive_nhead, build_extension_chain,  # noqa: E402
                              greedy_tree_walk_path, split_parity, _ad)

REC = {
    "specbench": "results/perpos_specbench_alleval/specbench_4way.jsonl",
    "bfcl": "results/perpos_bfcl_alleval/bfcl_4way.jsonl",
    "swebench": "results/perpos_swebench_alleval/swebench_4way.jsonl",
    "spider": "results/perpos_spider_alleval/spider_4way.jsonl",
    "tau2": "results/perpos_tau2_alleval/tau2_4way.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=list(REC))
    ap.add_argument("--max-rounds", type=int, default=4096)
    ap.add_argument("--out-dir", default="results/bias_variance")
    args = ap.parse_args()

    rec_path = Path(REC[args.workload])
    traces = json.load(open(rec_path.with_suffix(".traces.json")))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)

    recs = defaultdict(dict)
    for l in open(rec_path):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r

    # SAME eval subset as the window ablation: three-way convlabel, TEST half.
    _, split_par = split_parity(eval_traces, "convlabel")
    test_rids = [rid for rid in recs if split_par.get(rid, 0) == 1]  # recs insertion order

    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    hc, hm, hr = [], [], []          # head: conf, match, round
    ts, ta, tr = [], [], []          # tail: score, acc, round
    round_idx = 0
    for rid in test_rids:
        tr_ = eval_traces.get(rid)
        if tr_ is None:
            continue
        gt, pids, rby = tr_["output_ids"], tr_["prompt_ids"], recs[rid]
        suffix.new_eval(pids)
        m = 0
        for _ in range(args.max_rounds):
            rec = rby.get(m + 1)
            if rec is None or m >= len(gt):
                break
            block_full = rec["dflash_tok"]; cf = rec["dflash_conf"]; mt = rec["dflash_match"]
            W = rec["W"]
            ad = _ad(mt)
            root = gt[m]
            ctx_list = pids + gt[:m] + [root]
            # HEAD pairs: accept-conditioned depths 0..a_d (incl. the first miss)
            for d in range(min(ad + 1, len(cf))):
                hc.append(float(cf[d])); hm.append(int(mt[d])); hr.append(round_idx)
            # TAIL pairs: candidate handoffs k<=a_d (head alive), budget-aware score
            for kk in range(min(ad, W) + 1):
                budget = num_spec - kk
                if budget <= 0:
                    break
                tk, sc = suffix._spec(ctx_list + block_full[:kk], budget)
                r_acc = 0
                for t, g in zip(tk[:budget], gt[m + 1 + kk:]):
                    if t != g:
                        break
                    r_acc += 1
                ts.append(float(sc)); ta.append(float(r_acc)); tr.append(round_idx)
            # advance along the deployed chain policy (keeps the trajectory realistic)
            suf, T = suffix.probe(ctx_list, num_spec)
            k = adaptive_nhead(cf, T=T, num_spec=W)
            tk = suffix.speculate(ctx_list + block_full[:k], num_spec)
            trd = build_extension_chain(block_full[:k], tk[:max(0, num_spec - k)])
            pth = greedy_tree_walk_path(list(trd.tokens), list(trd.parents), gt[m + 1:])
            accepted = [trd.tokens[i] for i in pth]
            nxt = [root] + accepted
            if m + 1 + len(pth) < len(gt):
                nxt.append(gt[m + 1 + len(pth)])
            suffix.add_response(nxt)
            m += 1 + len(pth) + 1
            round_idx += 1

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"pairs_{args.workload}.npz"
    np.savez(fp,
             head_conf=np.asarray(hc, np.float32), head_match=np.asarray(hm, np.int8),
             head_round=np.asarray(hr, np.int32),
             tail_score=np.asarray(ts, np.float32), tail_acc=np.asarray(ta, np.float32),
             tail_round=np.asarray(tr, np.int32))
    print(f"{args.workload}: {len(hc)} head pairs, {len(ts)} tail pairs, "
          f"{round_idx} rounds -> {fp}")


if __name__ == "__main__":
    main()
