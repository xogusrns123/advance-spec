#!/usr/bin/env python3
"""Extension-sim capture over an ALREADY-CAPTURED bfcl_v4 gt trajectory
(chain_hybrid gt_tokens.jsonl = {input_ids, output_ids}). No greedy generation —
the pinned agentic trajectory is used directly; only the DFlash block proposals
per position are computed (GPU). Same record schema as capture_perpos, so
replay_extension / analyze_perpos work unchanged.

Suffix is warmed on the first --nwarm requests' outputs (train), evaluated on the
next --neval requests (agentic web_search has cross-request tool-call reuse).

  CUDA_VISIBLE_DEVICES=0 python3 scripts/capture_bfcl.py \
    --gt-tokens /workspace/simulation/results/chain_hybrid_perdepth/qwen35_27b_3way_real_full/gt_tokens.jsonl \
    --nwarm 76 --neval 40 --out results/perpos_bfcl/bfcl_v4.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/vendor/ddtree")
sys.path.insert(0, "/workspace/vendor/ddtree/model")
sys.path.insert(0, "/workspace/simulation/scripts/experiments")

from dflash_offline import DFlashOffline  # noqa: E402
from measure_k_fusion import ArcticSuffix  # noqa: E402
from capture_perpos import suffix_sweep  # noqa: E402  (reuse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=os.environ.get("TGT", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--draft", default=os.environ.get("DFT", "z-lab/Qwen3.5-27B-DFlash"))
    ap.add_argument("--gt-tokens", required=True, help="chain_hybrid gt_tokens.jsonl")
    ap.add_argument("--task", default="bfcl_v4")
    ap.add_argument("--nwarm", type=int, default=76)
    ap.add_argument("--neval", type=int, default=40)
    ap.add_argument("--conv-split", action="store_true",
                    help="split by CONVERSATION boundary (prompt-len drop), interleaved "
                         "even->warm/odd->eval; no straddling, disjoint tasks")
    ap.add_argument("--num-spec", type=int, default=int(os.environ.get("NUM_SPEC", "32")))
    ap.add_argument("--min-gt", type=int, default=1, help="skip eval reqs with fewer gt tokens")
    ap.add_argument("--max-prompt", type=int, default=16000,
                    help="skip eval reqs whose prompt exceeds this (OOM guard on 27B feature forward)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.gt_tokens) if l.strip()]
    if args.conv_split:
        # conversation boundary = prompt-length drop (within a multi-turn task the
        # prompt only grows; a drop signals a new conversation). Assign WHOLE
        # conversations: even index -> warm, odd -> eval (disjoint, no straddle).
        il = [len(r["input_ids"]) for r in rows]
        bnds = [0] + [i for i in range(1, len(rows)) if il[i] < il[i - 1] - 200] + [len(rows)]
        convs = list(zip(bnds, bnds[1:]))
        warm_rows, eval_rows = [], []
        for ci, (a, b) in enumerate(convs):
            (warm_rows if ci % 2 == 0 else eval_rows).extend(rows[a:b])
        nw = sum(1 for ci in range(len(convs)) if ci % 2 == 0)
        print(f"[{args.task}] conv-split: {len(convs)} convs -> warm {nw} convs/{len(warm_rows)} calls, "
              f"eval {len(convs)-nw} convs/{len(eval_rows)} calls (interleaved, disjoint)", flush=True)
    else:
        warm_rows = rows[:args.nwarm]
        eval_rows = rows[args.nwarm:args.nwarm + args.neval]
        print(f"[{args.task}] gt reqs={len(rows)}  warm={len(warm_rows)}  eval={len(eval_rows)}", flush=True)
    warm_traces = [r["output_ids"] for r in warm_rows if r.get("output_ids")]

    dfo = DFlashOffline(args.target, args.draft)
    suf_warm = ArcticSuffix(); suf_warm.fit(warm_traces)
    suf_cold = ArcticSuffix()

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    traces_path = outp.with_suffix(".traces.json")
    eval_traces, n_rec = [], 0
    W = dfo.block_size - 1
    with open(outp, "w") as f:
        import torch
        for ri, r in enumerate(eval_rows):
            prompt_ids = list(r.get("input_ids") or [])
            gt = list(r.get("output_ids") or [])
            if len(gt) < args.min_gt or not prompt_ids:
                continue
            if len(prompt_ids) > args.max_prompt:
                print(f"  [skip] eval {ri}: prompt {len(prompt_ids)} > {args.max_prompt} (OOM guard)", flush=True)
                continue
            seq = prompt_ids + gt
            try:
                recs = dfo.proposals_for_seq(seq, len(prompt_ids), [1] * len(gt))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [skip] eval {ri}: OOM at len={len(seq)} (skipped, cache cleared)", flush=True)
                continue
            if not recs:                              # OOB / too-long guard tripped
                print(f"  [skip] eval {ri}: proposals empty (len={len(seq)})", flush=True)
                continue
            eval_traces.append({"rid": ri, "task": args.task,
                                "prompt_ids": prompt_ids, "output_ids": gt})
            by_pos = {}
            for (ds, d, tokd, conf, gt_rank, gt_p) in recs:
                by_pos.setdefault(ds, {})[d] = (tokd, conf, gt_rank, gt_p)
            Tw, mlw = suffix_sweep(suf_warm, prompt_ids, gt, args.num_spec)
            Tc, mlc = suffix_sweep(suf_cold, prompt_ids, gt, args.num_spec)
            for ds in sorted(by_pos):
                depths = by_pos[ds]; _D = (-1, 0.0, 99, 0.0)
                rec = {
                    "task": args.task, "rid": ri, "pos": ds, "W": W,
                    "dflash_tok": [depths.get(d, _D)[0] for d in range(W)],
                    "dflash_conf": [depths.get(d, _D)[1] for d in range(W)],
                    "dflash_match": [1 if depths.get(d, _D)[2] == 0 else 0 for d in range(W)],
                    "dflash_gt_rank": [depths.get(d, _D)[2] for d in range(W)],
                    "dflash_gt_p": [depths.get(d, _D)[3] for d in range(W)],
                    "suffix_T_warm": Tw[ds - 1] if 0 <= ds - 1 < len(Tw) else 0.0,
                    "suffix_match_warm": mlw[ds - 1] if 0 <= ds - 1 < len(mlw) else 0,
                    "suffix_T_cold": Tc[ds - 1] if 0 <= ds - 1 < len(Tc) else 0.0,
                    "suffix_match_cold": mlc[ds - 1] if 0 <= ds - 1 < len(mlc) else 0,
                }
                f.write(json.dumps(rec) + "\n"); n_rec += 1
            print(f"  [{args.task}] eval {ri}: prompt={len(prompt_ids)} gt={len(gt)} pos={len(by_pos)}", flush=True)
    json.dump({"warm_traces": warm_traces, "eval_traces": eval_traces,
               "num_spec": args.num_spec, "block_size": dfo.block_size},
              open(traces_path, "w"))
    print(f"saved {n_rec} records ({len(eval_traces)} eval) -> {outp}", flush=True)
    print(f"saved traces -> {traces_path}", flush=True)


if __name__ == "__main__":
    main()
