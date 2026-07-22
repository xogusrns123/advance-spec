#!/usr/bin/env python3
"""Extension-sim capture over a FULL-TRAJECTORY collection (bfcl_v4_full_traj /
swebench_full_traj). Input = chain_hybrid gt_tokens.jsonl ({input_ids,
output_ids} per generation call, target-greedy) + conv_map.json
(build_conv_map.py) giving the EXACT conversation/task of every row. No greedy
generation — only the DFlash block proposals per position are computed (GPU);
the suffix side is model-free.

Split (whole conversations, interleaved): even conv -> warm (suffix corpus),
odd conv -> eval (per-position records). Dropped everywhere: orphan rows
(aborted-restart residue) and runaway rows (output >= --drop-out-ge, i.e.
max_tokens truncation loops — they are degenerate self-repeats that fake-
inflate suffix accept). Record schema = capture_perpos + "conv", so
replay_extension --group-mode conv splits calibrate/test exactly
conversation-disjoint; analyze_perpos works unchanged.

  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/capture_traj.py \
    --gt-tokens /workspace/simulation/results/bfcl_v4_full_traj/qwen35_27b_dflash/gt_tokens.jsonl \
    --conv-map results/perpos_bfcl_full/conv_map.json \
    --task bfcl_v4_full --out results/perpos_bfcl_full/bfcl_v4_full.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
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
    ap.add_argument("--conv-map", required=True, help="build_conv_map.py output")
    ap.add_argument("--task", default="traj", help="dataset label (fallback when a row has none)")
    ap.add_argument("--num-spec", type=int, default=int(os.environ.get("NUM_SPEC", "32")))
    ap.add_argument("--min-gt", type=int, default=1, help="skip eval rows with fewer gt tokens")
    ap.add_argument("--max-seq", type=int, default=24000,
                    help="skip eval rows whose prompt+gt exceeds this (OOM guard on the "
                         "single full-seq 27B feature forward)")
    ap.add_argument("--drop-out-ge", type=int, default=8192,
                    help="drop rows (warm AND eval) whose output length >= this "
                         "(max_tokens-truncated runaway loops); 0 disables")
    ap.add_argument("--limit-eval-convs", type=int, default=0,
                    help="smoke mode: keep only the first N eval conversations (0 = all)")
    ap.add_argument("--eval-set", default="eval", choices=["eval", "warm", "all"],
                    help="which half to EVALUATE (capture records for). 'eval' (default) "
                         "= held-out odd-parity convs; 'warm' = the SAME even-parity convs "
                         "the suffix corpus is built from — the in-corpus / exact-repeat "
                         "regime (multislot-k0 analog): the tree contains the evaluated "
                         "conversation's own outputs. The corpus is ALWAYS the warm half. "
                         "'all' = NO warm/eval split and NO warm corpus: every conversation "
                         "is evaluated (temporal order) and warm_traces is empty, so the "
                         "suffix tree self-warms only from the eval stream.")
    ap.add_argument("--split-mode", default="conv", choices=["conv", "label-rank"],
                    help="warm/eval split parity: global conv id (conv) or the conv's "
                         "rank WITHIN its task label (label-rank — required for "
                         "subtask-interleaved datasets like specbench, where global "
                         "parity aliases whole labels onto one side)")
    ap.add_argument("--per-task-eval-convs", type=int, default=0,
                    help="keep only the first N eval conversations PER task label (0 = all)")
    ap.add_argument("--max-calls-per-conv", type=int, default=0,
                    help="keep only the first M calls of each eval conversation (0 = all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="dump .traces.json every N evaluated rows (0 = only at end). "
                         "The jsonl is flushed per-position already; a periodic traces "
                         "dump keeps a consistent (partial) checkpoint that replay can "
                         "read if the run is killed mid-way.")
    ap.add_argument("--deadline-epoch", type=float, default=0.0,
                    help="unix epoch seconds; when reached, checkpoint and stop cleanly "
                         "BEFORE starting the next row (0 = no deadline).")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.gt_tokens) if l.strip()]
    cmap = json.load(open(args.conv_map))
    row_conv = cmap["rows"]
    if len(row_conv) != len(rows):
        raise SystemExit(f"conv_map rows {len(row_conv)} != gt rows {len(rows)}")

    conv_par = {}
    if args.split_mode == "label-rank":
        rank = {}
        for c in cmap["convs"]:                # collection order
            l = c["label"]
            conv_par[c["conv"]] = rank.get(l, 0) % 2
            rank[l] = rank.get(l, 0) + 1

    n_orphan = n_runaway = 0
    warm_rows, eval_rows = [], []          # (global_idx, row, conv, label)
    for gi, (r, m) in enumerate(zip(rows, row_conv)):
        if m is None:
            n_orphan += 1
            continue
        if args.drop_out_ge and len(r.get("output_ids") or []) >= args.drop_out_ge:
            n_runaway += 1
            continue
        item = (gi, r, m["conv"], m.get("task") or args.task)
        par = conv_par.get(m["conv"], m["conv"] % 2)
        (warm_rows if par == 0 else eval_rows).append(item)
    if args.eval_set == "warm":
        eval_rows = list(warm_rows)        # evaluate the corpus half itself
    elif args.eval_set == "all":           # no split, no corpus: eval everything
        eval_rows = sorted(warm_rows + eval_rows, key=lambda it: it[0])
        warm_rows = []                     # -> warm_traces empty; tree self-warms
    if args.limit_eval_convs:
        keep = sorted({c for _, _, c, _ in eval_rows})[:args.limit_eval_convs]
        eval_rows = [e for e in eval_rows if e[2] in set(keep)]
    if args.per_task_eval_convs:
        order, lab_of = [], {}
        for _, _, c, lab in eval_rows:              # temporal order; convs contiguous
            if c not in lab_of:
                order.append(c); lab_of[c] = lab
        cnt, keep = {}, set()
        for c in order:
            l = lab_of[c]
            if cnt.get(l, 0) < args.per_task_eval_convs:
                keep.add(c); cnt[l] = cnt.get(l, 0) + 1
        eval_rows = [e for e in eval_rows if e[2] in keep]
    if args.max_calls_per_conv:
        ncall, kept = {}, []
        for e in eval_rows:
            ncall[e[2]] = ncall.get(e[2], 0) + 1
            if ncall[e[2]] <= args.max_calls_per_conv:
                kept.append(e)
        eval_rows = kept
    warm_convs = {c for _, _, c, _ in warm_rows}
    eval_convs = {c for _, _, c, _ in eval_rows}
    print(f"[{args.task}] {len(rows)} rows: dropped {n_orphan} orphan + {n_runaway} runaway "
          f"(out>={args.drop_out_ge}); corpus {len(warm_convs)} convs/{len(warm_rows)} calls, "
          f"evaluated({args.eval_set} set) {len(eval_convs)} convs/{len(eval_rows)} calls",
          flush=True)
    warm_traces = [r["output_ids"] for _, r, _, _ in warm_rows if r.get("output_ids")]

    dfo = DFlashOffline(args.target, args.draft)
    suf_warm = ArcticSuffix(); suf_warm.fit(warm_traces)
    suf_cold = ArcticSuffix()

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    traces_path = outp.with_suffix(".traces.json")
    eval_traces, n_rec, n_skip_len, n_skip_oom = [], 0, 0, 0
    W = dfo.block_size - 1
    t0 = time.time()

    def dump_traces():
        # atomic: write to .tmp then rename so a kill mid-dump never corrupts it.
        tmp = traces_path.with_suffix(".json.tmp")
        json.dump({"warm_traces": warm_traces, "eval_traces": eval_traces,
                   "num_spec": args.num_spec, "block_size": dfo.block_size},
                  open(tmp, "w"))
        os.replace(tmp, traces_path)

    stopped_early = False
    with open(outp, "w") as f:
        import torch
        for ei, (gi, r, conv, label) in enumerate(eval_rows):
            if args.deadline_epoch and time.time() >= args.deadline_epoch:
                f.flush(); dump_traces()
                print(f"  [deadline] reached at eval {ei}/{len(eval_rows)}; "
                      f"checkpointed {len(eval_traces)} eval calls / {n_rec} records "
                      f"and stopping cleanly", flush=True)
                stopped_early = True
                break
            prompt_ids = list(r.get("input_ids") or [])
            gt = list(r.get("output_ids") or [])
            if len(gt) < args.min_gt or not prompt_ids:
                continue
            if len(prompt_ids) + len(gt) > args.max_seq:
                n_skip_len += 1
                print(f"  [skip] row {gi}: seq {len(prompt_ids)+len(gt)} > {args.max_seq}",
                      flush=True)
                continue
            seq = prompt_ids + gt
            try:
                recs = dfo.proposals_for_seq(seq, len(prompt_ids), [1] * len(gt),
                                             max_seq=args.max_seq)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_skip_oom += 1
                print(f"  [skip] row {gi}: OOM at len={len(seq)} (cache cleared)", flush=True)
                continue
            if not recs:                              # OOB / guard tripped inside
                print(f"  [skip] row {gi}: proposals empty (len={len(seq)})", flush=True)
                continue
            eval_traces.append({"rid": gi, "task": label, "conv": conv,
                                "prompt_ids": prompt_ids, "output_ids": gt})
            by_pos = {}
            for (ds, d, tokd, conf, gt_rank, gt_p) in recs:
                by_pos.setdefault(ds, {})[d] = (tokd, conf, gt_rank, gt_p)
            Tw, mlw = suffix_sweep(suf_warm, prompt_ids, gt, args.num_spec)
            Tc, mlc = suffix_sweep(suf_cold, prompt_ids, gt, args.num_spec)
            for ds in sorted(by_pos):
                depths = by_pos[ds]; _D = (-1, 0.0, 99, 0.0)
                rec = {
                    "task": label, "rid": gi, "conv": conv, "pos": ds, "W": W,
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
            torch.cuda.empty_cache()
            el = time.time() - t0
            print(f"  [{args.task}] eval {ei+1}/{len(eval_rows)} row={gi} conv={conv} "
                  f"({label}): prompt={len(prompt_ids)} gt={len(gt)} pos={len(by_pos)} "
                  f"cum_rec={n_rec} {el/60:.1f}min", flush=True)
            if args.checkpoint_every and (ei + 1) % args.checkpoint_every == 0:
                f.flush(); dump_traces()
                print(f"  [checkpoint] {len(eval_traces)} eval calls / {n_rec} records "
                      f"-> {traces_path}", flush=True)
    dump_traces()
    print(f"saved {n_rec} records ({len(eval_traces)} eval calls, "
          f"skipped {n_skip_len} long + {n_skip_oom} OOM"
          f"{'; STOPPED EARLY at deadline' if stopped_early else ''}) -> {outp}", flush=True)
    print(f"saved traces -> {traces_path}", flush=True)


if __name__ == "__main__":
    main()
