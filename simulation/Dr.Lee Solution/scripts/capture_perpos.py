#!/usr/bin/env python3
"""Rich per-position capture — the data engine for the analysis slides.

For every eval prompt, re-speculates a DFlash block at EVERY committed root
along the target-greedy trace (== DFlashOffline.proposals_for_seq with
commit_lens = all-1) and probes the Suffix tree (warm AND cold) at the same
positions. One dense record per position:

  {task, rid, pos,
   W, dflash_conf:[W], dflash_match:[W], dflash_gt_rank:[W], dflash_gt_p:[W],
   suffix_T_warm, suffix_match_warm, suffix_T_cold, suffix_match_cold}

- dflash_match[d] = 1 iff DFlash's depth-d argmax == gt[pos+1+d] (gt_rank==0).
  → hazard a_i = P(match_i | match_0..i-1), survival S_i = P(all match_0..i-1).
- dflash_conf[d] = softmax-max prob → conf↔a_k calibration (slide 18).
- suffix_T_* = probe.score (warmth). suffix_match_* = greedy-walk accept length
  of the suffix continuation vs gt → suffix survival warm/cold (slide 7).

Feeds slides 4/5/7/12/13b/17/18. GPU (27B). Run INSIDE sglang-bench on GPU0:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 scripts/capture_perpos.py --dataset scripts/bench_prompts_multislot_k4.jsonl \
      --task multislot_k4 --out results/perpos/multislot_k4.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/vendor/ddtree")
sys.path.insert(0, "/workspace/vendor/ddtree/model")
sys.path.insert(0, "/workspace/simulation/scripts/experiments")

from dflash_offline import DFlashOffline  # noqa: E402
from measure_k_fusion import ArcticSuffix  # noqa: E402
from simulation.evaluation.tree_knapsack import greedy_tree_walk  # noqa: E402


@torch.inference_mode()
def greedy_trace(model, tok, prompt, max_tokens, enable_thinking=False):
    """Target-greedy trace (KV-cached). Returns (prompt_ids, output_ids)."""
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  enable_thinking=enable_thinking)
    ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(model.device)
    prompt_ids = ids[0].tolist()
    eos = tok.eos_token_id
    eos = eos[0] if isinstance(eos, (list, tuple)) else eos
    out, past, cur = [], None, ids
    for _ in range(max_tokens):
        o = model(cur, use_cache=True, past_key_values=past)
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        out.append(nxt)
        if nxt == eos:
            break
        cur = torch.tensor([[nxt]], device=model.device)
    return prompt_ids, out


def suffix_sweep(suf: ArcticSuffix, prompt_ids, gt, num_spec):
    """Per-position (T, match_len) probing an already-warmed (or cold) suffix.
    Uses a fresh eval request; feeds gt tokens as it advances (eval-time warming
    OFF here — we want the position-p warmth from the *warm corpus*, so we do NOT
    add_response; the local request only holds the prompt+committed prefix)."""
    suf.new_eval(prompt_ids)
    Ts, mls = [], []
    ctx = list(prompt_ids)
    for p in range(len(gt)):
        toks, T = suf.probe(ctx, num_spec)
        ml = greedy_tree_walk(toks, list(range(-1, len(toks) - 1)), gt[p:]) if toks else 0
        Ts.append(float(T)); mls.append(int(ml))
        # advance the local request by the committed (gt) token
        suf.add_response([gt[p]])
        ctx.append(gt[p])
    return Ts, mls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=os.environ.get("TGT", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--draft", default=os.environ.get("DFT", "z-lab/Qwen3.5-27B-DFlash"))
    ap.add_argument("--dataset", required=True, help="jsonl with {prompt|turns, role?, ...}")
    ap.add_argument("--task", required=True, help="default task label")
    ap.add_argument("--task-field", default=None,
                    help="per-row field to use as the task label (e.g. 'subtask' for specbench)")
    ap.add_argument("--per-task", type=int, default=0,
                    help="if >0 with --task-field: keep this many eval rows PER task (no warm split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nwarm", type=int, default=int(os.environ.get("NWARM", "12")))
    ap.add_argument("--neval", type=int, default=int(os.environ.get("NEVAL", "8")))
    ap.add_argument("--maxtok", type=int, default=int(os.environ.get("MAXTOK", "96")))
    ap.add_argument("--num-spec", type=int, default=int(os.environ.get("NUM_SPEC", "32")))
    args = ap.parse_args()

    dfo = DFlashOffline(args.target, args.draft)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.target)

    def _prompt(r):
        return r.get("prompt") or (r["turns"][0] if r.get("turns") else None)

    def _label(r):
        return r.get(args.task_field, args.task) if args.task_field else args.task

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    if args.task_field and args.per_task:
        # per-subtask task-split (e.g. specbench): N eval rows per task, no warm.
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in rows:
            buckets[_label(r)].append(r)
        warm = []
        ev = [r for lab in sorted(buckets) for r in buckets[lab][:args.per_task]]
    else:
        warm = [r for r in rows if r.get("role") == "warm"][:args.nwarm]
        ev = [r for r in rows if r.get("role") == "eval"][:args.neval]
        if not warm and not ev:                # no split -> first nwarm warm, rest eval
            warm = rows[:args.nwarm]; ev = rows[args.nwarm:args.nwarm + args.neval]

    print(f"[{args.task}] warm={len(warm)} eval={len(ev)} maxtok={args.maxtok}", flush=True)
    warm_traces = [greedy_trace(dfo.target, tok, _prompt(r), args.maxtok)[1] for r in warm]

    suf_warm = ArcticSuffix(); suf_warm.fit(warm_traces)
    suf_cold = ArcticSuffix()                  # never fit -> cold

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    # companion traces file — lets replay_extension rebuild the (model-free) suffix
    # tree and GT-walk with ZERO GPU: warm output traces + per-eval (prompt, gt).
    traces_path = outp.with_suffix(".traces.json")
    eval_traces = []
    n_rec = 0
    with open(outp, "w") as f:
        for ri, r in enumerate(ev):
            prompt_ids, gt = greedy_trace(dfo.target, tok, _prompt(r), args.maxtok)
            if not gt:
                continue
            task_label = _label(r)
            eval_traces.append({"rid": ri, "task": task_label,
                                "prompt_ids": prompt_ids, "output_ids": gt})
            seq = prompt_ids + gt
            # DFlash per-position (block at every root): (ds, d, tok, conf, gt_rank, gt_p)
            recs = dfo.proposals_for_seq(seq, len(prompt_ids), [1] * len(gt))
            W = dfo.block_size - 1
            by_pos = {}
            for (ds, d, tokd, conf, gt_rank, gt_p) in recs:
                by_pos.setdefault(ds, {})[d] = (tokd, conf, gt_rank, gt_p)
            # Suffix per-position warm + cold
            Tw, mlw = suffix_sweep(suf_warm, prompt_ids, gt, args.num_spec)
            Tc, mlc = suffix_sweep(suf_cold, prompt_ids, gt, args.num_spec)
            for ds in sorted(by_pos):
                depths = by_pos[ds]
                _D = (-1, 0.0, 99, 0.0)
                dflash_tok = [depths.get(d, _D)[0] for d in range(W)]
                conf = [depths.get(d, _D)[1] for d in range(W)]
                rank = [depths.get(d, _D)[2] for d in range(W)]
                gtp = [depths.get(d, _D)[3] for d in range(W)]
                match = [1 if rank[d] == 0 else 0 for d in range(W)]
                pos = ds - 1                    # ds starts at 1 for the first gt position
                rec = {
                    "task": task_label, "rid": ri, "pos": ds,
                    "W": W, "dflash_tok": dflash_tok, "dflash_conf": conf,
                    "dflash_match": match, "dflash_gt_rank": rank, "dflash_gt_p": gtp,
                    "suffix_T_warm": Tw[pos] if 0 <= pos < len(Tw) else 0.0,
                    "suffix_match_warm": mlw[pos] if 0 <= pos < len(mlw) else 0,
                    "suffix_T_cold": Tc[pos] if 0 <= pos < len(Tc) else 0.0,
                    "suffix_match_cold": mlc[pos] if 0 <= pos < len(mlc) else 0,
                }
                f.write(json.dumps(rec) + "\n")
                n_rec += 1
            print(f"  [{args.task}] eval {ri}: gt_len={len(gt)} positions={len(by_pos)}", flush=True)
    json.dump({"warm_traces": warm_traces, "eval_traces": eval_traces,
               "num_spec": args.num_spec, "block_size": dfo.block_size},
              open(traces_path, "w"))
    print(f"saved {n_rec} per-position records -> {outp}", flush=True)
    print(f"saved traces ({len(warm_traces)} warm + {len(eval_traces)} eval) -> {traces_path}", flush=True)


if __name__ == "__main__":
    main()
