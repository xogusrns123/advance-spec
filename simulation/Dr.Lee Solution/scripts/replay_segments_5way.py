#!/usr/bin/env python3
"""Per-SEGMENT MAT: replay the five figure arms (DFlash, Suffix, Compose/calib,
SD-paper hybrid fallback, Oracle) with every round tagged by the output segment
its ROOT position falls in — the standing 4-way taxonomy:

  think      reasoning region, up to and incl </think> (thinking-ON workloads)
  preamble   non-think text of a turn that CONTAINS a tool call
  tool_call  the tool-call span itself (bfcl "[f(...)]", swebench
             ```mswea_bash_command```, spider "Action: Bash(...)")
  final      text of a turn WITHOUT any tool call (the answer/response turn)

specbench is non-agentic (no tool_call/preamble: all non-think = final);
swebench/spider were collected thinking-OFF (no think rounds).

Replay semantics are IDENTICAL to the 4-way figures (replay_extension:
two-way in-sample, warm-set-only tree, teacher-forced greedy, num_spec budget,
root+accepted+bonus advance) and to replay_fallback_sweep for the fallback arm
(round-level binary switch at a FIXED tau — pass the pooled-best tau from the
fallback sweep so the arm matches the 5-bar comparison figure).

Output: one JSONL per arm with one row per round:
  {"rid","task","m","seg","acc"}

  python3 scripts/replay_segments_5way.py \
      --record results/perpos_bfcl_full/bfcl_v4_full.jsonl --kind bfcl \
      --arms dflash suffix fallback --tau 16 \
      --out-dir /workspace/simulation/results/pipeline_4way/segments --tag _pre
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from replay_extension import _ad, _fit_beta, _fit_tail_iso, split_parity  # noqa: E402
from measure_k_fusion import ArcticSuffix  # noqa: E402
from fusion_tree import build_extension_chain  # noqa: E402
from simulation.evaluation.tree_knapsack import greedy_tree_walk_path  # noqa: E402

ARMS = ["dflash", "suffix", "calib", "fallback", "oracle"]


# ---------------------------------------------------------------- segmentation
def piece_offsets(ids, tok):
    """exact char offsets by per-token decode (byte-BPE concat == full decode)."""
    pieces = [tok.decode([t], skip_special_tokens=False) for t in ids]
    offs, c = [], 0
    for p in pieces:
        offs.append((c, c + len(p))); c += len(p)
    return offs, "".join(pieces)


# first-tool-call marker per workload. Everything BEFORE it in a turn is
# reasoning — both the <think>...</think> block and the ReAct "THOUGHT:"/"Thought:"
# planning line are the agent's reasoning, not a separate "preamble" (agentic tool
# turns have no conversational lead-in). Everything from the marker to the end is
# the tool call. A turn WITHOUT a marker is a text answer: reasoning up to
# </think>, then the "final" response. So "preamble" does not occur in practice.
_TOOL_MARK = {
    "bfcl": r"\[\s*[A-Za-z_]\w*\s*\(",       # [func(  (prompting-mode call)
    "swebench": r"```mswea_bash_command",     # mini-swe-agent bash action fence
    "spider": r"Action:\s*[A-Za-z_]\w*\s*\(",  # spider-dbt ReAct action (Bash /
                                               # LOCAL_DB_SQL / CreateFile / EditFile
                                               # / Terminate — ALL are tool actions,
                                               # no natural-language final response)
    "tau2": r"<tool_call>",                   # tau2 tool-call block
}


def tag(full, offs, kind):
    """Per-token segment: think (reasoning) / tool_call / final. In a tool-call
    turn all pre-call text (the <think> block + any THOUGHT:/Thought: planning) is
    reasoning and the call runs to the end; a no-call turn is reasoning up to
    </think> then a final text answer. (No "preamble" — see note above.)"""
    pat = _TOOL_MARK.get(kind)
    m = re.search(pat, full) if pat else None
    if m:                                     # tool-call turn
        call = m.start()
        return ["think" if e <= call else "tool_call" for s, e in offs]
    et = full.find("</think>")                # text-answer turn (or non-agentic)
    think_end = et + len("</think>") if et != -1 else 0
    return ["think" if e <= think_end else "final" for s, e in offs]


# ------------------------------------------------------------------- replay
def replay_arm(rby, gt, pids, suffix, arm, num_spec, max_rounds, calib, tau):
    """-> list of (m, acc) — round root position + realized accept length.
    Mirrors replay_extension.replay_proposer / replay_fallback_sweep exactly."""
    if suffix is not None:
        suffix.new_eval(pids)
    out = []
    m = 0
    for _ in range(max_rounds):
        rec = rby.get(m + 1)
        if rec is None or m >= len(gt):
            break
        block_full, conf = rec["dflash_tok"], rec["dflash_conf"]
        root = gt[m]
        ctx_list = pids + gt[:m] + [root]

        if arm == "dflash":
            tree = build_extension_chain(block_full[:num_spec], [])
        elif arm == "suffix":
            suf, T = suffix.probe(ctx_list, num_spec)
            tree = build_extension_chain([], suf[:num_spec])
        elif arm == "fallback":
            suf, T = suffix.probe(ctx_list, num_spec)
            if T >= tau:
                tree = build_extension_chain([], suf[:num_spec])
            else:
                tree = build_extension_chain(block_full[:num_spec], [])
        elif arm == "calib":
            cal_h, cal_t = calib
            W_ = min(num_spec, len(conf))
            tails = []
            for kk in range(W_ + 1):
                budget = num_spec - kk
                tails.append(suffix._spec(ctx_list + block_full[:kk], budget)
                             if budget > 0 else ([], 0.0))
            S_k, G_k = 1.0, 0.0
            best_val, k = 1.0 + cal_t(tails[0][1]), 0
            for j in range(W_):
                S_k *= cal_h(conf[j]); G_k += S_k
                val = 1.0 + G_k + S_k * cal_t(tails[j + 1][1])
                if val > best_val:
                    best_val, k = val, j + 1
            tree = build_extension_chain(block_full[:k],
                                         tails[k][0][:max(0, num_spec - k)])
        elif arm == "oracle":
            W = rec["W"]
            best_acc, best_tree = -1, build_extension_chain([], [])
            for kk in range(W + 1):
                tk = suffix.speculate(ctx_list + block_full[:kk], num_spec)
                tr = build_extension_chain(block_full[:kk],
                                           tk[:max(0, num_spec - kk)])
                pth = greedy_tree_walk_path(list(tr.tokens), list(tr.parents),
                                            gt[m + 1:])
                if len(pth) > best_acc:
                    best_acc, best_tree = len(pth), tr
            tree = best_tree
        else:
            raise ValueError(arm)

        path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents),
                                     gt[m + 1:])
        acc = len(path)
        out.append((m, acc))
        if suffix is not None:
            accepted_toks = [tree.tokens[i] for i in path]
            nxt = [root] + accepted_toks
            if m + 1 + acc < len(gt):
                nxt.append(gt[m + 1 + acc])
            suffix.add_response(nxt)
        m += 1 + acc + 1
        if m >= len(gt):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--kind", required=True,
                    choices=["bfcl", "specbench", "swebench", "spider", "tau2"])
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--tau", type=float, default=None,
                    help="fallback threshold (pooled-best from the sweep)")
    ap.add_argument("--max-rounds", type=int, default=4096)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--raw-cals", action="store_true",
                    help="calib arm with IDENTITY calibrators (raw conf + raw arctic "
                         "score; no beta hazard / isotonic tail). Default arm.")
    ap.add_argument("--head-cal", default="", choices=["", "affine", "logistic", "beta"],
                    help="calib arm head hazard (default beta); tail is isotonic.")
    ap.add_argument("--tail-cal", default="", choices=["", "isotonic"], help="(isotonic only)")
    ap.add_argument("--three-way", action="store_true",
                    help="deployable split: calibrators fit on the calibrate half "
                         "(even groups), rounds emitted for the test half only "
                         "(pass the calib-split-picked tau for the fallback arm)")
    ap.add_argument("--group-mode", default="conv",
                    choices=["rid", "lenreset", "conv", "convlabel"],
                    help="split unit for --three-way (see replay_extension)")
    args = ap.parse_args()
    if "fallback" in args.arms and args.tau is None:
        ap.error("--tau required for the fallback arm")

    traces = json.load(open(Path(args.record).with_suffix(".traces.json")))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)

    recs = defaultdict(dict)
    task_of = {}
    for l in open(args.record):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r
        task_of[r["rid"]] = r["task"]
    print(f"{args.kind}: {len(recs)} eval reqs, num_spec={num_spec}, "
          f"arms={args.arms}", flush=True)

    # deployable three-way split: fit on calib half, emit test half only
    test_rids = None
    calib_rids = set(recs)
    if args.three_way:
        _, par = split_parity(eval_traces, args.group_mode)
        calib_rids = {rid for rid in recs if par.get(rid, 0) == 0}
        test_rids = {rid for rid in recs if par.get(rid, 0) == 1}
        print(f"three-way({args.group_mode}): calib={len(calib_rids)} "
              f"test={len(test_rids)} calls", flush=True)

    # per-token segment tags for every eval trace
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cats_by_rid = {}
    for rid in recs:
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        offs, full = piece_offsets(tr["output_ids"], tok)
        cats_by_rid[rid] = tag(full, offs, args.kind)
    print(f"segment tags built for {len(cats_by_rid)} traces", flush=True)

    # in-sample calibrators (exactly replay_extension --calib-insample --hazard-fit beta)
    calib = None
    if "calib" in args.arms and args.raw_cals:
        calib = (lambda c: float(c), lambda t: float(t))
        print("raw-cals: identity calibrators (raw conf + raw arctic score)", flush=True)
    elif "calib" in args.arms:
        from replay_extension import _fit_logistic
        pairs = []
        for rid in sorted(calib_rids):
            for r in recs[rid].values():
                conf, match = r["dflash_conf"], r["dflash_match"]
                ad = _ad(match)
                for d in range(min(ad + 1, len(conf))):
                    pairs.append((float(conf[d]), int(match[d])))
        hx, hy = [x for x, _ in pairs], [y for _, y in pairs]
        hc, tc = args.head_cal or "beta", args.tail_cal or "isotonic"
        if hc == "logistic":
            cal_h = _fit_logistic(hx, hy)
        elif hc == "affine":
            cal_h = lambda c: min(1.0, max(0.0, 0.69 * float(c) + 0.29))  # noqa: E731
        else:
            cal_h = _fit_beta(hx, hy)
        cal_t = _fit_tail_iso(warm_traces, recs, eval_traces, calib_rids,
                              num_spec, args.max_rounds)   # isotonic (only tail option here)
        calib = (cal_h, cal_t)
        print(f"calib head={hc} tail={tc}", flush=True)
        print(f"calibrators fit on {len(calib_rids)} calls "
              + ("(three-way calibrate split)" if args.three_way else "(in-sample)"),
              flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for arm in args.arms:
        suffix = None
        if arm != "dflash":
            suffix = ArcticSuffix(); suffix.fit(warm_traces)
        fp = os.path.join(args.out_dir, f"seg_{args.kind}_{arm}{args.tag}.jsonl")
        n, tot = 0, 0.0
        seg_agg = defaultdict(lambda: [0, 0.0])
        with open(fp, "w") as f:
            for rid, rby in recs.items():
                if test_rids is not None and rid not in test_rids:
                    continue                     # three-way: test half only
                tr = eval_traces.get(rid)
                if tr is None:
                    continue
                cats = cats_by_rid[rid]
                for m, acc in replay_arm(rby, tr["output_ids"], tr["prompt_ids"],
                                         suffix, arm, num_spec, args.max_rounds,
                                         calib, args.tau):
                    seg = cats[m] if 0 <= m < len(cats) else "final"
                    f.write(json.dumps({"rid": rid, "task": task_of[rid],
                                        "m": m, "seg": seg, "acc": acc}) + "\n")
                    n += 1; tot += acc
                    a = seg_agg[seg]; a[0] += 1; a[1] += acc
        print(f"[{arm}] K={tot / n if n else 0:.2f} (rounds={n}) -> {fp}", flush=True)
        for seg in ("think", "preamble", "tool_call", "final"):
            if seg in seg_agg:
                c, s = seg_agg[seg]
                print(f"    {seg:<10} K={s / c:.2f} (rounds={c}, {100 * c / n:.1f}%)",
                      flush=True)


if __name__ == "__main__":
    main()
