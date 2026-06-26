#!/usr/bin/env python3
"""Capture the TARGET model's probability of each drafted token, offline.

For the continuous-objective calibration experiment: instead of the binary
``token == gt_token`` accept label, we want to regress each proposer's score onto
``q_target(drafted_token | GT prefix)`` — the target model's softmax probability
of the token it proposed, conditioned on the (teacher-forced) ground-truth
prefix. This script produces that label by re-running the target model
teacher-forced on the GT trajectories and reading the full-vocab softmax at each
decision position.

Inputs:
  --decision-log  decisions_select1_oracle.jsonl from an ORACLE arm (STEP D of
                  run_o4_*.sh). Carries per-decision rows (rid, decode_step,
                  depth, eagle_token, suffix_token, gt_token, oracle_hit), per-
                  step rows (accept_len), and — for runs that include the patch's
                  oracle-mode input_ids logging — per-rid {"type":"req"} rows.
  --gt            gt_tokens.jsonl from the RECORD arm: {input_ids, output_ids}.
  --model         HF name/path of the TARGET model (e.g. Qwen/Qwen3-14B).

Output (--out, JSONL, one row per labelable decision):
  {"rid","decode_step","depth","q_eagle","q_suffix","q_gt"}
joined back to the decision log by (rid, decode_step, depth) by
fit_chain_hybrid_calib_perpos.py --target-prob-labels.

Bridge (oracle rid -> GT trajectory):
  EXACT  — via the {"type":"req"} input_ids rows (preferred): rid -> input_ids ->
           output_ids through the same gt_map the patch builds. Greedy outputs can
           collide across DISTINCT prompts, so input_ids is the only safe key.
  CONTENT (fallback for legacy logs without req rows) — anchor each rid by its
           first decode step's contiguous depth-0.. gt_token run, score every
           (gt_row, L1) candidate by full gt_token consistency, and keep the
           unique >0.95 match; rids whose candidates collide (identical output,
           different prompt) are DROPPED and counted (never silently).

Position alignment (verified against real logs): a decision at (rid, decode_step,
depth) scores GT output index ``j = L(decode_step) + depth`` where ``L`` is the
output length before that step — reconstructed as ``L1 + cumsum(accept_len + 1)``
over prior steps (accepted prefix + 1 bonus token per step; L1 = the prefill
offset, empirically 1). Every emitted row is asserted ``output_ids[j] ==
gt_token``; ``L1`` is auto-detected per rid by trying a small offset set.

Usage (inside sglang-bench container, GPU0 free):
  python3 simulation/scripts/capture_target_probs.py \
      --decision-log .../qwen3_14b_tp_train/decisions_select1_oracle.jsonl \
      --gt          .../qwen3_14b_tp_train/gt_tokens.jsonl \
      --model Qwen/Qwen3-14B \
      --out         .../qwen3_14b_tp_train/target_probs.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

L1_CANDIDATES = (1, 0, 2, 3)   # prefill offset; 1 in practice, search to be safe


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

def load_decision_log(path: str):
    """-> (req_inputs{rid:tuple}, accept_len{(rid,step):int}, decs{rid:[rows]})."""
    req_inputs: dict = {}
    accept_len: dict = {}
    decs: dict = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("type")
            if t == "req":
                req_inputs[r["rid"]] = tuple(r["input_ids"])
            elif t == "step":
                accept_len[(r["rid"], r["decode_step"])] = r["accept_len"]
            elif t == "decision" and not r.get("tail"):
                decs[r["rid"]].append(r)
    return req_inputs, accept_len, dict(decs)


def load_gt(path: str):
    """-> (gt_by_input{tuple(input):output_ids}, gt_list[(input,output)])."""
    gt_by_input: dict = {}
    gt_list: list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            inp = tuple(r["input_ids"])
            out = list(r["output_ids"])
            gt_by_input[inp] = out
            gt_list.append((inp, out))
    return gt_by_input, gt_list


# ---------------------------------------------------------------------------
# L reconstruction + integrity
# ---------------------------------------------------------------------------

def build_Lmap(steps_al: dict, rid: str, L1: int) -> dict:
    """decode_step -> L (output length before that step)."""
    L = {}
    cur = L1
    for k in sorted(s for (r, s) in steps_al if r == rid):
        L[k] = cur
        cur = cur + steps_al[(rid, k)] + 1   # accepted prefix + 1 bonus
    return L


def consistency(rid, rows, out, steps_al, L1):
    """Fraction of gt-bearing rows whose output_ids[L+depth] == gt_token."""
    L = build_Lmap(steps_al, rid, L1)
    ok = tot = 0
    for d in rows:
        if d.get("gt_token") is None:
            continue
        k = d["decode_step"]
        if k not in L:
            continue
        j = L[k] + d["depth"]
        tot += 1
        if 0 <= j < len(out) and out[j] == d["gt_token"]:
            ok += 1
    return (ok / tot if tot else 0.0), tot


def pick_L1(rid, rows, out, steps_al):
    """Return the L1 offset giving full gt consistency, or None."""
    best = None
    for L1 in L1_CANDIDATES:
        frac, tot = consistency(rid, rows, out, steps_al, L1)
        if tot and frac > 0.999:
            return L1
        if best is None or frac > best[1]:
            best = (L1, frac)
    return None


# ---------------------------------------------------------------------------
# Bridges
# ---------------------------------------------------------------------------

def bridge_exact(req_inputs, gt_by_input):
    """rid -> output_ids via the input_ids req rows. Returns (map, n_unmatched)."""
    out = {}
    unmatched = 0
    for rid, inp in req_inputs.items():
        o = gt_by_input.get(inp)
        if o is None:
            unmatched += 1
        else:
            out[rid] = o
    return out, unmatched


def _first_step_run(rows):
    ds0 = min(d["decode_step"] for d in rows)
    s = sorted((d for d in rows if d["decode_step"] == ds0), key=lambda x: x["depth"])
    run = []
    for i, d in enumerate(s):
        if d["depth"] != i or d.get("gt_token") is None:
            break
        run.append(d["gt_token"])
    return run


def bridge_content(decs, accept_len, gt_list):
    """Fallback: anchor by the first step's contiguous gt run + full-consistency
    scoring. Returns (map{rid:output_ids}, n_dropped_collision, n_none)."""
    out = {}
    dropped = none = 0
    for rid, rows in decs.items():
        run = _first_step_run(rows)
        if len(run) < 3:
            none += 1
            continue
        winners = []
        for (_inp, o) in gt_list:
            n = len(o)
            for s in range(0, n - len(run) + 1):
                if o[s:s + len(run)] == run:
                    # validate this (gt_row, L1=s) against ALL rows
                    frac, tot = consistency(rid, rows, o, accept_len, s)
                    if tot and frac > 0.95:
                        winners.append(o)
                        break   # one anchor per gt row is enough
        # dedupe winners by content (identical outputs are harmless)
        uniq = {tuple(w) for w in winners}
        if len(uniq) == 1:
            out[rid] = winners[0]
        elif not uniq:
            none += 1
        else:
            dropped += 1   # genuine collision: same output, different prompt
    return out, dropped, none


# ---------------------------------------------------------------------------
# Target teacher-forcing
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decision-log", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seq-len", type=int, default=0,
                    help="0 = no limit; else left-truncate the fed sequence to "
                         "the last N tokens (rows whose context falls out of the "
                         "window are skipped + counted)")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--limit-rids", type=int, default=0,
                    help="0 = all; else process only the first N bridged rids "
                         "(smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    req_inputs, accept_len, decs = load_decision_log(args.decision_log)
    gt_by_input, gt_list = load_gt(args.gt)
    log(f"decision rids={len(decs)}  req(input_ids) rows={len(req_inputs)}  "
        f"gt trajectories={len(gt_list)}")

    # --- bridge rid -> output_ids
    if req_inputs:
        rid2out, unmatched = bridge_exact(req_inputs, gt_by_input)
        log(f"bridge=EXACT(input_ids): matched={len(rid2out)} unmatched={unmatched}")
    else:
        rid2out, dropped, none = bridge_content(decs, accept_len, gt_list)
        log(f"bridge=CONTENT(fallback): matched={len(rid2out)} "
            f"dropped_collision={dropped} no_anchor={none}  "
            f"(WARNING: no input_ids req rows; collisions are dropped, not guessed)")

    # --- per-rid L1 + integrity, collect work
    rid2L1 = {}
    bad_rids = []
    for rid, out in rid2out.items():
        L1 = pick_L1(rid, decs[rid], out, accept_len)
        if L1 is None:
            bad_rids.append(rid)
        else:
            rid2L1[rid] = L1
    for rid in bad_rids:
        rid2out.pop(rid, None)
    if bad_rids:
        log(f"WARNING: {len(bad_rids)} rids failed the output_ids[j]==gt_token "
            f"integrity check and were dropped")
    log(f"labelable rids={len(rid2out)}  (L1 offsets used: "
        f"{sorted(set(rid2L1.values()))})")

    # --- load target model
    import torch
    from transformers import AutoModelForCausalLM
    log(f"loading target model {args.model} on {args.device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True).to(args.device)
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_skip_window = 0
    q_gt_sum = q_e_sum = q_s_sum = 0.0
    q_gt_n = q_e_n = q_s_n = 0
    fh = open(out_path, "w")
    rids = sorted(rid2out)
    if args.limit_rids:
        rids = rids[:args.limit_rids]
        log(f"--limit-rids: processing first {len(rids)} rids only")
    for ri, rid in enumerate(rids):
        out = rid2out[rid]
        inp = list(req_inputs.get(rid) or ())
        if not inp:
            # content-bridge: recover input_ids from the matched gt row
            for (cand_in, cand_out) in gt_list:
                if cand_out == out:
                    inp = list(cand_in)
                    break
        L = build_Lmap(accept_len, rid, rid2L1[rid])
        seq = inp + out
        n_in = len(inp)

        # gather the logits indices we need: index t predicts seq[t+1], so the
        # distribution over output_ids[j] lives at logits[n_in + j - 1].
        rows = [d for d in decs[rid] if d.get("gt_token") is not None
                and d["decode_step"] in L]
        wanted = {}   # logits_index -> needed
        for d in rows:
            j = L[d["decode_step"]] + d["depth"]
            if 0 <= j < len(out):
                wanted[n_in + j - 1] = True

        offset = 0
        fed = seq
        if args.max_seq_len and len(seq) > args.max_seq_len:
            offset = len(seq) - args.max_seq_len
            fed = seq[offset:]
        needed = sorted(idx for idx in wanted if idx - offset >= 0)
        n_skip_window += len(wanted) - len(needed)
        if not needed:
            continue

        ids = torch.tensor([fed], dtype=torch.long, device=args.device)
        with torch.no_grad():
            logits = model(ids).logits[0]                 # [T, V]
            sub = logits[[idx - offset for idx in needed]].float()
            del logits
            probs = sub.softmax(-1)                        # [n_needed, V]
        row_of = {idx: i for i, idx in enumerate(needed)}

        for d in rows:
            j = L[d["decode_step"]] + d["depth"]
            li = n_in + j - 1
            if li not in row_of or not (0 <= j < len(out)):
                continue
            assert out[j] == d["gt_token"], (
                f"alignment broke: rid={rid} step={d['decode_step']} "
                f"depth={d['depth']} j={j} out={out[j]} gt={d['gt_token']}")
            pr = probs[row_of[li]]
            rec = {"rid": rid, "decode_step": d["decode_step"], "depth": d["depth"]}
            et = d.get("eagle_token")
            stk = d.get("suffix_token")
            gtk = d.get("gt_token")
            if et is not None:
                q = float(pr[et]); rec["q_eagle"] = round(q, 8)
                q_e_sum += q; q_e_n += 1
            if stk is not None:
                q = float(pr[stk]); rec["q_suffix"] = round(q, 8)
                q_s_sum += q; q_s_n += 1
            if gtk is not None:
                q = float(pr[gtk]); rec["q_gt"] = round(q, 8)
                q_gt_sum += q; q_gt_n += 1
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n_rows += 1

        del probs, sub
        if (ri + 1) % args.checkpoint_every == 0 or ri == len(rids) - 1:
            fh.flush()
            gc.collect()
            torch.cuda.empty_cache()
            log(f"  [{ri+1}/{len(rids)}] rows={n_rows} "
                f"({time.time()-t0:.0f}s)")
    fh.close()

    log("DONE  rows=%d  skipped_out_of_window=%d  elapsed=%.0fs" % (
        n_rows, n_skip_window, time.time() - t0))
    log("  mean q_gt=%.4f  q_eagle=%.4f  q_suffix=%.4f  (expect q_gt >> others)"
        % (q_gt_sum / max(q_gt_n, 1), q_e_sum / max(q_e_n, 1),
           q_s_sum / max(q_s_n, 1)))
    log(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
