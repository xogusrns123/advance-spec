"""Offline suffix-TAIL simulation for chain-hybrid arms (route b), model-free.

The per-depth SELECTION needs the model (real serving), but the suffix TAIL
appended BEYOND the EAGLE/MTP chain is suffix-only (a trie walk, no model) — so
it can be simulated offline, lifting sglang's static-mamba-cache cap (which
blocks the real tail on Qwen3.5/27B). For a fair comparison suffix must be
allowed its natural (history-based) match length (~32), not capped at 16.

For each chain-hybrid arm in a tail=0 run, per step:
  - real chain accept_len (from that arm's decision-log step records)
  - if the chain fully accepted (accept_len == steps): walk the SuffixDecodingCache
    from the chain end and count its continued match vs GT (capped at --tmax) =
    the tail it WOULD have appended. new_accept = steps + tail.
  - else: unchanged (the chain broke before the tail point).
GT per request = gt_tokens.jsonl output_ids (contiguous), order-aligned to the
arm's rids (oracle arm sanity-checks the alignment: gt_token@(step0,depth0) ==
output_ids[0]). MAT_tail is an ESTIMATE: it adds the per-step tail tokens over the
tail=0 step chunking (no re-chunk), so it is a slight LOWER bound on a real tail
run's MAT (which would also need fewer steps).

Usage:
  python3 simulation/scripts/sim_tail.py --dir <tail0 run dir> --tmax 32 \
      --arms select1,select1_calib_isotonic,select1_oracle
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.evaluation.tree_knapsack import greedy_tree_walk  # noqa: E402


def load_gt(path):
    prompts, outs = [], []
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        prompts.append(r.get("input_ids") or [])
        outs.append(r["output_ids"])
    return prompts, outs


def arm_steps(path):
    """rid -> ({decode_step: accept_len}, {decode_step: {depth: committed_token}}).
    committed = chosen proposer's token (suffix if chosen==suffix else eagle); for
    accepted depths these equal the GT tokens, used to content-align to gt_tokens."""
    steps = defaultdict(dict)
    committed = defaultdict(lambda: defaultdict(dict))
    order, seen = [], set()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rid = r.get("rid")
        if rid and rid not in seen:
            seen.add(rid); order.append(rid)
        if r.get("type") == "step":
            steps[rid][int(r["decode_step"])] = int(r["accept_len"])
        elif r.get("type") == "decision" and not r.get("tail"):
            tok = (r.get("suffix_token") if r.get("chosen") == "suffix"
                   else r.get("eagle_token"))
            if tok is not None:
                committed[rid][int(r["decode_step"])][int(r["depth"])] = int(tok)
    return order, steps, committed


def align_to_gt(order, steps, committed, gts):
    """rid -> gt index, by POSITION: reconstruct {abs_position: committed_token}
    (position = L_k+depth, L advances by accept_len+1 per step; accepted draft
    tokens equal GT) and find the gt entry whose output_ids agree at those
    positions. Robust to bonus-token gaps and accept_len==0 first steps."""
    used, rid2gi, n_ok = set(), {}, 0
    for rid in order:
        L = 1  # output_ids[0] is the prefill token; decode/draft starts at pos 1
        pos_tok = {}
        for ds in sorted(steps[rid]):
            al = steps[rid][ds]
            cd = committed[rid].get(ds, {})
            for d in range(al):
                if d in cd:
                    pos_tok[L + d] = cd[d]
            L += al + 1
        if not pos_tok:
            continue
        # NOTE only the oracle arm (output forced to GT) aligns/positions cleanly
        # throughout; raw/calib outputs FP-diverge from gt_tokens so deep
        # positions drift -> their tail is unreliable (treat as oracle-only).
        sample = sorted(pos_tok)[:8]
        best = None
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            if all(p < len(g) and g[p] == pos_tok[p] for p in sample):
                best = gi
                break
        if best is not None:
            rid2gi[rid] = best
            used.add(best)
            n_ok += 1
    return rid2gi, n_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arms", default="select1,select1_calib_histogram,"
                    "select1_calib_isotonic,select1_calib_logistic,"
                    "select1_calib_beta,select1_oracle")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--tmax", type=int, default=32)
    args = ap.parse_args()
    D = Path(args.dir)
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    prompts, gts = load_gt(D / "gt_tokens.jsonl")
    print(f"gt_tokens requests={len(gts)}")
    out = {}
    for arm in args.arms.split(","):
        dl = D / f"decisions_{arm}.jsonl"
        if not dl.exists():
            print(f"  {arm}: no decision log, skip"); continue
        order, steps, committed = arm_steps(dl)
        rid2gi, n_ok = align_to_gt(order, steps, committed, gts)
        cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
        real, tailed = [], []
        for i, rid in enumerate(order):
            if rid not in rid2gi:
                continue
            gt = gts[rid2gi[rid]]
            prompt = prompts[rid2gi[rid]]
            # seed trie with the PROMPT (tool results/context drive verbatim
            # suffix matches) + the prefill token, mirroring real serving.
            cache.start_request(i, np.asarray(list(prompt) + gt[:1], dtype=np.int32))
            L = 1  # output_ids[0] = prefill token; decode starts at pos 1
            for k in sorted(steps[rid]):
                al = steps[rid][k]
                # warm trie with the chain tokens committed up to the chain end
                end = min(L + al, len(gt))
                if al >= args.steps and L + args.steps < len(gt):
                    ctx = np.asarray(list(prompt) + gt[:L + args.steps],
                                     dtype=np.int32)
                    try:
                        draft = cache.speculate(
                            i, ctx, max_spec_tokens=args.tmax, max_spec_factor=4.0,
                            min_token_prob=0.0, use_tree_spec=True)
                        t = (greedy_tree_walk(list(draft.token_ids),
                                              list(draft.parents),
                                              gt[L + args.steps:])
                             if draft.token_ids else 0)
                    except Exception:
                        t = 0
                    t = min(t, args.tmax)
                    real.append(al); tailed.append(al + t)
                else:
                    real.append(al); tailed.append(al)
                # advance the (real, tail=0) trajectory + warm trie with committed
                if end > L:
                    cache.add_active_response(i, [int(x) for x in gt[L:end]])
                L += al + 1
            cache.stop_request(i)
        real = np.asarray(real, float); tailed = np.asarray(tailed, float)
        out[arm] = (real.mean(), tailed.mean(), len(real))
        print(f"  {arm:26s} MAT(no-tail)={real.mean():.3f}  "
              f"MAT(+tail<= {args.tmax})~{tailed.mean():.3f}  "
              f"(+{tailed.mean()-real.mean():.3f})  steps={len(real)}  "
              f"aligned {n_ok}/{len(order)}")
    json.dump({a: {"mat_notail": v[0], "mat_tail_sim": v[1], "n": v[2]}
               for a, v in out.items()},
              open(D / "tail_sim.json", "w"), indent=2)
    print(f"wrote {D/'tail_sim.json'}")


if __name__ == "__main__":
    main()
