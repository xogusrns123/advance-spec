"""Standalone per-position accept-rate computation.

Walks the FULL token trajectory for every (request, call) and measures, at
every target-token position, whether each proposer's draft tree contains
the next ground-truth token.

What's different from the in-sim pre-pass:
  - Iterate every position in the actual decode trajectory (not just the
    `records` whose proposer-tree dict is non-empty). Positions skipped by
    the record builder still increment the denominator, so deep-position
    rates aren't biased upward by a smaller denominator.
  - For SUFFIX: the cache is advanced by EXACTLY ONE token per position
    (the verified bonus token) — never by `_adv` ≥ 2 tokens at a time.
    Ensures cache state stays "no draft was actually accepted" so the
    measurement reflects what suffix could propose given only the verified
    trajectory, not the inflated state where multi-token accepts pre-train
    the trie on the future.
  - SUFFIX is also speculated at intermediate positions inside step-idx
    gaps (rare ~0.1% of positions), giving full per-position coverage.

Output JSON has the same `position_accepts` schema as the simulator:
    {
      "metadata": {...},
      "position_accepts": {
        "max_position": <int>,
        "by_proposer": {
          "<name>": {"seq_accept": [...], "ind_accept": [...], "depth_ge": [...]}
        }
      }
    }

Usage (one workload):
    python3 -m simulation.scripts.compute_position_accepts \\
        --agent-results .../agent_results_eagle3.json \\
        --dataset .../dataset.jsonl \\
        --draft-model-drafts .../draft_model_drafts.jsonl \\
        --reslice-steps 8 --reslice-topk 8 \\
        --capture-steps 8 --capture-topk 16 \\
        --output simulation/results/explorations/posacc_<wl>_s8k8.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.evaluation.tree_knapsack import position_accept_rates  # noqa: E402
from simulation.pipeline.assemble_records import (  # noqa: E402
    assemble_records_from_artifacts,
)


# Live-suffix speculate parameters — match the existing simulator pre-pass.
SUFFIX_FACTOR = 4.0
SUFFIX_MIN_PROB = 0.0
SUFFIX_MAX_TOKENS = 256
SUFFIX_USE_TREE = True


def _accumulate(stats: Dict[str, List[int]], seq, ind,
                cond_acc, cond_dn, denom_depth, max_pos):
    """Increment denom for every position in [0, denom_depth) regardless of
    whether the draft tree was deep enough — same convention as the
    in-sim pre-pass. seq/ind are 0/1 lists of length max_pos.
    cond_denom only increments when pos d-1 was accepted, so
    cond_rate[d] = cond_accept[d] / cond_denom[d] = P(accept[d] | accept[d-1]).
    """
    for d in range(denom_depth):
        stats["depth_ge"][d] += 1
        stats["seq_accept"][d] += seq[d]
        stats["ind_accept"][d] += ind[d]
        stats["cond_accept"][d] += cond_acc[d]
        stats["cond_denom"][d] += cond_dn[d]


def _empty_stats(max_pos: int) -> Dict[str, List[int]]:
    return {
        "seq_accept": [0] * max_pos,
        "ind_accept": [0] * max_pos,
        "depth_ge": [0] * max_pos,
        "cond_accept": [0] * max_pos,
        "cond_denom": [0] * max_pos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-results", required=True)
    ap.add_argument("--dataset", default=None,
                    help="Required for BFCL prompt reconstruction; specbench "
                         "and swebench can usually omit.")
    ap.add_argument("--responses", default=None)
    ap.add_argument("--draft-model-drafts", default=None)
    ap.add_argument("--model", default=None,
                    help="Tokenizer name; only used by BFCL prompt rebuild.")
    ap.add_argument("--reslice-steps", type=int, default=None)
    ap.add_argument("--reslice-topk", type=int, default=None)
    ap.add_argument("--capture-steps", type=int, default=8)
    ap.add_argument("--capture-topk", type=int, default=16)
    ap.add_argument("--max-position", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--exclude", default=None)
    args = ap.parse_args()

    eagle3_reslice = None
    if args.reslice_steps and args.reslice_topk:
        eagle3_reslice = (args.capture_steps, args.capture_topk,
                          args.reslice_steps, args.reslice_topk)

    print(f"[posacc] loading capture: {args.agent_results}", file=sys.stderr)
    t0 = time.time()
    records = assemble_records_from_artifacts(
        agent_results_path=args.agent_results,
        suffix_drafts_path=None,
        draft_model_drafts_path=args.draft_model_drafts,
        mtp_agent_results_path=None,
        exclude_path=args.exclude,
        model=args.model,
        dataset_path=args.dataset,
        responses_path=args.responses,
        eagle3_reslice=eagle3_reslice,
    )
    print(f"[posacc] {len(records)} records in {time.time() - t0:.1f}s",
          file=sys.stderr)

    max_pos = args.max_position

    # Group by (request, call) and sort by step_idx.
    by_seq: Dict[tuple, List[dict]] = defaultdict(list)
    for rec in records:
        by_seq[(rec["request_id"], rec.get("call_idx", 0))].append(rec)
    for k in by_seq:
        by_seq[k].sort(key=lambda r: r.get("step_idx", 0))

    # Stats per proposer.
    stats: Dict[str, Dict[str, List[int]]] = {
        "eagle3": _empty_stats(max_pos),
        "draft_model": _empty_stats(max_pos),
        "suffix": _empty_stats(max_pos),
    }

    # Live suffix cache for pre-pass B (bonus-only update).
    try:
        from hybrid_spec_decoding.suffix_decoding.suffix_tree import (
            SuffixDecodingCache,
        )
    except Exception as e:
        print(f"[posacc] FATAL: cannot import SuffixDecodingCache: {e}",
              file=sys.stderr)
        sys.exit(2)

    sx = SuffixDecodingCache(
        max_tree_depth=max_pos, max_cached_requests=100000)

    n_seqs = 0
    n_pos_total = 0
    n_pos_no_eg = 0
    n_pos_no_dm = 0
    n_suffix_speculates = 0
    n_gap_intermediate = 0
    last_log = time.time()
    seq_keys = list(by_seq.keys())

    for ki, (rid, cid) in enumerate(seq_keys):
        seq = by_seq[(rid, cid)]
        if not seq:
            continue
        n_seqs += 1
        cache_key = f"{rid}_{cid}"
        prompt_ctx = seq[0].get("context_token_ids") or []
        sx.start_request(cache_key, np.asarray(prompt_ctx, dtype=np.int32))

        # Build the "running context" we will feed to speculate. Maintained
        # explicitly so we can speculate at intermediate (gap) positions
        # using the right context.
        running_ctx = list(prompt_ctx)

        for ri, rec in enumerate(seq):
            gt = rec.get("ground_truth_future") or []
            if not gt:
                continue
            step = rec.get("step_idx", 0)
            # How many positions to cover until the next record (or end).
            if ri < len(seq) - 1:
                adv = seq[ri + 1].get("step_idx", 0) - step
            else:
                adv = 1
            if adv < 1:
                adv = 1

            # ── Position 0 within this record's gap (j=0): the recorded
            # position. Use captured eagle3 / draft_model trees.
            for j in range(adv):
                future_at_pos = gt[j:]
                if not future_at_pos:
                    break
                n_pos_total += 1

                # Helper to call position_accept_rates with optional
                # captured tree. When no tree is provided we still call the
                # function with empty token_ids/parents — it returns the
                # correct "all rejected" pattern (cond_denom[0]=1,
                # everything else 0) so cond denominators behave as if pos 1
                # was evaluated and rejected.
                def _eval(tids, pids):
                    sv, iv, cav, cdv, dn = position_accept_rates(
                        tids or [], pids or [], future_at_pos, max_pos)
                    if dn > 0:
                        _accumulate(stats[name_for_eval], sv, iv,
                                    cav, cdv, dn, max_pos)

                if j == 0:
                    # Captured proposers (eagle3, draft_model) only at the
                    # recorded position.
                    pp = rec.get("per_proposer") or {}
                    for prop_name in ("eagle3", "draft_model"):
                        prop = pp.get(prop_name)
                        name_for_eval = prop_name
                        if prop and prop.get("token_ids"):
                            _eval(prop["token_ids"], prop["parents"])
                        else:
                            _eval(None, None)
                            if prop_name == "eagle3":
                                n_pos_no_eg += 1
                            else:
                                n_pos_no_dm += 1
                else:
                    # Intermediate (gap) position — no captured tree.
                    n_gap_intermediate += 1
                    for prop_name in ("eagle3", "draft_model"):
                        name_for_eval = prop_name
                        _eval(None, None)

                # ── SUFFIX: live cache speculate at this position (every
                # position, including gaps), then advance cache by exactly
                # one bonus token.
                name_for_eval = "suffix"
                ctx_arr = np.asarray(
                    running_ctx[-max_pos:], dtype=np.int32)
                try:
                    draft = sx.speculate(
                        cache_key, ctx_arr,
                        max_spec_factor=SUFFIX_FACTOR,
                        min_token_prob=SUFFIX_MIN_PROB,
                        max_spec_tokens=SUFFIX_MAX_TOKENS,
                        use_tree_spec=SUFFIX_USE_TREE,
                    )
                    n_suffix_speculates += 1
                    if draft.token_ids:
                        _eval(list(draft.token_ids), list(draft.parents))
                    else:
                        _eval(None, None)
                except Exception:
                    pass  # best-effort

                # Advance cache by exactly ONE token (bonus = real next).
                bonus = gt[j]
                sx.add_active_response(cache_key, [int(bonus)])
                running_ctx.append(int(bonus))

        sx.stop_request(cache_key)

        if time.time() - last_log > 30:
            print(f"[posacc] {ki + 1}/{len(seq_keys)} seqs, "
                  f"{n_pos_total} positions, "
                  f"{n_suffix_speculates} suffix specs, "
                  f"{n_gap_intermediate} gap intermediates",
                  file=sys.stderr)
            last_log = time.time()

    elapsed = time.time() - t0
    print(f"[posacc] DONE — {n_seqs} seqs, {n_pos_total} positions in "
          f"{elapsed:.1f}s. eagle3 missing trees: {n_pos_no_eg}, "
          f"draft_model missing trees: {n_pos_no_dm}, "
          f"intermediate (gap) positions: {n_gap_intermediate}",
          file=sys.stderr)

    out = {
        "metadata": {
            "input_source": args.agent_results,
            "draft_model_drafts": args.draft_model_drafts,
            "max_position": max_pos,
            "reslice": (None if not eagle3_reslice
                        else {"S": eagle3_reslice[0],
                              "K": eagle3_reslice[1],
                              "s": eagle3_reslice[2],
                              "k": eagle3_reslice[3]}),
            "n_sequences": n_seqs,
            "n_positions": n_pos_total,
            "n_gap_intermediate": n_gap_intermediate,
            "elapsed_sec": elapsed,
        },
        "position_accepts": {
            "max_position": max_pos,
            "by_proposer": stats,
            "_doc": ("Accept rate at position d = "
                     "seq_accept[d-1]/depth_ge[d-1] (sequential greedy "
                     "walk) or ind_accept[d-1]/depth_ge[d-1] "
                     "(independent depth-d match). Suffix cache was "
                     "advanced by exactly one bonus token per position "
                     "(no draft-acceptance pollution). Every position in "
                     "the trajectory contributes to the denominator."),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[posacc] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
