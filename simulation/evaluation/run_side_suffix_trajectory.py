"""Side experiment — suffix-driven trajectory counterfactual accept dump.

For every request in Stage 1 (EAGLE3 oracle-vanilla) artifacts, walk the
decoding steps with advancement driven by ``single:suffix``'s accept
count (``step += suffix_acc + 1``). At each visited step, measure the
*counterfactual* accept length each of the following methods would
produce **at that same step_idx context**:

  * ``eagle3``   — budget-truncated EAGLE3 tree, greedy walk vs gt.
  * ``suffix``   — full suffix tree (never budget-truncated), greedy walk.
                   Drawn live from ``SuffixDecodingCache.speculate()``.
  * ``extension`` — EAGLE3 base (budget-truncated) + suffix extensions at
                    every anchor node including root (node_idx=0). Returns
                    a (backbone, extension) accept split via the existing
                    ``_extension_step._last_accepted_{base,suffix}``
                    side-channel attributes.

Per-step rows are dumped to a new JSONL. The notebook
``simulation/notebooks/side_suffix_trajectory.ipynb`` consumes it.

This script NEVER writes to ``tree_oracle_sim.json`` or any other existing
pipeline output. Output directory is ``simulation/results/side_suffix_trajectory/``.

Usage:
    python3 -m simulation.evaluation.run_side_suffix_trajectory \\
        --agent-results simulation/results/qwen3_8b/bfcl_v4/agent_results_eagle3.json \\
        --model Qwen/Qwen3-8B \\
        --budget 64 \\
        --output simulation/results/side_suffix_trajectory/qwen3_8b/bfcl_v4/B64/per_step.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from simulation.evaluation.run_tree_oracle_sim import (
    _extension_step,
    _proposer_tree_walk,
)
from simulation.evaluation.tree_knapsack import greedy_tree_walk


def _group_by_request(records):
    """Group records into {request_id: {call_idx: {step_idx: record}}}."""
    by_req: dict = defaultdict(lambda: defaultdict(dict))
    for r in records:
        by_req[r["request_id"]][r.get("call_idx", 0)][r.get("step_idx", 0)] = r
    return by_req


def _extract_prompt(steps: dict) -> np.ndarray:
    """Return the prompt token ids as int32 numpy array.

    Uses step_idx=0's context_token_ids when available (it equals the
    prompt since no tokens have been decoded yet). Otherwise reconstructs
    from the earliest step's context by chopping off the decoded prefix
    — but we don't have the prefix tokens here, so fall back to empty.
    Empty prompt is safe: the SuffixDecodingCache receives the full
    context via ``speculate(..., ext_context)`` where ``ext_context``
    already includes the prompt.
    """
    if 0 in steps:
        return np.array(steps[0]["context_token_ids"], dtype=np.int32)
    return np.array([], dtype=np.int32)


def _walk_tree_size(per_proposer: dict, name: str, budget: int) -> int:
    """Return the tree size this proposer would present at this step,
    matching ``_proposer_tree_walk`` truncation semantics."""
    tree = per_proposer.get(name) or {}
    tids = tree.get("token_ids") or []
    if not tids:
        return 0
    return min(budget, len(tids))


def _live_suffix_walk(cache, cache_req_id, context, gt):
    """Live suffix counterfactual: speculate on the given context and
    greedy-walk against ground truth. Returns (accepted, tree_size)."""
    ctx_np = np.asarray(context, dtype=np.int32)
    try:
        draft = cache.speculate(
            cache_req_id, ctx_np,
            max_spec_tokens=256, max_spec_factor=4.0,
            min_token_prob=0.0, use_tree_spec=True,
        )
    except Exception:
        return 0, 0
    if not draft.token_ids:
        return 0, 0
    tids = list(draft.token_ids)
    pids = list(draft.parents)
    acc = greedy_tree_walk(tids, pids, gt)
    return acc, len(tids)


def _collect_per_step(records, budget: int, *, verify: bool = False):
    """Iterate records request-by-request, drive advancement by suffix accept,
    emit per-step counterfactual rows.

    Returns (per_step_rows, stats).
    """
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    cache = SuffixDecodingCache(
        max_tree_depth=64, max_cached_requests=100000)

    per_step_rows: list[dict] = []
    stats = {
        "n_requests": 0,
        "n_calls": 0,
        "n_steps_visited": 0,
        "n_steps_with_ext_growth": 0,  # runtime proxy: ext_tree_size_total > ext_base_size
    }

    by_req = _group_by_request(records)
    seq_req_id = 0

    for bfcl_id, calls in by_req.items():
        stats["n_requests"] += 1
        for call_idx, steps in calls.items():
            if not steps:
                continue
            stats["n_calls"] += 1
            prompt = _extract_prompt(steps)
            cache.start_request(seq_req_id, prompt)

            step = min(steps.keys())
            while step in steps:
                rec = steps[step]
                gt = rec.get("ground_truth_future") or []
                if not gt:
                    break
                per_proposer = rec.get("per_proposer") or {}

                # eagle3 counterfactual
                e_acc = _proposer_tree_walk(
                    per_proposer, "eagle3", gt, budget)
                e_size = _walk_tree_size(per_proposer, "eagle3", budget)

                # suffix counterfactual (drives trajectory).
                # Drawn live from the shared SuffixDecodingCache so state
                # matches what _extension_step observes for this step.
                ctx = rec.get("context_token_ids") or []
                s_acc, s_size = _live_suffix_walk(cache, seq_req_id, ctx, gt)

                # extension (eagle3 backbone + suffix extensions at every
                # anchor node including root). Uses live SuffixDecodingCache.
                # Total accept is ext_acc_base + ext_acc_sfx — captured via
                # the split fields below — so we drop the total here.
                _, ext_size_total = _extension_step(
                    rec, budget, cache, seq_req_id, base_proposer="eagle3")
                ext_base = int(_extension_step._last_accepted_base)
                ext_sfx = int(_extension_step._last_accepted_suffix)
                ext_base_size = int(_extension_step._last_base_size)

                if ext_size_total > ext_base_size:
                    stats["n_steps_with_ext_growth"] += 1

                # Advance by SUFFIX accept (+ bonus committed token).
                commit = min(s_acc + 1, len(gt))

                per_step_rows.append({
                    "request_id": bfcl_id,
                    "call_idx": int(call_idx),
                    "step_idx": int(step),
                    "eagle3_acc": int(e_acc),
                    "eagle3_tree_size": int(e_size),
                    "suffix_acc": int(s_acc),
                    "suffix_tree_size": int(s_size),
                    "ext_acc_base": ext_base,
                    "ext_acc_sfx": ext_sfx,
                    "ext_base_size": ext_base_size,
                    "ext_tree_size_total": int(ext_size_total),
                    "advance": int(commit),
                })
                stats["n_steps_visited"] += 1

                # Warm cache with committed gt tokens. Feed one-at-a-time
                # to mirror the per-step cache accumulation used by Stage 3.
                for t in gt[:commit]:
                    cache.add_active_response(seq_req_id, [int(t)])

                step += commit

                if verify and stats["n_steps_visited"] >= 5:
                    break

            cache.stop_request(seq_req_id)
            seq_req_id += 1

            if verify:
                break
        if verify:
            break

    return per_step_rows, stats


def _slice_records_by_request(records, req_start: Optional[int],
                              req_end: Optional[int]):
    """Slice records at the request-level (not record-level)."""
    if req_start is None and req_end is None:
        return records
    # Preserve original request order as encountered.
    seen = []
    order = {}
    for r in records:
        rid = r["request_id"]
        if rid not in order:
            order[rid] = len(seen)
            seen.append(rid)
    a = req_start or 0
    b = req_end if req_end is not None else len(seen)
    kept = set(seen[a:b])
    return [r for r in records if r["request_id"] in kept]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-results", required=True,
                        help="Stage 1 EAGLE3 oracle-vanilla JSON path.")
    parser.add_argument("--model", required=True,
                        help="Target model name (for tokenizer).")
    parser.add_argument("--budget", type=int, required=True,
                        help="EAGLE3 base budget (tree size cap). Required; "
                             "no implicit default.")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path for per-step rows.")
    parser.add_argument("--dataset", default=None,
                        help="dataset.jsonl for BFCL/SpecBench prompt "
                             "reconstruction (if assemble_records needs it).")
    parser.add_argument("--responses", default=None,
                        help="agent_results_responses.json (BFCL only).")
    parser.add_argument("--exclude", default=None,
                        help="Exclude-ids file.")
    parser.add_argument("--req-start", type=int, default=None,
                        help="Slice requests at [req_start:req_end).")
    parser.add_argument("--req-end", type=int, default=None)
    parser.add_argument("--verify", action="store_true",
                        help="Smoke mode: process one request/call, up to 5 "
                             "steps, then assert extension grew the tree. "
                             "Also prints the static source reference that "
                             "confirms root-node extension.")
    args = parser.parse_args()

    # Static reference: every base-tree node is an anchor for suffix
    # extension (virtual root + for node_idx in range(n)) inside
    # _extension_step(). Printed so readers of --verify output know why we
    # expect ext_tree_size_total > ext_base_size at most steps.
    print(
        "[prereq] _extension_step() grafts suffix at the virtual root "
        "and at every base-tree node.",
        file=sys.stderr)

    from simulation.pipeline.assemble_records import (
        assemble_records_from_artifacts,
    )

    t0 = time.time()
    records = assemble_records_from_artifacts(
        agent_results_path=args.agent_results,
        suffix_drafts_path=None,
        draft_model_drafts_path=None,
        mtp_agent_results_path=None,
        exclude_path=args.exclude,
        model=args.model,
        dataset_path=args.dataset,
        responses_path=args.responses,
    )
    print(f"Assembled {len(records)} step records in "
          f"{time.time() - t0:.2f}s", file=sys.stderr)

    records = _slice_records_by_request(records, args.req_start, args.req_end)
    print(f"After slice [{args.req_start}:{args.req_end}): "
          f"{len(records)} records", file=sys.stderr)

    t0 = time.time()
    per_step_rows, stats = _collect_per_step(
        records, args.budget, verify=args.verify)
    print(f"Collected {len(per_step_rows)} per-step rows in "
          f"{time.time() - t0:.2f}s", file=sys.stderr)

    if args.verify:
        # Runtime proxy for "some extension happened". Source reference
        # above remains the authoritative guarantee for root specifically.
        if stats["n_steps_with_ext_growth"] == 0:
            raise AssertionError(
                "--verify: no step produced ext_tree_size_total > "
                "ext_base_size in the first 5 steps; either the suffix "
                "cache is not warming or the extension path is broken.")
        print(
            f"[verify] OK — {stats['n_steps_with_ext_growth']} of "
            f"{stats['n_steps_visited']} visited steps had extension "
            "growth (ext_tree_size_total > ext_base_size).",
            file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in per_step_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote per-step rows: {output_path}", file=sys.stderr)

    meta = {
        "script": "run_side_suffix_trajectory",
        "agent_results_path": str(Path(args.agent_results).resolve()),
        "model": args.model,
        "budget": args.budget,
        "req_start": args.req_start,
        "req_end": args.req_end,
        "dataset_path": args.dataset,
        "responses_path": args.responses,
        "verify": bool(args.verify),
        "n_requests": stats["n_requests"],
        "n_calls": stats["n_calls"],
        "n_steps_visited": stats["n_steps_visited"],
        "n_steps_with_ext_growth": stats["n_steps_with_ext_growth"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = output_path.with_name("meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote meta: {meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
