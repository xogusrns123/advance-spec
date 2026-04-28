#!/usr/bin/env python3
"""Per-step diagnostic: are eagle3 and suffix complementary?

Hypothesis (motivating extension):
  extension wins over best-of-singles only when, at the SAME step, eagle3 and
  suffix accept *different* token patterns. If both always accept similar
  amounts (high correlation, low divergence), then `single:eagle3` or
  `single:suffix` alone matches extension's accept count and the extra
  draft cost of extension is wasted.

Metrics (per workload × reslice × budget):
  synergy_ratio       = mean(a_ext) / mean(max(a_e, a_s))
                          1.0 → no synergy; >1.0 → extension exceeds best-single
  synergy_step_frac   = fraction of steps where a_ext > max(a_e, a_s)
  accept_correlation  = Pearson corr between a_e_t and a_s_t across steps
                          high → redundant patterns; low/neg → complementary
  disagreement_score  = mean(|a_e - a_s|) / max(mean(a_e), mean(a_s))
  frac_both_zero / frac_only_eagle3 / frac_only_suffix / frac_both_pos

Each step uses a SEPARATE cache for eagle3, suffix, and extension — so each
method follows its own cache-state trajectory (matches the simulator's
per-method local_cache).

Usage:
  python3 -m simulation.evaluation.analyze_extension_synergy \\
      --agent-results .../agent_results_eagle3.json \\
      --dataset .../dataset.jsonl \\
      --model Qwen/Qwen3-14B \\
      --budget 64 --steps 2 --topk 16 --F 4.0 --T 0.0 \\
      --output /tmp/synergy_<workload>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.pipeline.assemble_records import assemble_records_from_artifacts
from simulation.evaluation.run_tree_oracle_sim import (
    _live_suffix_draft,
    _extension_step,
)
from simulation.evaluation.tree_knapsack import greedy_tree_walk
from hybrid_spec_decoding.suffix_decoding.suffix_tree import SuffixDecodingCache


def _eagle3_alone_accept(rec: dict, budget: int) -> int:
    """Greedy walk on eagle3's tree alone, capped at budget nodes."""
    e3 = rec.get("per_proposer", {}).get("eagle3") or {}
    tids = e3.get("token_ids") or []
    pids = e3.get("parents") or []
    if not tids:
        return 0
    if len(tids) > budget:
        # Truncate to first budget nodes (matches simulator's _proposer_tree_walk)
        tids = tids[:budget]
        pids = pids[:budget]
    gt = rec.get("ground_truth_future") or []
    return greedy_tree_walk(tids, pids, gt)


def per_step_diagnostic(
    records: List[dict], budget: int, F: float, T: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay records 3 times with independent caches; capture per-step accepts."""
    # Group by sequence
    sequences = {}
    for r in records:
        key = (r["request_id"], r["call_idx"])
        sequences.setdefault(key, []).append(r)

    # Build 3 independent caches (eagle3 doesn't use cache, but we need
    # separate suffix + extension caches because their trajectories diverge).
    cache_s = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
    cache_ext = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)

    a_e: List[int] = []
    a_s: List[int] = []
    a_ext: List[int] = []

    for (req_id, call_idx), recs in sequences.items():
        recs.sort(key=lambda r: r["step_idx"])
        cache_id = f"{req_id}_{call_idx}"
        first = recs[0]
        prompt = first.get("context_token_ids") or []
        prompt_np = np.array(prompt, dtype=np.int32)
        cache_s.start_request(cache_id, prompt_np)
        cache_ext.start_request(cache_id, prompt_np)

        for rec in recs:
            gt = rec.get("ground_truth_future") or []
            if not gt:
                continue

            # 1) eagle3 alone (no cache)
            ae = _eagle3_alone_accept(rec, budget)

            # 2) suffix alone — own cache trajectory
            ctx = rec.get("context_token_ids") or []
            s_tids, s_pids, _ = _live_suffix_draft(
                cache_s, cache_id, ctx,
                max_spec_factor=F, min_token_prob=T)
            if s_tids:
                as_ = greedy_tree_walk(s_tids, s_pids, gt)
            else:
                as_ = 0

            # 3) extension — own cache trajectory
            ax, _ = _extension_step(
                rec, budget, cache_ext, cache_id,
                base_proposer="eagle3",
                suffix_max_spec_factor=F, suffix_min_token_prob=T,
                suffix_max_spec_tokens=0)

            a_e.append(ae)
            a_s.append(as_)
            a_ext.append(ax)

            # advance each cache by its own accept+1
            if as_ + 1 <= len(gt):
                cache_s.add_active_response(cache_id, gt[:as_ + 1])
            if ax + 1 <= len(gt):
                cache_ext.add_active_response(cache_id, gt[:ax + 1])

        cache_s.stop_request(cache_id)
        cache_ext.stop_request(cache_id)

    return np.array(a_e), np.array(a_s), np.array(a_ext)


def aggregate(a_e: np.ndarray, a_s: np.ndarray, a_ext: np.ndarray) -> dict:
    n = len(a_e)
    if n == 0:
        return {"n_steps": 0}
    max_single = np.maximum(a_e, a_s)
    sum_single = a_e + a_s
    mean_max = max_single.mean()
    mean_sum = sum_single.mean()
    me, ms = a_e.mean(), a_s.mean()

    # Pearson correlation
    if a_e.std() > 0 and a_s.std() > 0:
        corr = float(np.corrcoef(a_e, a_s)[0, 1])
    else:
        corr = 0.0

    return {
        "n_steps": n,
        "mean_a_eagle3": float(me),
        "mean_a_suffix": float(ms),
        "mean_a_extension": float(a_ext.mean()),
        "mean_max_single": float(mean_max),
        "mean_sum_single": float(mean_sum),
        # Synergy
        "synergy_ratio": float(a_ext.mean() / max(mean_max, 1e-6)),
        "synergy_step_frac": float((a_ext > max_single).mean()),
        # Disagreement / correlation
        "accept_correlation": corr,
        "disagreement_score": float(
            np.abs(a_e - a_s).mean() / max(max(me, ms), 1e-6)),
        # Joint distribution
        "frac_both_zero": float(((a_e == 0) & (a_s == 0)).mean()),
        "frac_only_eagle3": float(((a_e > 0) & (a_s == 0)).mean()),
        "frac_only_suffix": float(((a_e == 0) & (a_s > 0)).mean()),
        "frac_both_pos": float(((a_e > 0) & (a_s > 0)).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent-results", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--responses", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--F", type=float, default=4.0)
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-per-step",
                    action="store_true",
                    help="omit per-step arrays (smaller output)")
    args = ap.parse_args()

    print(f"Loading capture: {args.agent_results}", file=sys.stderr)
    t0 = time.time()
    records = assemble_records_from_artifacts(
        agent_results_path=args.agent_results,
        dataset_path=args.dataset,
        responses_path=args.responses,
        model=args.model,
        eagle3_reslice=(8, 16, args.steps, args.topk),
    )
    print(f"  records: {len(records)} ({time.time() - t0:.1f}s)",
          file=sys.stderr)

    print(f"Per-step diagnostic (budget={args.budget}, F={args.F}, T={args.T})",
          file=sys.stderr)
    t1 = time.time()
    a_e, a_s, a_ext = per_step_diagnostic(records, args.budget, args.F, args.T)
    print(f"  done in {time.time() - t1:.1f}s, n_steps={len(a_e)}",
          file=sys.stderr)

    stats = aggregate(a_e, a_s, a_ext)

    out = {
        "metadata": {
            "agent_results": args.agent_results,
            "dataset": args.dataset,
            "model": args.model,
            "budget": args.budget,
            "reslice": f"s{args.steps}k{args.topk}",
            "F": args.F, "T": args.T,
        },
        "stats": stats,
    }
    if not args.no_per_step:
        out["per_step"] = {
            "a_eagle3": a_e.tolist(),
            "a_suffix": a_s.tolist(),
            "a_extension": a_ext.tolist(),
        }

    with open(args.output, "w") as f:
        json.dump(out, f)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
