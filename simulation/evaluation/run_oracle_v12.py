#!/usr/bin/env python3
"""Standalone oracle v1/v2 simulator (independent cache per method × params).

Three new "oracle" methods:

  1. extension_oracle_v1 (per-step budget picker):
     For each step, evaluate extension at each B in BUDGETS={1,4,16,64,128}.
     Pick the budget that minimizes target_cost / accepted (= max
     accepted/target_cost). Accepts/cost summed using the picked budget
     per step. cost_b = target_forward(ext_tree_size_b) + eagle3_draft(b).

  2. extension_oracle_v2 (per-step budget picker + perfect verify):
     For each step, evaluate extension at each B in BUDGETS. Pick B with the
     MOST accepted tokens (ties: smaller B for cheaper draft).
     Cost = target_forward(a+1) + eagle3_draft(B). Strictly ≥ v1 by
     construction (v2 charges only accepted, v1 charges accepted-base + full
     traversed graft).

  3. hybrid_oracle_v2 (hybrid_e3 gating + perfect verify):
     If suffix_score >= τ: use suffix tree → cost = target_forward(a_sfx+1)
                                                   + suffix_speculate_ms
     else: fall back to eagle3 truncated to B → cost = target_forward(a_e3+1)
                                                       + eagle3_draft(B)
     Sweep B in BUDGETS and τ in HYBRID_THRESHOLDS.

Each (method, hyperparams) pair runs an independent pass through records,
with its own SuffixDecodingCache and its own skip-by-advance trajectory —
matches run_tree_oracle_sim's per-method local_cache convention.

FT applied to suffix calls: (F, T) ∈ FT_GRID. One run per FT pair.

Usage:
  python3 -m simulation.evaluation.run_oracle_v12 \\
    --agent-results .../agent_results_eagle3.json \\
    --dataset .../dataset.jsonl \\
    --latency-config .../latency_config.json \\
    --model Qwen/Qwen3-14B \\
    --steps 2 --topk 16 \\
    --output /tmp/oracle_v12_<wl>_s2k16.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.pipeline.assemble_records import assemble_records_from_artifacts
from simulation.evaluation.run_tree_oracle_sim import (
    _live_suffix_draft, _extension_step,
)
from simulation.evaluation.tree_knapsack import greedy_tree_walk
from hybrid_spec_decoding.suffix_decoding.suffix_tree import SuffixDecodingCache


BUDGETS = [1, 4, 16, 64, 128]
HYBRID_THRESHOLDS = [1.0, 5.0, 10.0]
FT_GRID = [(1.0, 0.1), (4.0, 0.0)]


# -------- latency helpers --------

def _interp(table: dict, B: int, fallback: float) -> float:
    if not table:
        return fallback
    keys = sorted(int(k) for k in table)
    if B <= keys[0]:
        if B == keys[0]:
            return float(table[str(keys[0])])
        v0 = float(table[str(keys[0])])
        return fallback + (B - 1) / max(keys[0] - 1, 1) * (v0 - fallback)
    if B >= keys[-1]:
        if len(keys) >= 2:
            k_hi, k_lo = keys[-1], keys[-2]
            v_hi = float(table[str(k_hi)])
            v_lo = float(table[str(k_lo)])
            slope = (v_hi - v_lo) / (k_hi - k_lo) if k_hi != k_lo else 0.0
            return v_hi + slope * (B - k_hi)
        return float(table[str(keys[-1])])
    lo = max(k for k in keys if k <= B)
    hi = min(k for k in keys if k >= B)
    if lo == hi:
        return float(table[str(lo)])
    frac = (B - lo) / (hi - lo)
    return float(table[str(lo)]) + frac * (float(table[str(hi)])
                                            - float(table[str(lo)]))


def build_latency(latency_config: dict, steps: int, topk: int):
    vanilla_ms = float(latency_config["vanilla_step_ms"])
    tfwd_by_topk = latency_config.get("target_forward_ms_by_topk", {}) or {}
    if str(topk) in tfwd_by_topk:
        target_fwd = tfwd_by_topk[str(topk)]
    else:
        target_fwd = latency_config.get("target_forward_ms", {})

    e3draft_by_ts = latency_config.get("eagle3_draft_ms_by_topk_steps", {}) or {}
    e3draft_by_s = latency_config.get("eagle3_draft_ms_by_steps", {}) or {}
    if str(topk) in e3draft_by_ts and str(steps) in e3draft_by_ts[str(topk)]:
        eagle3_table = e3draft_by_ts[str(topk)][str(steps)]
    elif str(steps) in e3draft_by_s:
        eagle3_table = e3draft_by_s[str(steps)]
    else:
        eagle3_table = latency_config.get("eagle3_draft_ms", {})

    suffix_speculate_ms = float(latency_config.get("suffix_speculate_ms", 0.0))

    def target_forward(B: int) -> float:
        return _interp(target_fwd, max(B, 1), vanilla_ms)

    def eagle3_draft(B: int) -> float:
        return _interp(eagle3_table, max(B, 1), 0.0)

    return target_forward, eagle3_draft, suffix_speculate_ms, vanilla_ms


# -------- per-method walkers --------

def _build_sequences(records: List[dict]) -> Dict[Tuple[str, int], dict]:
    """Group records into per-(req,call) sequences with positional index."""
    seqs: Dict[Tuple[str, int], dict] = {}
    for r in records:
        key = (r["request_id"], r["call_idx"])
        if key not in seqs:
            seqs[key] = {"by_pos": {}, "step_indices": []}
        pos = r["step_idx"]
        seqs[key]["by_pos"][pos] = r
        seqs[key]["step_indices"].append(pos)
    for v in seqs.values():
        v["step_indices"].sort()
    return seqs


def _seq_len(record_index: dict, step_indices: list) -> int:
    """Match simulator's seq_len logic from run_tree_oracle_sim."""
    first = record_index[step_indices[0]]
    last = record_index[step_indices[-1]]
    first_gt = first.get("gt_len", len(first.get("ground_truth_future", [])))
    last_gt = last.get("gt_len", len(last.get("ground_truth_future", [])))
    if last_gt <= 1:
        return step_indices[0] + 1 + first_gt
    return step_indices[0] + first_gt


def walk_method(
    sequences,
    method_step_fn,
    vanilla_ms: float,
):
    """Walk all sequences with one method. method_step_fn(rec, cache, cache_id)
    → (accepted, step_cost_ms). Maintains independent cache + skip-by-advance.

    Returns dict with totals.
    """
    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
    total_generated = 0
    total_accepted = 0
    total_steps = 0
    total_time_ms = 0.0

    for (req_id, call_idx), seq in sequences.items():
        record_index = seq["by_pos"]
        step_indices = seq["step_indices"]
        if not step_indices:
            continue
        cache_id = f"{req_id}_{call_idx}"
        first_rec = record_index[step_indices[0]]
        prompt = first_rec.get("context_token_ids") or []
        cache.start_request(cache_id, np.array(prompt, dtype=np.int32))

        seq_len = _seq_len(record_index, step_indices)
        max_pos = step_indices[-1]
        step_set = set(step_indices)

        pos = step_indices[0]
        while pos <= max_pos and pos in step_set:
            rec = record_index.get(pos)
            if rec is None:
                total_generated += 1
                total_steps += 1
                total_time_ms += vanilla_ms
                pos += 1
                continue

            accepted, step_cost = method_step_fn(rec, cache, cache_id)
            advance = accepted + 1

            total_generated += advance
            total_accepted += accepted
            total_steps += 1
            total_time_ms += step_cost

            gt = rec.get("ground_truth_future", []) or []
            if gt and advance <= len(gt):
                cache.add_active_response(cache_id, gt[:advance])

            pos += advance

        # Tail at vanilla cost
        remaining = seq_len - pos
        if remaining > 0:
            total_generated += remaining
            total_steps += remaining
            total_time_ms += remaining * vanilla_ms

        cache.stop_request(cache_id)

    vanilla_time_ms = total_generated * vanilla_ms
    return {
        "total_generated": total_generated,
        "total_accepted": total_accepted,
        "total_steps": total_steps,
        "total_time_ms": total_time_ms,
        "vanilla_time_ms": vanilla_time_ms,
        "speedup_real": vanilla_time_ms / total_time_ms if total_time_ms > 0 else 1.0,
        "mat": total_accepted / max(total_steps, 1),
    }


# -------- method step functions --------

def _eagle3_truncated_accept(rec: dict, B: int) -> int:
    """Mirrors run_tree_oracle_sim._proposer_tree_walk('eagle3', ..., budget)."""
    e3 = rec.get("per_proposer", {}).get("eagle3") or {}
    tids = e3.get("token_ids") or []
    pids = e3.get("parents") or []
    if not tids:
        return 0
    if B < len(tids):
        tids = tids[:B]
        # Fix parent refs that pointed beyond truncated range (matches base sim)
        pids = [p if p < B else -1 for p in pids[:B]]
    gt = rec.get("ground_truth_future") or []
    return greedy_tree_walk(tids, pids, gt)


def make_ext_v1_step(target_forward, eagle3_draft, F, T, budgets):
    """Per-step budget picker (oracle: only accepted suffix grafts counted).

    For each B, run _extension_step → (accepted, full_ext_size). The ORACLE
    ext_size = _last_base_size + _last_accepted_suffix (mirrors base sim's
    extension_oracle accounting). target_cost_b = target_forward(oracle_sz_b)
    + eagle3_draft(b). Pick B that maximizes accepted / target_cost (= min
    cost-per-accepted-token).
    """
    def step_fn(rec, cache, cache_id):
        gt = rec.get("ground_truth_future") or []
        if not gt:
            return 0, 0.0
        # Try each budget on SAME cache state (speculate is read-only)
        per_b: Dict[int, Tuple[int, int]] = {}
        for b in budgets:
            a_b, _sz_full = _extension_step(
                rec, b, cache, cache_id,
                base_proposer="eagle3",
                suffix_max_spec_factor=F,
                suffix_min_token_prob=T,
                suffix_max_spec_tokens=0)
            # Oracle accounting: count base + the FULL traversed graft tree
            # (the suffix subtree rooted at the last accepted base node).
            # Matches base sim's extension_oracle accounting after fix.
            oracle_sz = (_extension_step._last_base_size
                         + _extension_step._last_traversed_graft_size)
            per_b[b] = (a_b, oracle_sz)
        # Pick best efficiency
        best_b = budgets[0]
        best_eff = -1.0
        for b in budgets:
            a_b, sz_b = per_b[b]
            cost_b = target_forward(max(sz_b, 1)) + eagle3_draft(b)
            eff = (a_b / cost_b) if a_b > 0 else 0.0
            if eff > best_eff:
                best_eff = eff
                best_b = b
        a_pick, sz_pick = per_b[best_b]
        cost = target_forward(max(sz_pick, 1)) + eagle3_draft(best_b)
        step_fn._last_picked_b = best_b
        return a_pick, cost
    step_fn._last_picked_b = None
    return step_fn


def make_ext_v2_step(target_forward, eagle3_draft, F, T, budgets):
    """Per-step budget picker, accept-only verify cost.

    For each B, run extension. Pick B with MAX accepted tokens (ties: smaller B
    wins → cheaper draft). Cost = target_forward(a+1) + eagle3_draft(B). v2
    is strictly ≥ v1 because v2 charges only accepted (not the traversed
    graft) — both share the per-step budget freedom.
    """
    def step_fn(rec, cache, cache_id):
        gt = rec.get("ground_truth_future") or []
        if not gt:
            return 0, 0.0
        per_b: Dict[int, int] = {}
        for b in budgets:
            a_b, _sz = _extension_step(
                rec, b, cache, cache_id,
                base_proposer="eagle3",
                suffix_max_spec_factor=F,
                suffix_min_token_prob=T,
                suffix_max_spec_tokens=0)
            per_b[b] = a_b
        # Max accept; on ties, smallest B (cheapest draft)
        best_b = max(budgets, key=lambda b: (per_b[b], -b))
        a_pick = per_b[best_b]
        cost = target_forward(a_pick + 1) + eagle3_draft(best_b)
        step_fn._last_picked_b = best_b
        return a_pick, cost
    step_fn._last_picked_b = None
    return step_fn


def make_hybrid_v2_step(target_forward, eagle3_draft, suffix_speculate_ms,
                         F, T, B, tau):
    """Hybrid_e3 gating + accept-only verify."""
    def step_fn(rec, cache, cache_id):
        gt = rec.get("ground_truth_future") or []
        if not gt:
            return 0, 0.0
        ctx = rec.get("context_token_ids") or []
        sfx_tids, sfx_pids, sfx_score = _live_suffix_draft(
            cache, cache_id, ctx,
            max_spec_factor=F, min_token_prob=T)
        use_sfx = (sfx_tids is not None and sfx_score >= tau)
        if use_sfx:
            a = greedy_tree_walk(sfx_tids, sfx_pids, gt)
            cost = target_forward(a + 1) + suffix_speculate_ms
            step_fn._last_used_suffix = 1
        else:
            a = _eagle3_truncated_accept(rec, B)
            cost = target_forward(a + 1) + eagle3_draft(B)
            step_fn._last_used_suffix = 0
        return a, cost
    step_fn._last_used_suffix = 0
    return step_fn


# -------- driver --------

def run_one_ft(records, target_forward, eagle3_draft, suffix_speculate_ms,
               vanilla_ms, F, T, budgets, hybrid_thresholds):
    sequences = _build_sequences(records)
    out = {"F": F, "T": T,
           "extension_oracle_v1": None,
           "extension_oracle_v2": {},
           "hybrid_oracle_v2": {}}

    # ext_v1
    print(f"  ext_v1 ...", end=" ", flush=True)
    t0 = time.time()
    fn_v1 = make_ext_v1_step(target_forward, eagle3_draft, F, T, budgets)
    res_v1 = walk_method(sequences, fn_v1, vanilla_ms)
    out["extension_oracle_v1"] = res_v1
    print(f"spd={res_v1['speedup_real']:.3f} ({time.time() - t0:.1f}s)")

    # ext_v2 — per-step picker, single result (not per-B)
    print(f"  ext_v2 ...", end=" ", flush=True)
    t0 = time.time()
    fn_v2 = make_ext_v2_step(target_forward, eagle3_draft, F, T, budgets)
    res_v2 = walk_method(sequences, fn_v2, vanilla_ms)
    out["extension_oracle_v2"] = res_v2
    print(f"spd={res_v2['speedup_real']:.3f} ({time.time() - t0:.1f}s)")

    # hybrid_v2 per (B, τ)
    for B in budgets:
        for tau in hybrid_thresholds:
            print(f"  hyb_v2 B={B} τ={tau} ...", end=" ", flush=True)
            t0 = time.time()
            fn = make_hybrid_v2_step(
                target_forward, eagle3_draft, suffix_speculate_ms,
                F, T, B, tau)
            res = walk_method(sequences, fn, vanilla_ms)
            out["hybrid_oracle_v2"][f"B{B}_t{tau}"] = res
            print(f"spd={res['speedup_real']:.3f} ({time.time() - t0:.1f}s)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent-results", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--responses", default=None)
    ap.add_argument("--latency-config", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--ft", default="1.0:0.1,4.0:0.0")
    ap.add_argument("--budgets", default="1,4,16,64,128")
    ap.add_argument("--hybrid-thresholds", default="1.0,5.0,10.0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.latency_config) as f:
        lc = json.load(f)
    target_forward, eagle3_draft, sfx_ms, vanilla_ms = build_latency(
        lc, args.steps, args.topk)

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

    ft_list = [tuple(float(x) for x in p.split(":"))
               for p in args.ft.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    thresholds = [float(t) for t in args.hybrid_thresholds.split(",")]

    out = {
        "metadata": {
            "agent_results": args.agent_results,
            "model": args.model,
            "reslice": f"s{args.steps}k{args.topk}",
            "vanilla_ms": vanilla_ms,
            "budgets": budgets,
            "hybrid_thresholds": thresholds,
        },
        "by_ft": {},
    }
    for (F, T) in ft_list:
        print(f"\n=== F={F}, T={T} ===", file=sys.stderr)
        res = run_one_ft(
            records, target_forward, eagle3_draft, sfx_ms, vanilla_ms,
            F, T, budgets, thresholds)
        out["by_ft"][f"F{F}_T{T}"] = res

    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nOutput: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
