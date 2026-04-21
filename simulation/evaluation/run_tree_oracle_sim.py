"""Tree-budget oracle simulation for heterogeneous speculative decoding.

Evaluates two oracle strategies on union tries built from multiple
proposers (EAGLE3, Suffix, Draft Model):

1. Choose-One Oracle: Pick the single best proposer's tree per step.
2. Expected-Utility Oracle: DP (tree knapsack) to find the optimal
   subtree under budget B from the union trie.

Input: union_trie_data.jsonl (from collect_union_trie + collect_target_probs)

Usage:
    python3 -m simulation.evaluation.run_tree_oracle_sim \
        --union-trie-data results/.../union_trie_data_with_pt.jsonl \
        --budgets 1,2,4,8,16,32,64 \
        --output results/.../tree_oracle_sim.json \
        --print-summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.evaluation.tree_knapsack import (
    greedy_tree_walk,
    tree_knapsack_dp,
    tree_knapsack_dp_all_budgets,
)


def evaluate_choose_one(
    records: List[dict],
) -> dict:
    """Evaluate Choose-One Oracle: pick the best single proposer per step.

    For each step, evaluates each proposer's tree independently against
    ground truth and picks the one with highest accepted tokens.
    """
    per_step = []
    proposer_wins: Dict[str, int] = {}
    proposer_total_acc: Dict[str, int] = {}
    proposer_total_size: Dict[str, int] = {}
    proposer_count: Dict[str, int] = {}
    total_best_acc = 0
    total_best_size = 0
    ties = 0

    for rec in records:
        gt = rec["ground_truth_future"]
        per_proposer = rec.get("per_proposer", {})

        best_acc = 0
        best_name = ""
        best_size = 0
        step_results = {}

        for name, tree_data in per_proposer.items():
            tids = tree_data["token_ids"]
            pids = tree_data["parents"]
            acc = greedy_tree_walk(tids, pids, gt)
            size = len(tids)
            u = acc / max(size, 1)

            step_results[name] = {"acc": acc, "size": size, "utility": u}
            proposer_total_acc[name] = proposer_total_acc.get(name, 0) + acc
            proposer_total_size[name] = proposer_total_size.get(name, 0) + size
            proposer_count[name] = proposer_count.get(name, 0) + 1

            if acc > best_acc or (acc == best_acc and size < best_size):
                best_acc = acc
                best_name = name
                best_size = size

        if best_name:
            proposer_wins[best_name] = proposer_wins.get(best_name, 0) + 1
        total_best_acc += best_acc
        total_best_size += best_size

        per_step.append({
            "request_id": rec["request_id"],
            "step_idx": rec["step_idx"],
            "best_proposer": best_name,
            "best_acc": best_acc,
            "best_size": best_size,
            "per_proposer": step_results,
        })

    # Aggregate
    n_steps = len(records)
    aggregate = {
        "n_steps": n_steps,
        "total_acc": total_best_acc,
        "total_size": total_best_size,
        "avg_acc": total_best_acc / max(n_steps, 1),
        "avg_utility": total_best_acc / max(total_best_size, 1),
        "proposer_wins": proposer_wins,
        "per_proposer_avg_acc": {
            name: proposer_total_acc[name] / max(proposer_count[name], 1)
            for name in proposer_total_acc
        },
        "per_proposer_avg_utility": {
            name: proposer_total_acc[name] / max(proposer_total_size[name], 1)
            for name in proposer_total_acc
        },
    }

    return {"aggregate": aggregate, "per_step": per_step}


def evaluate_expected_utility(
    records: List[dict],
    budgets: List[int],
    p_t_key: str = "p_t_oracle",
) -> dict:
    """Evaluate Expected-Utility Oracle via DP tree knapsack.

    For each step and each budget B, finds the optimal subtree of the
    union trie maximising expected accepted tokens.
    """
    budget_results: Dict[int, dict] = {}

    for B in budgets:
        total_eu = 0.0
        total_selected = 0
        total_actual_acc = 0
        per_step = []

        for rec in records:
            tids = rec["union_trie"]["token_ids"]
            pids = rec["union_trie"]["parents"]
            p_t = rec.get(p_t_key, [])

            if not tids or not p_t:
                per_step.append({
                    "request_id": rec["request_id"],
                    "step_idx": rec["step_idx"],
                    "budget": B,
                    "eu": 0.0,
                    "selected_nodes": 0,
                    "union_size": 0,
                })
                continue

            eu, selected = tree_knapsack_dp(tids, pids, p_t, budget=B)
            total_eu += eu
            total_selected += len(selected)

            # Measure actual acceptance of DP-selected subtree against ground truth
            gt = rec.get("ground_truth_future", [])
            if selected and gt:
                sel_set = set(selected)
                sel_tids = [tids[j] for j in range(len(tids)) if j in sel_set]
                sel_pids_raw = [pids[j] for j in range(len(tids)) if j in sel_set]
                # Remap parent indices to new compact indices
                old_to_new = {old: new for new, old in enumerate(
                    j for j in range(len(tids)) if j in sel_set)}
                sel_pids = [old_to_new[p] if p in old_to_new else -1
                            for p in sel_pids_raw]
                actual_acc = greedy_tree_walk(sel_tids, sel_pids, gt)
            else:
                actual_acc = 0
            total_actual_acc += actual_acc

            per_step.append({
                "request_id": rec["request_id"],
                "step_idx": rec["step_idx"],
                "budget": B,
                "eu": eu,
                "actual_acc": actual_acc,
                "selected_nodes": len(selected),
                "union_size": len(tids),
            })

        n_steps = len(records)
        budget_results[B] = {
            "budget": B,
            "total_eu": total_eu,
            "avg_eu": total_eu / max(n_steps, 1),
            "total_actual_acc": total_actual_acc,
            "avg_actual_acc": total_actual_acc / max(n_steps, 1),
            "avg_selected_nodes": total_selected / max(n_steps, 1),
            "per_step": per_step,
        }

    return budget_results


def evaluate_choose_one_at_budget(
    records: List[dict],
    budgets: List[int],
) -> Dict[int, dict]:
    """Choose-One Oracle: pick the best proposer's full tree per step.

    No budget truncation — each proposer's entire tree is evaluated.
    Result is the same for all budgets (budget-independent).
    """
    # Compute once (budget doesn't matter)
    n_steps = len(records)
    total_acc = 0

    for rec in records:
        gt = rec["ground_truth_future"]
        per_proposer = rec.get("per_proposer", {})
        best_acc = 0

        for name, tree_data in per_proposer.items():
            tids = tree_data["token_ids"]
            pids = tree_data["parents"]
            acc = greedy_tree_walk(tids, pids, gt)
            best_acc = max(best_acc, acc)

        total_acc += best_acc

    avg = total_acc / max(n_steps, 1)
    return {B: {"budget": B, "total_acc": total_acc, "avg_acc": avg}
            for B in budgets}


def print_summary(
    choose_one: dict,
    eu_results: Dict[int, dict],
    choose_one_budget: Dict[int, dict],
    budgets: List[int],
    p_t_key: str,
):
    """Print a formatted summary to stderr."""
    print("\n" + "=" * 70, file=sys.stderr)
    print("TREE-BUDGET ORACLE SIMULATION RESULTS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Choose-One summary
    agg = choose_one["aggregate"]
    print(f"\n--- Choose-One Oracle (unconstrained) ---", file=sys.stderr)
    print(f"Steps: {agg['n_steps']}", file=sys.stderr)
    print(f"Avg accepted: {agg['avg_acc']:.3f}", file=sys.stderr)
    print(f"Avg utility (acc/size): {agg['avg_utility']:.4f}", file=sys.stderr)
    print(f"Proposer wins: {agg['proposer_wins']}", file=sys.stderr)
    print(f"Per-proposer avg acc: {agg['per_proposer_avg_acc']}", file=sys.stderr)

    # Budget sweep comparison
    print(f"\n--- Budget Sweep (p_t: {p_t_key}) ---", file=sys.stderr)
    print(f"{'Budget':>8} | {'Choose-1':>9}", file=sys.stderr)
    print("-" * 22, file=sys.stderr)
    for B in budgets:
        c1 = choose_one_budget[B]["avg_acc"]
        print(f"{B:>8} | {c1:>9.4f}", file=sys.stderr)

    print("=" * 22, file=sys.stderr)


# ---------------------------------------------------------------------------
# Step-by-step simulation (correct skip-ahead behavior)
# ---------------------------------------------------------------------------

def simulate_decoding(
    records: List[dict],
    budget: int,
    method: str,
    p_t_key: str = "p_t_oracle",
    *,
    vanilla_latency_ms: float,
    verify_latency_ms: float = 0.0,
    suffix_cache=None,
    draft_ratios: Optional[List[float]] = None,
    real_step_cost_ms: Optional[float] = None,
    real_step_cost_suffix_ms: Optional[float] = None,
) -> dict:
    """Simulate speculative decoding with skip-ahead.

    Computes MAT + speedup for multiple draft cost ratios in a single pass.

    draft_ratios: list of ratios (e.g. [0.05, 0.1, 0.2, 0.3, 0.5]).
        step_cost = vanilla_ms * (1 + ratio) for methods with draft cost.
        step_cost = vanilla_ms for suffix-only (no draft cost).

    real_step_cost_ms: measured step cost in ms (when draft is active). Used to
        compute a second speedup based on actual measured latencies.
    real_step_cost_suffix_ms: for hybrid only, cost when suffix branch selected
        (no draft cost). If None, defaults to vanilla_latency_ms.
    """
    record_index: Dict[Tuple, dict] = {}
    sequences: Dict[Tuple, List[int]] = {}

    for rec in records:
        key = (rec["request_id"], rec.get("call_idx", 0), rec.get("step_idx", 0))
        record_index[key] = rec
        seq_key = (rec["request_id"], rec.get("call_idx", 0))
        sequences.setdefault(seq_key, []).append(rec.get("step_idx", 0))

    for sk in sequences:
        sequences[sk].sort()

    # Determine if this method has draft cost
    is_hybrid = method.startswith("hybrid_e3:") or method.startswith("hybrid_dm:")
    no_draft = method in ("single:suffix", "extension", "extension_dmsfx")  # no draft cost

    ratios = draft_ratios or []
    # Ratio-based time accumulation
    time_per_ratio = {r: 0.0 for r in ratios}
    # For hybrid: conditional (draft only on fallback) + always (draft every step)
    time_per_ratio_always = {r: 0.0 for r in ratios} if is_hybrid else None

    # Real-cost accumulators (use measured latencies)
    sfx_cost_ms = real_step_cost_suffix_ms if real_step_cost_suffix_ms is not None else vanilla_latency_ms
    total_time_real_ms = 0.0 if real_step_cost_ms is not None else None
    total_time_real_always_ms = 0.0 if (real_step_cost_ms is not None and is_hybrid) else None

    total_generated = 0
    total_accepted = 0
    total_steps = 0
    total_time_ms = 0.0
    v_ms = vanilla_latency_ms

    for seq_key, step_indices in sorted(sequences.items()):
        req_id, call_idx = seq_key
        if not step_indices:
            continue

        max_pos = max(step_indices)
        first_rec = record_index.get((req_id, call_idx, step_indices[0]))
        if not first_rec:
            continue
        last_rec = record_index.get((req_id, call_idx, step_indices[-1]))
        last_gt_len = len(last_rec.get("ground_truth_future", [])) if last_rec else 1
        first_gt_len = len(first_rec.get("ground_truth_future", []))
        if last_gt_len <= 1:
            seq_len = step_indices[0] + 1 + first_gt_len
        else:
            seq_len = step_indices[0] + first_gt_len

        cache_req_id = f"{req_id}_{call_idx}"
        if suffix_cache is not None:
            prompt = first_rec.get("context_token_ids", [])
            suffix_cache.start_request(cache_req_id,
                                       np.array(prompt, dtype=np.int32))

        pos = step_indices[0]
        step_set = set(step_indices)

        while pos <= max_pos and pos in step_set:
            rec = record_index.get((req_id, call_idx, pos))
            if rec is None:
                total_generated += 1
                total_steps += 1
                total_time_ms += v_ms
                for r in ratios:
                    time_per_ratio[r] += v_ms
                    if time_per_ratio_always is not None:
                        time_per_ratio_always[r] += v_ms
                if total_time_real_ms is not None:
                    total_time_real_ms += v_ms
                    if total_time_real_always_ms is not None:
                        total_time_real_always_ms += v_ms
                pos += 1
                continue

            # Dispatch method
            used_suffix = False
            if method == "union_trie":
                accepted = _union_trie_step(rec, budget)
            elif method.startswith("union_trie:"):
                names = frozenset(method.split(":", 1)[1].split(","))
                accepted = _union_trie_step(rec, budget, proposer_set=names)
            elif method == "extension":
                accepted = _extension_step(rec, budget, suffix_cache,
                                           cache_req_id, base_proposer="eagle3")
            elif method == "extension_dmsfx":
                accepted = _extension_step(rec, budget, suffix_cache,
                                           cache_req_id, base_proposer="draft_model")
            elif method == "eu":
                accepted = _eu_step(rec, budget, p_t_key)
            elif method.startswith("eu_pair:"):
                names = method.split(":", 1)[1].split(",")
                accepted = _eu_step(rec, budget, p_t_key,
                                    allowed_proposers=set(names))
            elif is_hybrid:
                if method.startswith("hybrid_e3:"):
                    threshold = float(method.split(":", 1)[1])
                    accepted, used_suffix = _hybrid_step(
                        rec, budget, threshold, fallback="eagle3")
                else:
                    threshold = float(method.split(":", 1)[1])
                    accepted, used_suffix = _hybrid_step(
                        rec, budget, threshold, fallback="draft_model")
            elif method.startswith("single:"):
                proposer_name = method.split(":", 1)[1]
                accepted = _single_proposer_step(rec, budget, proposer_name,
                                                 p_t_key)
            elif method.startswith("subset:"):
                names = method.split(":", 1)[1].split(",")
                accepted = _subset_step(rec, budget, names, p_t_key)
            else:  # choose_one
                accepted = _choose_one_step(rec, budget, p_t_key)

            advance = accepted + 1
            total_generated += advance
            total_accepted += accepted
            total_steps += 1
            total_time_ms += verify_latency_ms if verify_latency_ms > 0 else v_ms

            # Accumulate ratio-based time
            for r in ratios:
                if no_draft:
                    time_per_ratio[r] += v_ms  # no draft cost
                elif is_hybrid:
                    # conditional: draft only when fallback used
                    if used_suffix:
                        time_per_ratio[r] += v_ms
                    else:
                        time_per_ratio[r] += v_ms * (1 + r)
                    # always: draft cost every step
                    time_per_ratio_always[r] += v_ms * (1 + r)
                else:
                    time_per_ratio[r] += v_ms * (1 + r)

            # Accumulate real-cost time (measured latencies)
            if total_time_real_ms is not None:
                if no_draft:
                    total_time_real_ms += sfx_cost_ms
                elif is_hybrid:
                    if used_suffix:
                        total_time_real_ms += sfx_cost_ms
                    else:
                        total_time_real_ms += real_step_cost_ms
                    total_time_real_always_ms += real_step_cost_ms
                else:
                    total_time_real_ms += real_step_cost_ms

            # Feed accepted tokens to suffix cache
            if suffix_cache is not None:
                gt = rec.get("ground_truth_future", [])
                if gt and advance <= len(gt):
                    suffix_cache.add_active_response(
                        cache_req_id, gt[:advance])

            pos += advance

        remaining = seq_len - pos
        if remaining > 0:
            total_generated += remaining
            total_steps += remaining
            total_time_ms += remaining * v_ms
            for r in ratios:
                time_per_ratio[r] += remaining * v_ms
                if time_per_ratio_always is not None:
                    time_per_ratio_always[r] += remaining * v_ms
            if total_time_real_ms is not None:
                total_time_real_ms += remaining * v_ms
                if total_time_real_always_ms is not None:
                    total_time_real_always_ms += remaining * v_ms

        if suffix_cache is not None:
            suffix_cache.stop_request(cache_req_id)

    vanilla_time_ms = total_generated * v_ms
    speedup = vanilla_time_ms / total_time_ms if total_time_ms > 0 else 1.0
    mat = total_accepted / total_steps if total_steps > 0 else 0.0

    # Compute speedup per ratio
    speedup_per_ratio = {}
    for r in ratios:
        t = time_per_ratio[r]
        speedup_per_ratio[r] = vanilla_time_ms / t if t > 0 else 1.0
    speedup_per_ratio_always = {}
    if time_per_ratio_always is not None:
        for r in ratios:
            t = time_per_ratio_always[r]
            speedup_per_ratio_always[r] = vanilla_time_ms / t if t > 0 else 1.0

    result = {
        "total_generated": total_generated,
        "total_accepted": total_accepted,
        "total_steps": total_steps,
        "total_time_ms": total_time_ms,
        "vanilla_time_ms": vanilla_time_ms,
        "speedup": speedup,
        "mat": mat,
    }
    if speedup_per_ratio:
        result["speedup_per_ratio"] = speedup_per_ratio
    if speedup_per_ratio_always:
        result["speedup_per_ratio_always"] = speedup_per_ratio_always
    # Real-cost speedups (measured latencies)
    if total_time_real_ms is not None:
        result["speedup_real"] = (vanilla_time_ms / total_time_real_ms
                                  if total_time_real_ms > 0 else 1.0)
    if total_time_real_always_ms is not None:
        result["speedup_real_always"] = (vanilla_time_ms / total_time_real_always_ms
                                         if total_time_real_always_ms > 0 else 1.0)
    return result


def _filter_union_trie(rec: dict, allowed: set, p_t_key: str = "p_t"):
    """Filter union trie to only include nodes from allowed proposers.

    Returns (tids, pids, p_t) for the filtered subtree, or None if empty.
    Preserves tree structure: if a node is kept, all ancestors up to root
    are also kept (even if they belong to other proposers).
    """
    tids = rec["union_trie"]["token_ids"]
    pids = rec["union_trie"]["parents"]
    source_map = rec.get("source_map", [])
    p_t_key_vals = rec.get(p_t_key, [])

    if not tids or not source_map:
        return None

    n = len(tids)
    # Mark nodes whose sources overlap with allowed set
    keep = set()
    for i in range(n):
        sources = source_map[i] if i < len(source_map) else []
        if set(sources) & allowed:
            # Keep this node and all ancestors
            j = i
            while j >= 0 and j not in keep:
                keep.add(j)
                j = pids[j] if j < len(pids) else -1

    if not keep:
        return None

    # Build filtered arrays preserving order
    kept_sorted = sorted(keep)
    old_to_new = {old: new for new, old in enumerate(kept_sorted)}
    f_tids = [tids[j] for j in kept_sorted]
    f_pids = [old_to_new.get(pids[j], -1) if pids[j] >= 0 else -1
              for j in kept_sorted]
    f_pt = [p_t_key_vals[j] if j < len(p_t_key_vals) else 0.0
            for j in kept_sorted]

    return f_tids, f_pids, f_pt


def precompute_eu_results(
    records: List[dict],
    budgets: List[int],
    p_t_key: str,
) -> None:
    """Precompute EU Oracle (all proposers) for all budgets in one DP pass.

    Stores results in rec["_eu_cache"]["all"][budget] = accepted_tokens.
    """
    t0 = time.time()
    n = len(records)
    for i, rec in enumerate(records):
        gt = rec.get("ground_truth_future", [])
        if not gt:
            rec["_eu_cache"] = {}
            continue

        cache = {}

        # Full union trie EU
        tids = rec["union_trie"]["token_ids"]
        pids = rec["union_trie"]["parents"]
        p_t = rec.get(p_t_key, [])
        if tids and p_t:
            all_results = tree_knapsack_dp_all_budgets(tids, pids, p_t, budgets)
            budget_accepted = {}
            for b, (eu, selected) in all_results.items():
                if not selected:
                    budget_accepted[b] = 0
                    continue
                sel_set = set(selected)
                sel_tids = [tids[j] for j in range(len(tids)) if j in sel_set]
                sel_pids_raw = [pids[j] for j in range(len(tids)) if j in sel_set]
                old_to_new = {old: new for new, old in enumerate(
                    j for j in range(len(tids)) if j in sel_set)}
                sel_pids = [old_to_new.get(p, -1) if p >= 0 else -1
                            for p in sel_pids_raw]
                budget_accepted[b] = greedy_tree_walk(sel_tids, sel_pids, gt)
            cache["all"] = budget_accepted

        # EU for eagle3+suffix pair
        source_map = rec.get("source_map", [])
        proposer_set = set()
        for sm in source_map:
            proposer_set.update(sm)
        if "eagle3" in proposer_set and "suffix" in proposer_set:
            filtered = _filter_union_trie(rec, {"eagle3", "suffix"}, p_t_key)
            if filtered is not None:
                f_tids, f_pids, f_pt = filtered
                all_results = tree_knapsack_dp_all_budgets(
                    f_tids, f_pids, f_pt, budgets)
                budget_accepted = {}
                for b, (eu, selected) in all_results.items():
                    if not selected:
                        budget_accepted[b] = 0
                        continue
                    sel_set = set(selected)
                    s_tids = [f_tids[j] for j in range(len(f_tids)) if j in sel_set]
                    s_pids_raw = [f_pids[j] for j in range(len(f_tids))
                                  if j in sel_set]
                    old_to_new = {old: new for new, old in enumerate(
                        j for j in range(len(f_tids)) if j in sel_set)}
                    s_pids = [old_to_new.get(p, -1) if p >= 0 else -1
                              for p in s_pids_raw]
                    budget_accepted[b] = greedy_tree_walk(s_tids, s_pids, gt)
                cache["eagle3,suffix"] = budget_accepted

        rec["_eu_cache"] = cache

        if (i + 1) % 500 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  EU precompute: [{i+1}/{n}] {rate:.1f} rec/s",
                  file=sys.stderr)


def _eu_step(rec: dict, budget: int, p_t_key: str,
             allowed_proposers: Optional[set] = None) -> int:
    """EU Oracle: select optimal subtree via DP, return accepted tokens.

    Uses precomputed cache if available (from precompute_eu_results).
    """
    # Check cache first
    cache = rec.get("_eu_cache")
    if cache is not None:
        if allowed_proposers is not None:
            cache_key = ",".join(sorted(allowed_proposers))
        else:
            cache_key = "all"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached.get(budget, 0)

    # Fallback: compute on the fly
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0

    if allowed_proposers is not None:
        filtered = _filter_union_trie(rec, allowed_proposers, p_t_key)
        if filtered is None:
            return 0
        tids, pids, p_t = filtered
    else:
        tids = rec["union_trie"]["token_ids"]
        pids = rec["union_trie"]["parents"]
        p_t = rec.get(p_t_key, [])

    if not tids or not p_t:
        return 0

    _, selected = tree_knapsack_dp(tids, pids, p_t, budget=budget)
    if not selected:
        return 0

    sel_set = set(selected)
    sel_tids = [tids[j] for j in range(len(tids)) if j in sel_set]
    sel_pids_raw = [pids[j] for j in range(len(tids)) if j in sel_set]
    old_to_new = {old: new for new, old in enumerate(
        j for j in range(len(tids)) if j in sel_set)}
    sel_pids = [old_to_new[p] if p in old_to_new else -1 for p in sel_pids_raw]
    return greedy_tree_walk(sel_tids, sel_pids, gt)


def _hybrid_step(rec: dict, budget: int, threshold: float,
                 fallback: str = "eagle3") -> tuple:
    """Hybrid: use suffix if score >= threshold, else fallback proposer.

    Returns (accepted_tokens, used_suffix: bool).
    used_suffix=True → suffix was chosen (no draft cost).
    used_suffix=False → fallback was chosen (draft cost applies).
    """
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0, False

    per_proposer = rec.get("per_proposer", {})
    suffix_data = per_proposer.get("suffix")
    fallback_data = per_proposer.get(fallback)

    use_suffix = (suffix_data is not None
                  and suffix_data.get("score", 0.0) >= threshold
                  and suffix_data.get("token_ids"))

    if use_suffix:
        return _proposer_tree_walk(per_proposer, "suffix", gt, budget), True
    elif fallback_data and fallback_data.get("token_ids"):
        return _proposer_tree_walk(per_proposer, fallback, gt, budget), False
    else:
        return 0, False


def _union_trie_step(rec: dict, budget: int,
                     proposer_set: Optional[frozenset] = None) -> int:
    """Union trie: truncate to budget by BFS order, then greedy walk.

    proposer_set: if None, uses precomputed rec["union_trie"] (eagle3+suffix
    by default). Otherwise rebuilds union from the specified proposers in
    rec["per_proposer"].
    """
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0

    if proposer_set is None:
        trie = rec.get("union_trie")
        if not trie or not trie.get("token_ids"):
            return 0
        tids = trie["token_ids"]
        pids = trie["parents"]
    else:
        # Rebuild union trie on the fly from per_proposer subset
        from simulation.pipeline.collect_union_trie import (
            build_union_trie,
        )
        per_prop = rec.get("per_proposer", {})
        trees = {}
        for name in proposer_set:
            pd = per_prop.get(name)
            if pd and pd.get("token_ids"):
                trees[name] = (pd["token_ids"], pd["parents"])
        if not trees:
            return 0
        tids, pids, _ = build_union_trie(trees)
        if not tids:
            return 0

    n = min(budget, len(tids))
    if n < len(tids):
        tids = tids[:n]
        pids = pids[:n]
        pids = [p if p < n else -1 for p in pids]

    return greedy_tree_walk(tids, pids, gt)


def _extension_step(rec: dict, budget: int, suffix_cache, cache_req_id: str,
                    base_proposer: str = "eagle3") -> int:
    """Extension: base proposer's tree (truncated to budget) + suffix extension
    at every node.

    For EVERY node in the base tree, trace root→node path, build extended context,
    and call suffix_cache.speculate() to extend. Then greedy walk on the combined
    (base + suffix extensions) tree.

    base_proposer: "eagle3" (default) or "draft_model".
    """
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0

    base = rec.get("per_proposer", {}).get(base_proposer)
    if not base or not base.get("token_ids"):
        return 0

    tids = base["token_ids"]
    pids = base["parents"]

    # Truncate base tree to budget
    n = min(budget, len(tids))
    tids = tids[:n]
    pids = pids[:n]
    pids = [p if p < n else -1 for p in pids]

    # Build extended tree
    ext_tids = list(tids)
    ext_pids = list(pids)

    base_context = rec.get("context_token_ids")
    if base_context is None or suffix_cache is None:
        return greedy_tree_walk(ext_tids, ext_pids, gt)

    # Precompute root→node paths for all nodes
    paths = [None] * n
    for i in range(n):
        path = []
        node = i
        while node >= 0:
            path.append(tids[node])
            node = pids[node]
        path.reverse()
        paths[i] = path

    # For every node, extend with suffix
    for node_idx in range(n):
        ext_context = np.array(base_context + paths[node_idx], dtype=np.int32)

        try:
            draft = suffix_cache.speculate(cache_req_id, ext_context)
        except Exception:
            continue

        if not draft.token_ids:
            continue

        # Attach suffix chain to this node
        offset = len(ext_tids)
        for j, (tid, pid) in enumerate(zip(draft.token_ids, draft.parents)):
            ext_tids.append(tid)
            if pid == -1:
                ext_pids.append(node_idx)  # suffix root children → attach to node
            else:
                ext_pids.append(offset + pid)

    return greedy_tree_walk(ext_tids, ext_pids, gt)


def _truncate_and_walk(tids, pids, p_t, gt, budget):
    """Truncate tree to budget nodes via knapsack DP, then greedy walk."""
    if not tids or not gt:
        return 0

    n = len(tids)
    if budget >= n:
        # Budget covers full tree, no truncation needed
        return greedy_tree_walk(tids, pids, gt)

    if not p_t or len(p_t) < n:
        # No p_t available, fall back to full tree
        return greedy_tree_walk(tids, pids, gt)

    _, selected = tree_knapsack_dp(tids, pids, p_t, budget=budget)
    if not selected:
        return 0

    sel_set = set(selected)
    sel_tids = [tids[j] for j in range(n) if j in sel_set]
    sel_pids_raw = [pids[j] for j in range(n) if j in sel_set]
    old_to_new = {old: new for new, old in enumerate(
        j for j in range(n) if j in sel_set)}
    sel_pids = [old_to_new.get(p, -1) if p >= 0 else -1 for p in sel_pids_raw]
    return greedy_tree_walk(sel_tids, sel_pids, gt)


def _proposer_tree_walk(per_proposer: dict, name: str, gt: list, budget: int) -> int:
    """Walk a single proposer's per_proposer tree.

    Suffix has no draft cost (CPU-free), so its tree is never budget-limited.
    EAGLE3/draft_model trees are truncated to budget by BFS order.
    """
    tree_data = per_proposer.get(name)
    if not tree_data or not tree_data.get("token_ids"):
        return 0

    tids = tree_data["token_ids"]
    pids = tree_data["parents"]

    # Suffix is free — always use full tree
    if name != "suffix" and budget < len(tids):
        # Truncate: keep first B nodes (BFS/tree order from proposer)
        tids = tids[:budget]
        pids = pids[:budget]
        # Fix parent references that point beyond truncated range
        pids = [p if p < budget else -1 for p in pids]

    return greedy_tree_walk(tids, pids, gt)


def _single_proposer_step(rec: dict, budget: int, proposer_name: str,
                          p_t_key: str = "p_t") -> int:
    """Single proposer: use per_proposer tree directly, truncate to budget."""
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0
    return _proposer_tree_walk(rec.get("per_proposer", {}), proposer_name, gt, budget)


def _choose_one_step(rec: dict, budget: int, p_t_key: str = "p_t") -> int:
    """Choose-One Oracle: try each proposer's full tree, pick best.

    No budget truncation — each proposer uses its full tree.
    This is an oracle that picks the best proposer per step.
    """
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0

    best_acc = 0
    for name, tree_data in rec.get("per_proposer", {}).items():
        if not tree_data or not tree_data.get("token_ids"):
            continue
        acc = greedy_tree_walk(tree_data["token_ids"], tree_data["parents"], gt)
        best_acc = max(best_acc, acc)
    return best_acc


def _subset_step(rec: dict, budget: int, proposer_names: List[str],
                 p_t_key: str = "p_t") -> int:
    """Choose-One over a subset of proposers. Full trees, no truncation."""
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0

    best_acc = 0
    for name in proposer_names:
        tree_data = rec.get("per_proposer", {}).get(name)
        if not tree_data or not tree_data.get("token_ids"):
            continue
        acc = greedy_tree_walk(tree_data["token_ids"], tree_data["parents"], gt)
        best_acc = max(best_acc, acc)
    return best_acc


def _discover_proposers(records: List[dict]) -> List[str]:
    """Find all proposer names present in the records."""
    names: set = set()
    for rec in records:
        names.update(rec.get("per_proposer", {}).keys())
    return sorted(names)


def compute_latency_speedup(
    records: List[dict],
    budgets: List[int],
    latency_config: dict,
    p_t_key: str = "p_t",
    enable_eu: bool = False,
) -> dict:
    """Run step-by-step simulation for each budget with measured latencies.

    Returns per-budget simulation results including speedup.

    Parameters
    ----------
    enable_eu : bool
        Enable EU Oracle (tree knapsack DP). Slow for large trees. Default off.

    Latency config should contain decomposed costs:
        vanilla_step_ms: target TPOT with no speculation
        target_forward_ms: {B: ms} — pure target verify cost for B tokens
        eagle3_draft_ms: {B: ms} — EAGLE3 draft generation cost
        draft_lm_tpot_ms: draft model per-token cost

    Backward compatible: derives from legacy verify_latencies_ms if needed.
    """
    vanilla_ms = latency_config["vanilla_step_ms"]
    proposers = _discover_proposers(records)

    # --- Decomposed latencies ---
    # target_forward_ms[B]: pure target model verify cost for B tokens
    # eagle3_draft_ms[B]: EAGLE3 draft generation cost for B tokens
    # Backward compat: derive from verify_latencies_ms if new keys absent
    target_fwd = latency_config.get("target_forward_ms", {})
    eagle3_draft = latency_config.get("eagle3_draft_ms", {})
    legacy_verify = latency_config.get("verify_latencies_ms",
                                       latency_config.get("eagle3_step_ms", {}))

    if not target_fwd and legacy_verify:
        # Derive from legacy: target_forward ≈ vanilla, eagle3_draft = remainder
        for b_str, step in legacy_verify.items():
            target_fwd[b_str] = vanilla_ms
            eagle3_draft[b_str] = max(float(step) - vanilla_ms, 0.0)

    # Per-proposer draft costs (non-EAGLE3)
    draft_lm_tpot = latency_config.get("draft_lm_tpot_ms", 0.0)

    def _target_forward(B: int) -> float:
        """Pure target model forward cost for verifying B tokens."""
        return float(target_fwd.get(str(B), vanilla_ms))

    def _eagle3_draft(B: int) -> float:
        """EAGLE3 draft generation cost for budget B."""
        return float(eagle3_draft.get(str(B), 0.0))

    def _proposer_draft_cost(name: str, B: int) -> float:
        """Draft cost for a single proposer at budget B."""
        if name == "eagle3":
            return _eagle3_draft(B)
        elif name == "draft_model":
            return B * draft_lm_tpot  # generates B tokens autoregressively
        elif name == "suffix":
            return 0.0  # CPU lookup, free
        elif name == "mtp":
            return 0.0  # uses target model MTP heads, cost in target_forward
        return 0.0

    def _step_cost(active_proposers: List[str], B: int) -> float:
        """Step cost = target_forward(B) + max(draft costs of GPU proposers).

        Proposers draft in parallel → cost = max, not sum.
        """
        t_fwd = _target_forward(B)
        draft_costs = [_proposer_draft_cost(p, B) for p in active_proposers]
        max_draft = max(draft_costs) if draft_costs else 0.0
        return t_fwd + max_draft

    DRAFT_RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50]

    def _real_cost(active_proposers, B):
        """Step cost in ms using measured latencies.
        target_forward + max(draft costs of non-suffix active proposers)."""
        tf = _target_forward(B)
        drafts = [_proposer_draft_cost(p, B) for p in active_proposers
                  if p != "suffix"]
        return tf + (max(drafts) if drafts else 0.0)

    def _store_sim(entry, prefix, sim):
        """Store MAT + ratio-based + real-cost speedups from a simulation result."""
        entry[f"{prefix}_mat"] = sim["mat"]
        entry[f"{prefix}_steps"] = sim.get("total_steps", 0)
        spr = sim.get("speedup_per_ratio", {})
        for r, spd in spr.items():
            entry[f"{prefix}_speedup_r{r}"] = spd
        spr_always = sim.get("speedup_per_ratio_always", {})
        for r, spd in spr_always.items():
            entry[f"{prefix}_always_speedup_r{r}"] = spd
        if "speedup_real" in sim:
            entry[f"{prefix}_speedup_real"] = sim["speedup_real"]
        if "speedup_real_always" in sim:
            entry[f"{prefix}_always_speedup_real"] = sim["speedup_real_always"]

    if enable_eu:
        print(f"Precomputing EU Oracle DP for {len(records)} records...",
              file=sys.stderr)
        precompute_eu_results(records, budgets, p_t_key)
        print("EU precompute done.", file=sys.stderr)

    def _run(method_key, sim_fn_kwargs, prefix):
        """Run one simulation and log timing. sim_fn_kwargs is kwargs dict for simulate_decoding."""
        t0 = time.time()
        sim = simulate_decoding(**sim_fn_kwargs)
        dt = time.time() - t0
        _store_sim(entry, prefix, sim)
        print(f"    {method_key}: {dt:5.1f}s  mat={sim['mat']:.2f} spd={sim['speedup']:.2f}x",
              file=sys.stderr)
        sys.stderr.flush()

    results = {}
    total_budgets = len(budgets)
    for b_idx, B in enumerate(budgets):
        entry = {
            "budget": B,
            "target_forward_ms": _target_forward(B),
            "eagle3_draft_ms": _eagle3_draft(B),
            "draft_lm_tpot_ms": draft_lm_tpot,
        }
        b_t0 = time.time()
        print(f"\n[{b_idx+1}/{total_budgets}] Budget={B} ---", file=sys.stderr)
        sys.stderr.flush()

        if enable_eu:
            _run("eu", dict(records=records, budget=B, method="eu",
                            p_t_key=p_t_key, vanilla_latency_ms=vanilla_ms,
                            draft_ratios=DRAFT_RATIOS), "eu")

            if "suffix" in proposers and "eagle3" in proposers:
                _run("eu_e3sfx", dict(records=records, budget=B,
                                      method="eu_pair:eagle3,suffix",
                                      vanilla_latency_ms=vanilla_ms,
                                      p_t_key=p_t_key,
                                      draft_ratios=DRAFT_RATIOS), "eu_e3sfx")

        common = dict(records=records, budget=B,
                      vanilla_latency_ms=vanilla_ms,
                      draft_ratios=DRAFT_RATIOS)

        # Union trie
        _run("union_trie_e3sfx", {**common,
             "method": "union_trie:eagle3,suffix",
             "real_step_cost_ms": _real_cost(["eagle3", "suffix"], B)},
             "union_trie_e3sfx")

        if "draft_model" in proposers:
            _run("union_trie_all", {**common,
                 "method": "union_trie:eagle3,suffix,draft_model",
                 "real_step_cost_ms": _real_cost(["eagle3", "suffix", "draft_model"], B)},
                 "union_trie_all")

        # Choose-One Oracle (all proposers, full tree)
        _run("c1", {**common, "method": "choose_one",
             "real_step_cost_ms": _real_cost(list(proposers), B)}, "c1")

        # Single-proposer baselines
        for pname in proposers:
            _run(f"single:{pname}",
                 {**common, "method": f"single:{pname}",
                  "real_step_cost_ms": _real_cost([pname], B)},
                 pname)

        # Hybrid (suffix score threshold): suffix if score >= t, else fallback
        thresholds = [1.0, 3.0, 5.0]

        if "suffix" in proposers and "eagle3" in proposers:
            e3_cost = _real_cost(["eagle3"], B)
            for t in thresholds:
                _run(f"hybrid_e3:{t}",
                     {**common, "method": f"hybrid_e3:{t}",
                      "real_step_cost_ms": e3_cost,
                      "real_step_cost_suffix_ms": _target_forward(B)},
                     f"hybrid_e3_t{t:.1f}")

        if "suffix" in proposers and "draft_model" in proposers:
            dm_cost = _real_cost(["draft_model"], B)
            for t in thresholds:
                _run(f"hybrid_dm:{t}",
                     {**common, "method": f"hybrid_dm:{t}",
                      "real_step_cost_ms": dm_cost,
                      "real_step_cost_suffix_ms": _target_forward(B)},
                     f"hybrid_dm_t{t:.1f}")

        # C1 for suffix+eagle3 pair
        if "suffix" in proposers and "eagle3" in proposers:
            _run("c1_e3sfx",
                 {**common, "method": "subset:eagle3,suffix",
                  "p_t_key": p_t_key,
                  "real_step_cost_ms": _real_cost(["eagle3", "suffix"], B)},
                 "c1_e3sfx")

        # Extension: base tree + suffix extension at every node
        from hybrid_spec_decoding.suffix_decoding.suffix_tree import (
            SuffixDecodingCache,
        )
        if "suffix" in proposers and "eagle3" in proposers:
            if not hasattr(compute_latency_speedup, '_ext_cache'):
                compute_latency_speedup._ext_cache = SuffixDecodingCache(
                    max_tree_depth=64, max_cached_requests=100000)
            _run("extension",
                 {**common, "method": "extension",
                  "suffix_cache": compute_latency_speedup._ext_cache,
                  "real_step_cost_ms": _real_cost(["eagle3"], B)},
                 "extension")

        # Extension with draft_model base + suffix extensions
        if "suffix" in proposers and "draft_model" in proposers:
            if not hasattr(compute_latency_speedup, '_ext_cache_dm'):
                compute_latency_speedup._ext_cache_dm = SuffixDecodingCache(
                    max_tree_depth=64, max_cached_requests=100000)
            _run("extension_dmsfx",
                 {**common, "method": "extension_dmsfx",
                  "suffix_cache": compute_latency_speedup._ext_cache_dm,
                  "real_step_cost_ms": _real_cost(["draft_model"], B)},
                 "extension_dmsfx")

        results[B] = entry
        b_dt = time.time() - b_t0
        print(f"[{b_idx+1}/{total_budgets}] Budget={B} done in {b_dt:.1f}s",
              file=sys.stderr)
        sys.stderr.flush()

    return results


def print_latency_summary(
    latency_results: dict,
    budgets: List[int],
    vanilla_ms: float,
):
    """Print latency-aware speedup summary.

    Groups methods by prefix and prints MAT + best-available speedup per
    budget. Speedup preference: real (measured costs) > lowest ratio.
    Silently skips sections with no data so it stays useful even when only
    a subset of methods was evaluated.
    """
    first = latency_results[budgets[0]]

    # Collect all prefixes that have a MAT column. A prefix is a method tag
    # like "c1", "eu", "single:eagle3", "extension", "union_trie_e3sfx".
    prefixes = sorted({k[:-4] for k in first if k.endswith("_mat")})
    if not prefixes:
        return

    def _best_speedup(r: dict, prefix: str) -> tuple:
        """Return (speedup, source_label) for a method at a budget."""
        if f"{prefix}_speedup_real" in r:
            return r[f"{prefix}_speedup_real"], "real"
        ratio_keys = sorted(
            [k for k in r if k.startswith(f"{prefix}_speedup_r")],
            key=lambda k: float(k.split("_r")[-1]))
        if ratio_keys:
            k = ratio_keys[0]
            return r[k], k.split("_speedup_")[1]
        return 0.0, ""

    t_fwd = first.get("target_forward_ms", vanilla_ms)
    e3_draft = first.get("eagle3_draft_ms", 0.0)
    dm_tpot = first.get("draft_lm_tpot_ms", 0.0)

    print("\n" + "=" * 90, file=sys.stderr)
    print("LATENCY-AWARE SPEEDUP SUMMARY", file=sys.stderr)
    print("=" * 90, file=sys.stderr)
    print(f"Vanilla TPOT: {vanilla_ms:.2f} ms/tok  |  "
          f"Target fwd (B={budgets[0]}): {t_fwd:.2f} ms  |  "
          f"EAGLE3 draft: {e3_draft:.2f} ms  |  "
          f"Draft LM TPOT: {dm_tpot:.2f} ms", file=sys.stderr)
    print("Step cost = target_forward(B) + max(draft costs); suffix = 0 (CPU)",
          file=sys.stderr)

    # One row per (budget, method), columns: budget | mat | speedup(source)
    label_w = max(len(p) for p in prefixes)
    hdr = (f"{'Budget':>6} | {'Method':<{label_w}} | "
           f"{'MAT':>6} | {'Speedup':>8} | Source")
    print("\n" + hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)

    best: Dict[str, tuple] = {p: (0, 0.0, "") for p in prefixes}

    for B in budgets:
        r = latency_results[B]
        for p in prefixes:
            mat = r.get(f"{p}_mat", 0.0)
            spd, src = _best_speedup(r, p)
            print(f"{B:>6} | {p:<{label_w}} | "
                  f"{mat:>6.2f} | {spd:>7.2f}x | {src}",
                  file=sys.stderr)
            if spd > best[p][1]:
                best[p] = (B, spd, src)

    print("\n-- Best speedup per method --", file=sys.stderr)
    for p in prefixes:
        b, s, src = best[p]
        src_tag = f" ({src})" if src else ""
        print(f"  {p:<{label_w}}: budget={b:>4}, speedup={s:.2f}x{src_tag}",
              file=sys.stderr)
    print("=" * 90, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--union-trie-data", required=True,
                        help="Input JSONL with union trie data + p_t")
    parser.add_argument("--output", default=None,
                        help="Output JSON for simulation results")
    parser.add_argument("--budgets", default="1,2,4,8,16,32,64",
                        help="Comma-separated budget values for sweep")
    parser.add_argument("--p-t-key", default="p_t_oracle",
                        help="Key for p_t values in records (default: p_t_oracle)")
    parser.add_argument("--latency-config", required=True,
                        help="Path to latency_config.json (from measure_verify_latency.py). "
                             "Required: use measure_sglang_verify_latency.py to generate it.")
    parser.add_argument("--draft-latencies", default=None,
                        help="Per-proposer draft cost in ms, e.g. "
                             "'eagle3=3.3,mtp=2.0,suffix=0.3'. "
                             "Multi-proposer methods use max() (parallel).")
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--enable-eu", action="store_true",
                        help="Enable EU Oracle (slow tree knapsack DP)")
    args = parser.parse_args()

    if not args.output and not args.print_summary:
        parser.error("At least one of --output or --print-summary required")

    budgets = [int(b) for b in args.budgets.split(",")]

    # Load records
    print(f"Loading: {args.union_trie_data}", file=sys.stderr)
    records = []
    with open(args.union_trie_data) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} step records", file=sys.stderr)

    # If p_t_oracle not present, compute it
    if records and args.p_t_key == "p_t_oracle" and "p_t_oracle" not in records[0]:
        from simulation.pipeline.collect_target_probs import (
            enrich_with_ground_truth_p_t,
        )
        enrich_with_ground_truth_p_t(records)
        print("Computed oracle p_t from ground truth", file=sys.stderr)

    # Evaluate Choose-One Oracle (unconstrained)
    t0 = time.time()
    choose_one = evaluate_choose_one(records)
    print(f"Choose-One Oracle: {time.time() - t0:.2f}s", file=sys.stderr)

    # Evaluate Choose-One at each budget (for fair comparison)
    t0 = time.time()
    choose_one_budget = evaluate_choose_one_at_budget(records, budgets)
    print(f"Choose-One at budget: {time.time() - t0:.2f}s", file=sys.stderr)

    # Latency-aware simulation (all costs from latency_config)
    with open(args.latency_config) as f:
        latency_config = json.load(f)
    latency_results = compute_latency_speedup(
        records, budgets, latency_config, p_t_key=args.p_t_key,
        enable_eu=args.enable_eu)

    if args.print_summary:
        print_summary(choose_one, {}, choose_one_budget,
                      budgets, args.p_t_key)
        try:
            print_latency_summary(latency_results, budgets,
                                  latency_config["vanilla_step_ms"])
        except Exception as e:
            print(f"Warning: print_latency_summary failed: {e}",
                  file=sys.stderr)

    if args.output:
        output = {
            "metadata": {
                "union_trie_data": args.union_trie_data,
                "n_steps": len(records),
                "budgets": budgets,
                "p_t_key": args.p_t_key,
            },
            "choose_one": {
                "aggregate": choose_one["aggregate"],
                # Omit per_step for compactness (can be large)
            },
            "choose_one_budget": {
                "budget_sweep": [
                    {
                        "budget": B,
                        "avg_acc": choose_one_budget[B]["avg_acc"],
                    }
                    for B in budgets
                ],
            },
        }

        proposers = _discover_proposers(records)
        pairs = [f"{proposers[i]}+{proposers[j]}"
                 for i in range(len(proposers))
                 for j in range(i + 1, len(proposers))]
        all_methods = proposers + pairs
        output["latency"] = {
            "vanilla_step_ms": latency_config["vanilla_step_ms"],
            "proposers": proposers,
            "pairs": pairs,
            "budget_sweep": [
                {
                    "budget": B,
                    "target_forward_ms": latency_results[B].get("target_forward_ms", 0),
                    "eagle3_draft_ms": latency_results[B].get("eagle3_draft_ms", 0),
                    **{
                        k: v for k, v in latency_results[B].items()
                        if k != 'budget'
                           and k not in ('target_forward_ms', 'eagle3_draft_ms')
                    },
                }
                for B in budgets if B in latency_results
            ],
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
