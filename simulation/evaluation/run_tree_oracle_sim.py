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
        --output simulation/results/.../tree_oracle_sim.json \
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
    real_step_target_fn=None,
    real_step_draft_only_ms: Optional[float] = None,
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
    real_step_target_fn: Optional callable (int → float). When provided,
        extension methods compute real cost per step using
        ``real_step_target_fn(ext_tree_size) + real_step_draft_only_ms``
        (instead of the flat ``real_step_cost_ms``) so that target-forward
        latency scales with the actual extended tree size — extension can
        verify far more tokens per step than the base EAGLE3 budget B.
    real_step_draft_only_ms: draft-only cost (EAGLE3 draft + B×suffix_speculate)
        that complements real_step_target_fn.
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

    # Determine if this method has draft cost.
    is_hybrid = method.startswith("hybrid_e3:") or method.startswith("hybrid_dm:")
    # ``no_draft`` is used ONLY by the ratio-based cost model to represent
    # "this method has zero draft overhead" (single:suffix's draft is
    # CPU-side, overlapped with target forward). The real-cost accumulator
    # ignores this flag — it always uses real_step_cost_ms which the caller
    # computes as target_forward(B) + draft_cost (so suffix-only still pays
    # target_forward[B], just with draft_cost ≈ 0).
    no_draft = method == "single:suffix"

    ratios = draft_ratios or []
    # Ratio-based time accumulation
    time_per_ratio = {r: 0.0 for r in ratios}
    # For hybrid: conditional (draft only on fallback) + always (draft every step)
    time_per_ratio_always = {r: 0.0 for r in ratios} if is_hybrid else None

    # Real-cost accumulators (use measured latencies)
    sfx_cost_ms = real_step_cost_suffix_ms if real_step_cost_suffix_ms is not None else vanilla_latency_ms
    total_time_real_ms = 0.0 if real_step_cost_ms is not None else None
    # Breakdown: per-step (target_forward part, draft-only part, tokens
    # fed to target forward = ext_size). Populated only when real-cost is
    # computed via the dynamic ext_size path.
    total_target_ms = 0.0 if real_step_cost_ms is not None else None
    total_draft_ms = 0.0 if real_step_cost_ms is not None else None
    total_target_tokens = 0 if real_step_cost_ms is not None else None
    # Per-step ext_size distribution (for variance / box plots).
    target_tokens_sq = 0 if real_step_cost_ms is not None else None
    target_tokens_min = None
    target_tokens_max = None
    total_time_real_always_ms = 0.0 if (real_step_cost_ms is not None and is_hybrid) else None

    total_generated = 0
    total_accepted = 0
    total_steps = 0
    total_time_ms = 0.0
    v_ms = vanilla_latency_ms

    # Fresh SuffixDecodingCache PER METHOD (i.e. per simulate_decoding call).
    # Global tree is shared across requests within this method, matching
    # Stage 3a's `collect_suffix_drafts.py` behavior (one cache for the
    # whole run). Per-request LOCAL tree is reset via start_request below.
    # This prevents cross-method state leakage (oracle vs realistic getting
    # different speculate() results) while preserving in-method global-tree
    # accumulation that's essential for suffix cache's purpose.
    if suffix_cache is not None:
        from hybrid_spec_decoding.suffix_decoding.suffix_tree import (
            SuffixDecodingCache as _FreshCache,
        )
        local_cache = _FreshCache(
            max_tree_depth=64, max_cached_requests=100000,
        )
    else:
        local_cache = None

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
        # Per-request LOCAL reset. Global tree from previous requests
        # in this method is retained (matches Stage 3a).
        if local_cache is not None:
            prompt = first_rec.get("context_token_ids", [])
            local_cache.start_request(
                cache_req_id, np.array(prompt, dtype=np.int32))

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
            ext_size = None  # set by extension_* branches; used for real cost
            if method == "union_trie":
                accepted = _union_trie_step(rec, budget)
            elif method.startswith("union_trie:"):
                names = frozenset(method.split(":", 1)[1].split(","))
                accepted = _union_trie_step(rec, budget, proposer_set=names)
            elif method == "extension":
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3")
            elif method == "extension_oracle":
                # Oracle: full base tree still goes through target verify
                # (EAGLE3 draft is always produced and loaded on GPU), but
                # only the accepted suffix extension tokens are charged —
                # i.e. "if the right suffix chain were known a priori, we
                # would skip verifying the alternative suffix candidates".
                #   ext_size_oracle = base_size + accepted_in_suffix
                accepted, _ = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3")
                ext_size = (_extension_step._last_base_size
                            + _extension_step._last_accepted_suffix)
            elif method == "extension_dmsfx":
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model")
            elif method == "extension_dmsfx_oracle":
                accepted, _ = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model")
                ext_size = (_extension_step._last_base_size
                            + _extension_step._last_accepted_suffix)
            elif method.startswith("extension_by_count:"):
                # Count cap = B × count_ratio (ratio > 1). Base tree stays
                # at budget B; suffix extensions fill up to C = B × ratio.
                ratio = float(method.split(":", 1)[1])
                cap = max(1, int(round(budget * ratio)))
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3", max_count=cap)
            elif method.startswith("extension_dmsfx_by_count:"):
                ratio = float(method.split(":", 1)[1])
                cap = max(1, int(round(budget * ratio)))
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model", max_count=cap)
            elif method.startswith("extension_by_count_score:"):
                # Combo: count cap (C = B × ratio) + score filter.
                _, rest = method.split(":", 1)
                ratio_s, t_s = rest.split(":", 1)
                ratio = float(ratio_s)
                thr = float(t_s)
                cap = max(1, int(round(budget * ratio)))
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3", max_count=cap,
                    score_threshold=thr)
            elif method.startswith("extension_dmsfx_by_count_score:"):
                _, rest = method.split(":", 1)
                ratio_s, t_s = rest.split(":", 1)
                ratio = float(ratio_s)
                thr = float(t_s)
                cap = max(1, int(round(budget * ratio)))
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model", max_count=cap,
                    score_threshold=thr)
            elif method.startswith("extension_by_score:"):
                threshold = float(method.split(":", 1)[1])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3", score_threshold=threshold)
            elif method.startswith("extension_dmsfx_by_score:"):
                threshold = float(method.split(":", 1)[1])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model", score_threshold=threshold)
            elif method.startswith("extension_by_pathprob:"):
                t = float(method.split(":", 1)[1])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3", pathprob_threshold=t)
            elif method.startswith("extension_by_pt:"):
                t = float(method.split(":", 1)[1])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3", pt_threshold=t)
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
                # Suffix branch: target verifies the full suffix tree (no
                # budget truncation). Expose its size so the cost path can
                # interpolate target_forward(actual_sfx_size) instead of
                # under-charging target_forward(B).
                if used_suffix:
                    sfx_tree = rec.get("per_proposer", {}).get("suffix", {})
                    tok_ids = sfx_tree.get("token_ids") or []
                    # _hybrid_step only returns used_suffix=True when
                    # token_ids is non-empty, but defensively fall back to
                    # ext_size=1 (vanilla-sized verify) if it is.
                    ext_size = len(tok_ids) if tok_ids else 1
            elif method.startswith("single:"):
                proposer_name = method.split(":", 1)[1]
                accepted = _single_proposer_step(rec, budget, proposer_name,
                                                 p_t_key)
                # Target-forward cost should scale with the actually-verified
                # tree size, not the budget B:
                #   * suffix: full tree (no budget truncation)
                #   * eagle3: min(B, actual_tree_size) — actual is bounded
                #     by num_draft_tokens (≈ 256), so at B=512 we over-charge
                #     target_forward(512) vs the correct target_forward(256).
                #   * draft_model: min(B, actual_tree_size) — already capped
                #     at 16 tokens in Stage 3b.
                # Empty tree → vanilla-sized verify (ext_size=1).
                tree_data = rec.get("per_proposer", {}).get(proposer_name, {})
                tok_ids = tree_data.get("token_ids") or []
                if not tok_ids:
                    ext_size = 1
                elif proposer_name == "suffix":
                    ext_size = len(tok_ids)
                else:  # eagle3, draft_model, mtp, …
                    ext_size = min(budget, len(tok_ids))
            elif method.startswith("subset:"):
                names = method.split(":", 1)[1].split(",")
                accepted, chosen_size = _subset_step(rec, budget, names,
                                                    p_t_key)
                # chosen_size == 0 → all proposers had empty trees this
                # step, so realistically nothing to verify — vanilla cost.
                ext_size = chosen_size if chosen_size > 0 else 1
            else:  # choose_one
                accepted, chosen_size = _choose_one_step(rec, budget, p_t_key)
                ext_size = chosen_size if chosen_size > 0 else 1

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
                # Dynamic target path: methods that verify a tree whose size
                # isn't bounded by the EAGLE3 budget (extension, single:suffix,
                # hybrid's suffix branch, c1, c1_e3sfx). ext_size was set
                # earlier in the dispatch to the step's actual verified tree
                # size. real_step_draft_only_ms is the draft-only cost that
                # doesn't depend on ext_size.
                if (ext_size is not None and real_step_target_fn is not None
                        and real_step_draft_only_ms is not None):
                    target_ms_step = real_step_target_fn(ext_size)
                    step_real = target_ms_step + real_step_draft_only_ms
                    total_time_real_ms += step_real
                    total_target_ms += target_ms_step
                    total_draft_ms += real_step_draft_only_ms
                    total_target_tokens += ext_size
                    target_tokens_sq += ext_size * ext_size
                    if target_tokens_min is None or ext_size < target_tokens_min: target_tokens_min = ext_size
                    if target_tokens_max is None or ext_size > target_tokens_max: target_tokens_max = ext_size
                    # For hybrid: the 'always' variant assumes draft every
                    # step (no suffix shortcut), so it still uses the flat
                    # fallback cost.
                    if is_hybrid and total_time_real_always_ms is not None:
                        total_time_real_always_ms += real_step_cost_ms
                elif is_hybrid:
                    # Fallback branch of hybrid (used_suffix=False), or
                    # no dynamic cost wired in: use the flat fallback cost.
                    if used_suffix:
                        total_time_real_ms += sfx_cost_ms
                        total_target_ms += sfx_cost_ms * 0.85   # rough split
                        total_draft_ms += sfx_cost_ms * 0.15
                        total_target_tokens += budget
                        target_tokens_sq += budget * budget
                        if target_tokens_min is None or budget < target_tokens_min: target_tokens_min = budget
                        if target_tokens_max is None or budget > target_tokens_max: target_tokens_max = budget
                    else:
                        total_time_real_ms += real_step_cost_ms
                        total_target_ms += real_step_cost_ms * 0.85
                        total_draft_ms += real_step_cost_ms * 0.15
                        total_target_tokens += budget
                        target_tokens_sq += budget * budget
                        if target_tokens_min is None or budget < target_tokens_min: target_tokens_min = budget
                        if target_tokens_max is None or budget > target_tokens_max: target_tokens_max = budget
                    if total_time_real_always_ms is not None:
                        total_time_real_always_ms += real_step_cost_ms
                else:
                    # single:eagle3, single:draft_model, union_trie, eu —
                    # flat real_step_cost_ms (verified size == B in all
                    # these cases). Approximate split from B-dependent
                    # target_forward + draft_only (not tracked directly).
                    total_time_real_ms += real_step_cost_ms
                    # Best-effort split: full B tokens to target, draft
                    # portion unknown here so lump it into target for
                    # the breakdown.
                    total_target_ms += real_step_cost_ms
                    total_target_tokens += budget
                    target_tokens_sq += budget * budget
                    if target_tokens_min is None or budget < target_tokens_min: target_tokens_min = budget
                    if target_tokens_max is None or budget > target_tokens_max: target_tokens_max = budget

            # Feed accepted tokens to suffix cache
            if local_cache is not None:
                gt = rec.get("ground_truth_future", [])
                if gt and advance <= len(gt):
                    local_cache.add_active_response(
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

        if local_cache is not None:
            local_cache.stop_request(cache_req_id)

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
        result["total_time_real_ms"] = total_time_real_ms
        result["total_target_ms"] = total_target_ms
        result["total_draft_ms"] = total_draft_ms
        result["total_target_tokens"] = total_target_tokens
        result["total_target_tokens_sq"] = target_tokens_sq
        result["total_target_tokens_min"] = target_tokens_min
        result["total_target_tokens_max"] = target_tokens_max
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
                    base_proposer: str = "eagle3",
                    score_threshold: Optional[float] = None,
                    max_count: Optional[int] = None,
                    pathprob_threshold: Optional[float] = None,
                    pt_threshold: Optional[float] = None):
    """Extension: base proposer's tree (truncated to budget) + suffix extension
    at every node.

    For EVERY node in the base tree, trace root→node path, build extended context,
    and call suffix_cache.speculate() to extend. Then greedy walk on the combined
    (base + suffix extensions) tree.

    Returns ``(accepted, ext_tree_size)`` — accepted token count and total
    nodes in the extended tree (needed so the target-verify cost can scale
    with the actually-verified tree size, not just the EAGLE3 base budget).

    base_proposer: "eagle3" (default) or "draft_model".
    Filtering strategies (pick one at a time; orthogonal to max_count):
      score_threshold  — attach only if suffix ``draft.score >= t_score``.
      pathprob_threshold — attach only if
                         ``product(p_t along root→node) × draft.score >= t``
                         (weights deeper nodes less since reaching them
                         requires all ancestors to also be accepted).
    max_count: overall extended-tree size cap (stops extending once
        len(ext_tids) >= max_count). Combines with any filter above.
    """
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0, 0

    base = rec.get("per_proposer", {}).get(base_proposer)
    if not base or not base.get("token_ids"):
        return 0, 0

    tids = base["token_ids"]
    pids = base["parents"]

    # Base tree always truncated to budget. The count cap (max_count)
    # is set by the caller; when max_count > budget, len(ext_tids) can
    # grow beyond the base tree via suffix extensions (up to max_count).
    n = min(budget, len(tids))
    tids = tids[:n]
    pids = pids[:n]
    pids = [p if p < n else -1 for p in pids]

    # Build extended tree
    ext_tids = list(tids)
    ext_pids = list(pids)

    base_context = rec.get("context_token_ids")
    if base_context is None or suffix_cache is None:
        return greedy_tree_walk(ext_tids, ext_pids, gt), len(ext_tids)

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

    # EAGLE3 draft-side path probability (root→node cumulative). Captured
    # at Stage 1 by the oracle_patch organize_draft_results tracer, so it
    # is available PRE-verify — realistic to use as a filter signal.
    # Shape: list[n] with path_p_t[0] == 1.0 (root) and path_p_t[i] >= 0.
    path_draft_p_t_raw = (base.get("path_draft_p_t")
                          if isinstance(base, dict) else None)
    if (path_draft_p_t_raw is not None
            and len(path_draft_p_t_raw) < n):
        path_draft_p_t_raw = None  # length mismatch → disable filter

    # Derive per-edge p_t from path_draft_p_t via division by parent's
    # cumulative: p_t[i] = path_p_t[i] / path_p_t[parent]. node_p_t stays
    # None when draft-side p_t is unavailable (e.g. mango3 artifacts that
    # predate the capture) — filters needing it are then skipped.
    node_p_t = None
    path_p_t = None
    if path_draft_p_t_raw is not None:
        path_p_t = [float(path_draft_p_t_raw[i] or 0.0) for i in range(n)]
        node_p_t = [1.0] * n
        for i in range(n):
            parent = pids[i]
            parent_path = path_p_t[parent] if parent >= 0 else 1.0
            if parent_path > 1e-12:
                node_p_t[i] = path_p_t[i] / parent_path
            else:
                node_p_t[i] = 0.0

    allowed_nodes = None

    # Trie-invariant children index: maps parent_idx → {token_id: child_idx}.
    # Populated with the base tree first; suffix extensions then merge
    # into this structure so that a (parent, token) pair never occurs
    # twice in the extended tree (deduplicates base/suffix overlap).
    children = {}
    for i in range(len(ext_tids)):
        p = ext_pids[i]
        tok = ext_tids[i]
        children.setdefault(p, {})[tok] = i

    # Virtual-root extension: speculate from base_context alone (no base
    # tree prefix) and graft the returned suffix tree as root-level
    # children of the extended tree (tree_parent=-1). Without this,
    # extension's root-children = eagle3's root-children only, so when
    # eagle3 misses at the first position the greedy walk terminates
    # before it can reach any deeper suffix extension. Adding this
    # ensures extension ≥ single:suffix at the same step (modulo cache
    # state): suffix's root predictions become siblings to eagle3's
    # root predictions in the extended tree.
    try:
        _root_draft = suffix_cache.speculate(
            cache_req_id,
            np.array(base_context, dtype=np.int32),
            max_spec_tokens=256,
            max_spec_factor=4.0,
            min_token_prob=0.0,
            use_tree_spec=True,
        )
    except Exception:
        _root_draft = None
    if _root_draft is not None and _root_draft.token_ids:
        _root_local = {}
        for _j, (_tid, _pid) in enumerate(
                zip(_root_draft.token_ids, _root_draft.parents)):
            if _pid == -1:
                _tparent = -1
            else:
                _tparent = _root_local.get(_pid)
                if _tparent is None:
                    break  # malformed draft — abort this chain
            _existing = children.get(_tparent, {}).get(_tid)
            if _existing is not None:
                _root_local[_j] = _existing
                continue
            if max_count is not None and len(ext_tids) >= max_count:
                break
            _new_idx = len(ext_tids)
            ext_tids.append(_tid)
            ext_pids.append(_tparent)
            children.setdefault(_tparent, {})[_tid] = _new_idx
            _root_local[_j] = _new_idx

    for node_idx in range(n):
        if max_count is not None and len(ext_tids) >= max_count:
            break  # hit the overall tree-size cap before iterating this node

        if allowed_nodes is not None and node_idx not in allowed_nodes:
            continue  # ptopk filter

        ext_context = np.array(base_context + paths[node_idx], dtype=np.int32)

        max_spec = 256

        try:
            draft = suffix_cache.speculate(
                cache_req_id, ext_context,
                max_spec_tokens=max_spec,
                max_spec_factor=4.0,
                min_token_prob=0.0,
                use_tree_spec=True,
            )
        except Exception:
            continue

        if not draft.token_ids:
            continue
        draft_score = float(getattr(draft, "score", 0.0))
        if score_threshold is not None and draft_score < score_threshold:
            continue
        if pathprob_threshold is not None and path_p_t is not None:
            if draft_score * path_p_t[node_idx] < pathprob_threshold:
                continue
        if pt_threshold is not None and path_p_t is not None:
            # EAGLE3 path_p_t alone (no suffix score multiplier) —
            # "how likely to reach this node if earlier drafts are all accepted".
            if path_p_t[node_idx] < pt_threshold:
                continue

        # Attach suffix chain with dedup. Each draft token is checked
        # against the current children[tree_parent] map; if the same
        # token already exists under that parent (backbone or previously-
        # merged suffix), reuse it — otherwise append. local_to_tree
        # threads parent-index resolution for multi-token chains.
        # Assumes draft.parents is topologically ordered (parent idx <
        # child idx) — sglang's SuffixDecodingCache returns BFS.
        local_to_tree = {}
        for j, (tid, pid) in enumerate(zip(draft.token_ids, draft.parents)):
            if pid == -1:
                tree_parent = node_idx
            else:
                tree_parent = local_to_tree.get(pid)
                if tree_parent is None:
                    break  # malformed draft — abort this chain
            existing = children.get(tree_parent, {}).get(tid)
            if existing is not None:
                local_to_tree[j] = existing  # merge into existing node
                continue
            if max_count is not None and len(ext_tids) >= max_count:
                break  # cap reached — stop adding new nodes
            new_idx = len(ext_tids)
            ext_tids.append(tid)
            ext_pids.append(tree_parent)
            children.setdefault(tree_parent, {})[tid] = new_idx
            local_to_tree[j] = new_idx

    # Inline greedy walk that also tracks how many accepted steps reside
    # in the base portion (node_idx < n). Once the walk transitions into
    # suffix (node_idx >= n), all subsequent accepts are suffix. Needed
    # for the realistic oracle which charges full base + accepted suffix.
    from collections import defaultdict as _dd
    _children = _dd(list)
    for _i, _p in enumerate(ext_pids):
        _children[_p].append(_i)
    _node = -1
    _acc = 0
    _acc_base = 0
    for _t in gt:
        _picked = None
        for _c in _children.get(_node, []):
            if ext_tids[_c] == _t:
                _picked = _c
                break
        if _picked is None:
            break
        _acc += 1
        if _picked < n:
            _acc_base += 1
        _node = _picked

    # Stash the breakdown on the function so the oracle dispatch can use
    # it (side-channel — avoids widening the return signature which many
    # existing callers unpack as a 2-tuple).
    _extension_step._last_base_size = n
    _extension_step._last_ext_size_full = len(ext_tids)
    _extension_step._last_accepted_base = _acc_base
    _extension_step._last_accepted_suffix = _acc - _acc_base
    return _acc, len(ext_tids)


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


def _choose_one_step(rec: dict, budget: int, p_t_key: str = "p_t"):
    """Choose-One Oracle: try each proposer's full tree, pick best.

    No budget truncation — each proposer uses its full tree.
    This is an oracle that picks the best proposer per step.

    Returns ``(accepted, chosen_tree_size)`` — the tree size of the winning
    proposer so target-forward cost can be interpolated against the actual
    verified tree (not the EAGLE3 budget B, which does not bound suffix).
    """
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0, 0

    best_acc = 0
    best_size = 0
    for name, tree_data in rec.get("per_proposer", {}).items():
        if not tree_data or not tree_data.get("token_ids"):
            continue
        acc = greedy_tree_walk(tree_data["token_ids"], tree_data["parents"], gt)
        if acc > best_acc:
            best_acc = acc
            best_size = len(tree_data["token_ids"])
    return best_acc, best_size


def _subset_step(rec: dict, budget: int, proposer_names: List[str],
                 p_t_key: str = "p_t"):
    """Choose-One over a subset of proposers. Full trees, no truncation.

    Returns ``(accepted, chosen_tree_size)`` — see _choose_one_step.
    """
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0, 0

    best_acc = 0
    best_size = 0
    for name in proposer_names:
        tree_data = rec.get("per_proposer", {}).get(name)
        if not tree_data or not tree_data.get("token_ids"):
            continue
        acc = greedy_tree_walk(tree_data["token_ids"], tree_data["parents"], gt)
        if acc > best_acc:
            best_acc = acc
            best_size = len(tree_data["token_ids"])
    return best_acc, best_size


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
    enable_union_trie: bool = True,
    topk: Optional[int] = None,
    steps: Optional[int] = None,
) -> dict:
    """Run step-by-step simulation for each budget with measured latencies.

    Returns per-budget simulation results including speedup.

    Parameters
    ----------
    enable_eu : bool
        Enable EU Oracle (tree knapsack DP). Slow for large trees. Default off.
    enable_union_trie : bool
        Enable ``union_trie_*`` methods. When False (``UNION_TRIE=0`` path),
        both the union-trie methods and EU methods are skipped — EU relies
        on the precomputed ``union_trie``/``source_map`` fields which are
        absent in that mode. Default True.

    Latency config should contain decomposed costs:
        vanilla_step_ms: target TPOT with no speculation
        target_forward_ms: {B: ms} — pure target verify cost for B tokens
        eagle3_draft_ms: {B: ms} — EAGLE3 draft generation cost
        draft_lm_tpot_ms: draft model per-token cost
        suffix_speculate_ms: per-call cost of SuffixDecodingCache.speculate()

    Missing budgets in the per-B tables are linearly interpolated using the
    nearest measured bracket (and clamped at the extremes).

    Backward compatible: derives from legacy verify_latencies_ms if needed.
    """
    vanilla_ms = latency_config["vanilla_step_ms"]
    proposers = _discover_proposers(records)

    # --- Decomposed latencies ---
    # target_forward_ms[B]: pure target model verify cost for B tokens
    # eagle3_draft_ms[B]: EAGLE3 draft generation cost for B tokens
    #
    # Topk-aware tables (new schema):
    #   target_forward_ms_by_topk[K][B]
    #   eagle3_draft_ms_by_topk_steps[K][S][B]
    # When `topk` is supplied and the per-topk table exists, use it. Else
    # fall back to the legacy flat tables (cross-topk median / canonical topk).
    tfwd_by_topk = latency_config.get("target_forward_ms_by_topk", {}) or {}
    e3draft_by_ts = latency_config.get("eagle3_draft_ms_by_topk_steps", {}) or {}

    def _pick_topk_table(table_by_k: dict, label: str) -> dict:
        if not table_by_k:
            return {}
        if topk is None:
            return {}
        key = str(int(topk))
        if key in table_by_k:
            return dict(table_by_k[key])
        # Nearest-topk fallback
        avail = sorted(int(k) for k in table_by_k.keys())
        nearest = min(avail, key=lambda k: abs(k - int(topk)))
        print(f"WARN: {label} has no topk={topk} entry; using nearest "
              f"measured topk={nearest} (available={avail})", file=sys.stderr)
        return dict(table_by_k[str(nearest)])

    target_fwd = _pick_topk_table(tfwd_by_topk, "target_forward_ms_by_topk")
    if not target_fwd:
        target_fwd = dict(latency_config.get("target_forward_ms", {}))

    eagle3_draft: dict = {}
    if e3draft_by_ts and topk is not None:
        key_k = str(int(topk))
        if key_k not in e3draft_by_ts:
            avail = sorted(int(k) for k in e3draft_by_ts.keys())
            nearest = min(avail, key=lambda k: abs(k - int(topk)))
            print(f"WARN: eagle3_draft_ms_by_topk_steps has no topk={topk}; "
                  f"using nearest={nearest}", file=sys.stderr)
            key_k = str(nearest)
        per_steps = e3draft_by_ts.get(key_k, {}) or {}
        if per_steps and steps is not None:
            key_s = str(int(steps))
            if key_s in per_steps:
                eagle3_draft = dict(per_steps[key_s])
            else:
                avail_s = sorted(int(s) for s in per_steps.keys())
                if avail_s:
                    nearest_s = min(avail_s, key=lambda s: abs(s - int(steps)))
                    print(f"WARN: eagle3_draft_ms_by_topk_steps[{key_k}] has "
                          f"no steps={steps}; using nearest={nearest_s}",
                          file=sys.stderr)
                    eagle3_draft = dict(per_steps[str(nearest_s)])

    if not eagle3_draft:
        # Fall back to legacy flat table (canonical topk/steps from compile)
        eagle3_draft = dict(latency_config.get("eagle3_draft_ms", {}))

    legacy_verify = latency_config.get("verify_latencies_ms",
                                       latency_config.get("eagle3_step_ms", {}))

    if not target_fwd and legacy_verify:
        # Derive from legacy: target_forward ≈ vanilla, eagle3_draft = remainder
        for b_str, step in legacy_verify.items():
            target_fwd[b_str] = vanilla_ms
            eagle3_draft[b_str] = max(float(step) - vanilla_ms, 0.0)

    # Per-proposer draft costs (non-EAGLE3)
    draft_lm_tpot = float(latency_config.get("draft_lm_tpot_ms", 0.0) or 0.0)
    suffix_speculate_ms = float(
        latency_config.get("suffix_speculate_ms", 0.0) or 0.0)
    # Draft-model chain length cap. Stage 3b (collect_draft_model.py) hard-codes
    # --max-draft-tokens=16; anything above that is filled by other proposers,
    # not by more draft forwards.
    MAX_DRAFT_MODEL_N = int(latency_config.get("max_draft_model_n", 16))

    def _interp(table: dict, B: int, fallback: float) -> float:
        """Linear interpolation on measured budgets.

        Within the measured range: standard piecewise-linear interp.
        Below the smallest key: clamp at that key's value (target_forward
        cannot be meaningfully below the vanilla-step cost).
        Above the largest key: linear extrapolation using the two largest
        measurements. Extension methods may need this because the extended
        tree size (base + suffix drafts at every node) often exceeds the
        largest measured budget — e.g. B=16 base × 50 suffix extensions
        per node ≈ 800 tokens to verify.
        """
        if not table:
            return fallback
        key = str(B)
        if key in table:
            return float(table[key])
        keys = sorted(int(k) for k in table.keys())
        if B <= keys[0]:
            # Linear interpolation from (B=1, vanilla_ms) up to the
            # smallest measured key. Previously this clamped to the
            # smallest key which made suffix cost flat for tiny trees.
            if B <= 1:
                return fallback
            v_at_small = float(table[str(keys[0])])
            frac = (B - 1) / (keys[0] - 1)
            return fallback + frac * (v_at_small - fallback)
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

    def _target_forward(B: int) -> float:
        """Pure target model forward cost for verifying B tokens."""
        return _interp(target_fwd, B, vanilla_ms)

    def _eagle3_draft(B: int) -> float:
        """EAGLE3 draft generation cost for budget B."""
        return _interp(eagle3_draft, B, 0.0)

    def _proposer_draft_cost(name: str, B: int,
                             suffix_matches: int = 1) -> float:
        """Draft cost for a single proposer at verify budget B.

        Note on terminology: ``B`` here is the global "verify budget"
        (num_draft_tokens sent to the target model for verification).
        Its interpretation per proposer differs:
          * eagle3:      B = max tree size (branching tree, topk × steps)
          * draft_model: k = linear chain length (capped at
                             MAX_DRAFT_MODEL_N, so effective k = min(B, cap))
          * suffix:      matches × speculate call count (``suffix_matches``)

        ``suffix_matches`` is only meaningful for suffix-family costs: how
        many ``speculate()`` calls a method makes per step. Defaults to 1
        (single / hybrid-suffix path); ``extension`` passes ~B.
        """
        if name == "eagle3":
            return _eagle3_draft(B)
        elif name == "draft_model":
            # Draft model is autoregressive linear: each extra token =
            # one extra forward. Stage 3b caps k at MAX_DRAFT_MODEL_N
            # (=16); for verify budgets above the cap, the remaining slots
            # are filled by the co-proposer (suffix / eagle3) rather than
            # additional draft-model forwards. Using the uncapped B × tpot
            # here previously over-charged extension_dmsfx at high B by ~15×.
            k = min(B, MAX_DRAFT_MODEL_N)
            return k * draft_lm_tpot
        elif name == "suffix":
            return suffix_matches * suffix_speculate_ms
        elif name == "mtp":
            return 0.0  # uses target model MTP heads, cost in target_forward
        return 0.0

    def _step_cost(active_proposers: List[str], B: int) -> float:
        """Step cost = target_forward(B) + max(draft costs of GPU proposers).

        Proposers draft in parallel → cost = max, not sum. Suffix runs on
        CPU in parallel with the target GPU forward, so ``max()`` rather
        than sum is still correct even with non-zero suffix cost (CPU vs GPU
        overlap — suffix rarely dominates max unless extension explodes the
        match count).
        """
        t_fwd = _target_forward(B)
        draft_costs = [_proposer_draft_cost(p, B) for p in active_proposers]
        max_draft = max(draft_costs) if draft_costs else 0.0
        return t_fwd + max_draft

    DRAFT_RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50]

    def _real_cost(active_proposers, B, *, suffix_matches: int = 1,
                   verify_tokens: Optional[int] = None):
        """Step cost in ms using measured latencies.

        target_forward(verify_tokens) + max(parallel draft costs). Suffix is
        kept in the max() now that it has a real per-match cost — it still
        usually costs far less than eagle3/draft_model so rarely dominates,
        but accounting for it here makes extension / extension_by_score
        comparisons fair.

        ``verify_tokens`` overrides the budget used for target_forward
        interpolation. Defaults to B. Pass a smaller value when the base
        proposer is known to emit fewer tokens than the verify budget
        (e.g. single:draft_model where the chain caps at MAX_DRAFT_MODEL_N
        so target only verifies those, not the full B).
        """
        tf = _target_forward(verify_tokens if verify_tokens is not None else B)
        drafts = [_proposer_draft_cost(p, B, suffix_matches=suffix_matches)
                  for p in active_proposers]
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
        # Cost/token breakdowns (per-run totals; per-step = total / steps).
        for k in ("total_time_real_ms", "total_target_ms",
                  "total_draft_ms", "total_target_tokens",
                  "total_target_tokens_sq",
                  "total_target_tokens_min",
                  "total_target_tokens_max"):
            if k in sim:
                entry[f"{prefix}_{k}"] = sim[k]

    if enable_eu and not enable_union_trie:
        print("WARN: enable_eu=True requires enable_union_trie=True; "
              "EU methods disabled.", file=sys.stderr)
        enable_eu = False

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
        if enable_union_trie:
            _run("union_trie_e3sfx", {**common,
                 "method": "union_trie:eagle3,suffix",
                 "real_step_cost_ms": _real_cost(["eagle3", "suffix"], B)},
                 "union_trie_e3sfx")

            if "draft_model" in proposers:
                _run("union_trie_all", {**common,
                     "method": "union_trie:eagle3,suffix,draft_model",
                     "real_step_cost_ms": _real_cost(["eagle3", "suffix", "draft_model"], B)},
                     "union_trie_all")

        # Choose-One Oracle (all proposers, full tree). Target verifies the
        # full tree of whichever proposer wins the step (often suffix,
        # which has no budget truncation), so use dynamic target cost.
        c1_draft_only = max(
            _eagle3_draft(B),
            min(B, MAX_DRAFT_MODEL_N) * draft_lm_tpot,
            suffix_speculate_ms,
        )
        _run("c1",
             {**common, "method": "choose_one",
              "real_step_cost_ms": _real_cost(list(proposers), B),
              "real_step_target_fn": _target_forward,
              "real_step_draft_only_ms": c1_draft_only},
             "c1")

        # Single-proposer baselines. The simulator now uses a per-step
        # dynamic target cost for each single-proposer method (keyed on
        # the actual tree size this step), with a per-method draft-only cost.
        for pname in proposers:
            if pname == "eagle3":
                draft_only = _eagle3_draft(B)
            elif pname == "draft_model":
                draft_only = min(B, MAX_DRAFT_MODEL_N) * draft_lm_tpot
            elif pname == "suffix":
                draft_only = suffix_speculate_ms
            elif pname == "mtp":
                draft_only = 0.0  # MTP overhead baked into target_forward
            else:
                draft_only = 0.0
            # Fallback (used only if dispatch can't set ext_size for any
            # reason): coarse flat cost using budget B.
            if pname == "draft_model":
                verify_n_fallback = min(B, MAX_DRAFT_MODEL_N)
            else:
                verify_n_fallback = None
            kwargs = {**common, "method": f"single:{pname}",
                      "real_step_cost_ms": _real_cost(
                          [pname], B, verify_tokens=verify_n_fallback),
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": draft_only}
            _run(f"single:{pname}", kwargs, pname)

        # Hybrid (suffix score threshold): suffix if score >= t, else fallback.
        # When suffix wins, step cost = target_forward + 1 suffix match.
        thresholds = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

        if "suffix" in proposers and "eagle3" in proposers:
            e3_cost = _real_cost(["eagle3"], B)
            suffix_only_cost = _target_forward(B) + suffix_speculate_ms
            for t in thresholds:
                _run(f"hybrid_e3:{t}",
                     {**common, "method": f"hybrid_e3:{t}",
                      "real_step_cost_ms": e3_cost,
                      "real_step_cost_suffix_ms": suffix_only_cost,
                      # Dynamic target cost when suffix branch is taken —
                      # suffix tree is not budget-truncated so target may
                      # verify many more tokens than B.
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": suffix_speculate_ms},
                     f"hybrid_e3_t{t:.1f}")

        if "suffix" in proposers and "draft_model" in proposers:
            # hybrid_dm's dm-branch: draft is min(B, 16) linear tokens;
            # target only verifies that many.
            dm_cost = _real_cost(["draft_model"], B,
                                 verify_tokens=min(B, MAX_DRAFT_MODEL_N))
            suffix_only_cost = _target_forward(B) + suffix_speculate_ms
            for t in thresholds:
                _run(f"hybrid_dm:{t}",
                     {**common, "method": f"hybrid_dm:{t}",
                      "real_step_cost_ms": dm_cost,
                      "real_step_cost_suffix_ms": suffix_only_cost,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": suffix_speculate_ms},
                     f"hybrid_dm_t{t:.1f}")

        # C1 for suffix+eagle3 pair
        if "suffix" in proposers and "eagle3" in proposers:
            c1e3_draft_only = max(_eagle3_draft(B), suffix_speculate_ms)
            _run("c1_e3sfx",
                 {**common, "method": "subset:eagle3,suffix",
                  "p_t_key": p_t_key,
                  "real_step_cost_ms": _real_cost(["eagle3", "suffix"], B),
                  "real_step_target_fn": _target_forward,
                  "real_step_draft_only_ms": c1e3_draft_only},
                 "c1_e3sfx")

        # Extension: base tree + suffix extension at every node.
        # Real cost per step = target_forward(actual_ext_tree_size)
        #                    + max(base_draft, node_count × suffix_speculate_ms)
        # The target cost is the expensive part and it scales with the full
        # extended tree (not the EAGLE3 base budget B), because target
        # verifies every node. We pass _target_forward as a per-step callable
        # so the simulator can interpolate for any ext_size.
        # Passed as `suffix_cache` arg; simulate_decoding treats non-None
        # as a truthy flag and instantiates a FRESH SuffixDecodingCache
        # per (req, call) iteration internally (see simulate_decoding).
        _SUFFIX_ENABLED = object()  # sentinel, any truthy non-None value
        if "suffix" in proposers and "eagle3" in proposers:
            # Approximate eagle3 base-tree size: capped by B, typically
            # around topk × steps (pipeline default topk=16, steps ∈ {2..8}).
            e3_nodes = min(B, 16 * 8)
            # Draft-only part: EAGLE3 forward + B suffix speculate calls,
            # overlapped (max). Constant per step regardless of ext_size.
            ext_draft_only = max(
                _eagle3_draft(B),
                e3_nodes * suffix_speculate_ms,
            )
            # Fallback cost if ext_size somehow isn't observed (shouldn't happen):
            ext_cost_fallback = _target_forward(B) + ext_draft_only
            _run("extension",
                 {**common, "method": "extension",
                  "suffix_cache": _SUFFIX_ENABLED,
                  "real_step_cost_ms": ext_cost_fallback,
                  "real_step_target_fn": _target_forward,
                  "real_step_draft_only_ms": ext_draft_only},
                 "extension")
            # Extension Oracle: same draft cost, but target verifies only
            # the accepted path (best-case scenario where the useful suffix
            # chains were known in advance). Serves as an upper bound.
            _run("extension_oracle",
                 {**common, "method": "extension_oracle",
                  "suffix_cache": _SUFFIX_ENABLED,
                  "real_step_cost_ms": ext_cost_fallback,
                  "real_step_target_fn": _target_forward,
                  "real_step_draft_only_ms": ext_draft_only},
                 "extension_oracle")
            # Extension by count cap. Base tree stays at budget B; total
            # tree (base + suffix extensions) capped at C = B × ratio.
            # ratio must be > 1 to leave room for suffix beyond the base
            # tree; ratio ≤ 1 degenerates to single:eagle3.
            for r in [2, 4, 8]:
                _run(f"extension_by_count:{r}",
                     {**common, "method": f"extension_by_count:{r}",
                      "suffix_cache": _SUFFIX_ENABLED,
                      "real_step_cost_ms": ext_cost_fallback,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": ext_draft_only},
                     f"extension_by_count_r{r}")
            # Combo: count cap (C = B × ratio) + score filter. Only
            # attach suffix when suffix.score ≥ t, total tree still
            # capped at C.
            for r in [2, 4]:
                for t in [1.0, 3.0, 10.0]:
                    _run(f"extension_by_count_score:{r}:{t}",
                         {**common,
                          "method": f"extension_by_count_score:{r}:{t}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": ext_draft_only},
                         f"extension_by_count_score_r{r}_t{t}")
            # Extended score threshold range: prior micro-analysis of
            # extension_oracle showed 97%+ of winning suffix chains have
            # score ≥ 20, so we probe higher thresholds too.
            for t in thresholds + [25.0, 30.0, 35.0]:
                _run(f"extension_by_score:{t}",
                     {**common, "method": f"extension_by_score:{t}",
                      "suffix_cache": _SUFFIX_ENABLED,
                      "real_step_cost_ms": ext_cost_fallback,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": ext_draft_only},
                     f"extension_by_score_t{t:.1f}")
            # p_t–based extension filters (eagle3 base only — require
            # EAGLE3 draft-side path probabilities, captured at Stage 1 by
            # the oracle_patch organize_draft_results tracer).
            # Legacy artifacts (mango3 union_trie-based, no re-collection
            # possible) don't carry path_draft_p_t → skip these methods.
            has_draft_p_t = any(
                (rec.get("per_proposer", {})
                    .get("eagle3", {}) or {}).get("path_draft_p_t") is not None
                for rec in records)
            if not has_draft_p_t:
                print("NOTE: no path_draft_p_t available — skipping "
                      "ptopk/product/pathprob/topp/dynsfx methods",
                      file=sys.stderr)

            # pathprob: attach only when path_p_t × draft.score ≥ t
            # (weights deeper nodes less — closer to Oracle's
            # "accepted path only" ideal).
            if has_draft_p_t:
                for t in [0.001, 0.01, 0.05, 0.1, 0.3]:
                    _run(f"extension_by_pathprob:{t}",
                         {**common, "method": f"extension_by_pathprob:{t}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": ext_draft_only},
                         f"extension_by_pathprob_t{t}")
            # by_pt: same idea without the suffix.score multiplier —
            # "how likely is this node reached" alone, no suffix
            # cache confidence involved.
            if has_draft_p_t:
                for t in [0.001, 0.005, 0.01, 0.05, 0.1, 0.3]:
                    _run(f"extension_by_pt:{t}",
                         {**common, "method": f"extension_by_pt:{t}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": ext_draft_only},
                         f"extension_by_pt_t{t}")

        # Extension with draft_model base + suffix extensions
        if "suffix" in proposers and "draft_model" in proposers:
            # (suffix_cache sentinel enables suffix in simulate_decoding,
            # which creates a fresh cache per (req, call) iteration)
            # Draft-model base tree is a linear chain of exactly
            # min(B, MAX_DRAFT_MODEL_N=16) tokens.
            dm_nodes = min(B, MAX_DRAFT_MODEL_N)
            ext_dm_draft_only = max(
                dm_nodes * draft_lm_tpot,
                dm_nodes * suffix_speculate_ms,
            )
            ext_dm_cost_fallback = _target_forward(B) + ext_dm_draft_only
            _run("extension_dmsfx",
                 {**common, "method": "extension_dmsfx",
                  "suffix_cache": _SUFFIX_ENABLED,
                  "real_step_cost_ms": ext_dm_cost_fallback,
                  "real_step_target_fn": _target_forward,
                  "real_step_draft_only_ms": ext_dm_draft_only},
                 "extension_dmsfx")
            _run("extension_dmsfx_oracle",
                 {**common, "method": "extension_dmsfx_oracle",
                  "suffix_cache": _SUFFIX_ENABLED,
                  "real_step_cost_ms": ext_dm_cost_fallback,
                  "real_step_target_fn": _target_forward,
                  "real_step_draft_only_ms": ext_dm_draft_only},
                 "extension_dmsfx_oracle")
            for r in [2, 4, 8]:
                _run(f"extension_dmsfx_by_count:{r}",
                     {**common, "method": f"extension_dmsfx_by_count:{r}",
                      "suffix_cache": _SUFFIX_ENABLED,
                      "real_step_cost_ms": ext_dm_cost_fallback,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": ext_dm_draft_only},
                     f"extension_dmsfx_by_count_r{r}")
            # Combo: count cap + score filter (DM base).
            for r in [2, 4]:
                for t in [1.0, 3.0, 10.0]:
                    _run(f"extension_dmsfx_by_count_score:{r}:{t}",
                         {**common,
                          "method": f"extension_dmsfx_by_count_score:{r}:{t}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_dm_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": ext_dm_draft_only},
                         f"extension_dmsfx_by_count_score_r{r}_t{t}")
            for t in thresholds:
                _run(f"extension_dmsfx_by_score:{t}",
                     {**common, "method": f"extension_dmsfx_by_score:{t}",
                      "suffix_cache": _SUFFIX_ENABLED,
                      "real_step_cost_ms": ext_dm_cost_fallback,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": ext_dm_draft_only},
                     f"extension_dmsfx_by_score_t{t:.1f}")

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
    # Input: either precomputed --union-trie-data, or raw Stage 1/3
    # artifacts (assembled on the fly). Validated below.
    parser.add_argument("--union-trie-data", default=None,
                        help="Input JSONL with union trie data + p_t "
                             "(produced by Stage 4)")
    parser.add_argument("--agent-results", default=None,
                        help="Alternative to --union-trie-data: Stage 1 "
                             "EAGLE3 results JSON. Records are assembled "
                             "on-the-fly from Stage 1/3 artifacts.")
    parser.add_argument("--suffix-drafts", default=None,
                        help="Stage 3a per-step suffix JSONL "
                             "(used with --agent-results)")
    parser.add_argument("--draft-model-drafts", default=None,
                        help="Stage 3b per-step draft-model JSONL "
                             "(used with --agent-results)")
    parser.add_argument("--mtp-agent-results", default=None,
                        help="Stage 3c MTP agent_results JSON "
                             "(used with --agent-results)")
    parser.add_argument("--dataset", default=None,
                        help="dataset.jsonl for BFCL/SpecBench prompt "
                             "reconstruction (used with --agent-results)")
    parser.add_argument("--responses", default=None,
                        help="agent_results_responses.json "
                             "(used with --agent-results, BFCL only)")
    parser.add_argument("--model", default=None,
                        help="Target model name for tokenizer "
                             "(used with --agent-results)")
    parser.add_argument("--exclude", default=None,
                        help="Exclude-ids file (used with --agent-results)")
    parser.add_argument("--output", default=None,
                        help="Output JSON for simulation results")
    parser.add_argument("--budgets", default="1,2,4,8,16,32,64",
                        help="Comma-separated budget values for sweep")
    parser.add_argument("--p-t-key", default="p_t_oracle",
                        help="Key for p_t values in records (default: p_t_oracle)")
    parser.add_argument("--latency-config", default=None,
                        help="Path to latency_config.json (from measure_decomposed_latency.py). "
                             "When omitted, MAT / accept-rate stats are still "
                             "reported but latency-aware speedup numbers are skipped.")
    parser.add_argument("--draft-latencies", default=None,
                        help="Per-proposer draft cost in ms, e.g. "
                             "'eagle3=3.3,mtp=2.0,suffix=0.3'. "
                             "Multi-proposer methods use max() (parallel).")
    parser.add_argument("--topk", type=int, default=None,
                        help="EAGLE3 topk used for this Stage 1 run. "
                             "When set, latency lookups pull from the "
                             "per-topk tables in latency_config.json "
                             "(target_forward_ms_by_topk / "
                             "eagle3_draft_ms_by_topk_steps).")
    parser.add_argument("--steps", type=int, default=None,
                        help="EAGLE3 num_steps used for this Stage 1 run. "
                             "Used together with --topk to pick the right "
                             "eagle3_draft_ms table.")
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--enable-eu", action="store_true",
                        help="Enable EU Oracle (slow tree knapsack DP). "
                             "Ignored when --no-union-trie is set.")
    parser.add_argument("--no-union-trie", action="store_true",
                        help="Skip union_trie_* methods (and EU, which "
                             "depends on the union trie). Use when Stage 4 "
                             "was skipped (UNION_TRIE=0).")
    args = parser.parse_args()

    if not args.output and not args.print_summary:
        parser.error("At least one of --output or --print-summary required")
    if not args.union_trie_data and not args.agent_results:
        parser.error("Provide --union-trie-data or --agent-results")
    if args.union_trie_data and args.agent_results:
        parser.error("--union-trie-data and --agent-results are mutually exclusive")

    budgets = [int(b) for b in args.budgets.split(",")]
    enable_union_trie = not args.no_union_trie

    # Load records — either directly from Stage 4 JSONL or assemble on the fly
    if args.union_trie_data:
        print(f"Loading: {args.union_trie_data}", file=sys.stderr)
        records = []
        with open(args.union_trie_data) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"Loaded {len(records)} step records", file=sys.stderr)
        input_source = args.union_trie_data
    else:
        from simulation.pipeline.collect_union_trie import (
            assemble_records_from_artifacts,
        )
        records = assemble_records_from_artifacts(
            agent_results_path=args.agent_results,
            suffix_drafts_path=args.suffix_drafts,
            draft_model_drafts_path=args.draft_model_drafts,
            mtp_agent_results_path=args.mtp_agent_results,
            exclude_path=args.exclude,
            model=args.model,
            dataset_path=args.dataset,
            responses_path=args.responses,
            include_union_trie=enable_union_trie,
        )
        input_source = args.agent_results

    # If p_t_oracle requested but missing, compute it from ground truth.
    # The enrichment uses the union_trie field, so it's only relevant when
    # union-trie methods / EU are enabled.
    if (records and args.p_t_key == "p_t_oracle"
            and "p_t_oracle" not in records[0]
            and enable_union_trie):
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

    # Latency-aware simulation. compute_latency_speedup always runs so
    # MAT / accept-rate / method rankings are reported for every method
    # (c1, single:*, hybrid:*, extension, and when enabled union_trie_* /
    # eu). When --latency-config is missing, we feed a stub config
    # (vanilla_step_ms=1.0 with empty per-budget tables); the speedup
    # numbers become meaningless but MAT is unaffected.
    have_latency = bool(args.latency_config)
    if have_latency:
        with open(args.latency_config) as f:
            latency_config = json.load(f)
    else:
        print("NOTE: --latency-config not provided; MAT is still reported "
              "but speedup numbers will be stub values (not measured).",
              file=sys.stderr)
        latency_config = {
            "vanilla_step_ms": 1.0,
            "target_forward_ms": {},
            "eagle3_draft_ms": {},
            "draft_lm_tpot_ms": 0.0,
        }

    latency_results = compute_latency_speedup(
        records, budgets, latency_config, p_t_key=args.p_t_key,
        enable_eu=args.enable_eu, enable_union_trie=enable_union_trie,
        topk=args.topk, steps=args.steps)

    if args.print_summary:
        print_summary(choose_one, {}, choose_one_budget,
                      budgets, args.p_t_key)
        if have_latency:
            try:
                print_latency_summary(latency_results, budgets,
                                      latency_config["vanilla_step_ms"])
            except Exception as e:
                print(f"Warning: print_latency_summary failed: {e}",
                      file=sys.stderr)
        else:
            # Without latency data, still show per-method MAT rankings
            try:
                print_latency_summary(latency_results, budgets,
                                      latency_config["vanilla_step_ms"])
                print("(WARNING: speedup columns above use stub latency; "
                      "only MAT is meaningful)", file=sys.stderr)
            except Exception as e:
                print(f"Warning: MAT summary failed: {e}", file=sys.stderr)

    if args.output:
        output = {
            "metadata": {
                "input_source": input_source,
                "n_steps": len(records),
                "budgets": budgets,
                "p_t_key": args.p_t_key,
                "enable_eu": args.enable_eu and enable_union_trie,
                "enable_union_trie": enable_union_trie,
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
            "has_latency_config": have_latency,
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
        if not have_latency:
            output["latency"]["note"] = (
                "latency_config not provided; MAT values are accurate but "
                "speedup_* columns use stub latencies (not meaningful)")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
