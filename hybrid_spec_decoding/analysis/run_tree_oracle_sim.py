"""Tree-budget oracle simulation for heterogeneous speculative decoding.

Evaluates two oracle strategies on union tries built from multiple
proposers (EAGLE3, Suffix, Draft Model):

1. Choose-One Oracle: Pick the single best proposer's tree per step.
2. Expected-Utility Oracle: DP (tree knapsack) to find the optimal
   subtree under budget B from the union trie.

Input: union_trie_data.jsonl (from collect_union_trie + collect_target_probs)

Usage:
    python -m hybrid_spec_decoding.analysis.run_tree_oracle_sim \
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

from hybrid_spec_decoding.analysis.tree_knapsack import (
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
    print(f"{'Budget':>8} | {'EU(expect)':>11} | {'EU(actual)':>11} | "
          f"{'Choose-1':>9} | {'Gap(act)':>9} | {'Gap%':>7}", file=sys.stderr)
    print("-" * 68, file=sys.stderr)
    for B in budgets:
        eu_exp = eu_results[B]["avg_eu"]
        eu_act = eu_results[B].get("avg_actual_acc", eu_exp)
        c1 = choose_one_budget[B]["avg_acc"]
        gap = eu_act - c1
        gap_pct = (gap / max(c1, 1e-9)) * 100
        print(f"{B:>8} | {eu_exp:>11.4f} | {eu_act:>11.4f} | "
              f"{c1:>9.4f} | {gap:>9.4f} | {gap_pct:>6.1f}%", file=sys.stderr)

    print("=" * 68, file=sys.stderr)


# ---------------------------------------------------------------------------
# Step-by-step simulation (correct skip-ahead behavior)
# ---------------------------------------------------------------------------

def simulate_decoding(
    records: List[dict],
    budget: int,
    method: str,
    p_t_key: str = "p_t_oracle",
    *,
    verify_latency_ms: float,
    vanilla_latency_ms: float,
) -> dict:
    """Simulate actual speculative decoding with skip-ahead.

    Walks through the token sequence position by position.
    At each step:
    1. Select tree (EU Oracle or Choose-One) at current position
    2. Verify against ground truth → accepted tokens
    3. Advance by (accepted + 1) positions (bonus token)
    4. Accumulate latency

    Parameters
    ----------
    records : list[dict]
        Union trie records indexed by (request_id, call_idx, step_idx).
    budget : int
        Number of nodes in the verification tree.
    method : str
        "eu" for Expected-Utility Oracle, "choose_one" for Choose-One.
    p_t_key : str
        Key for p_t values (only used for method="eu").
    verify_latency_ms : float
        Time for one tree verification step. Must be from measured latency config.
    vanilla_latency_ms : float
        Time for one vanilla decode step (baseline). Must be from measured latency config.

    Returns
    -------
    dict with total_tokens, total_steps, total_time_ms, speedup, per_step, etc.
    """
    # Index records by (request_id, call_idx, step_idx)
    record_index: Dict[Tuple, dict] = {}
    # Group by (request_id, call_idx) to find sequence boundaries
    sequences: Dict[Tuple, List[int]] = {}

    for rec in records:
        key = (rec["request_id"], rec.get("call_idx", 0), rec.get("step_idx", 0))
        record_index[key] = rec
        seq_key = (rec["request_id"], rec.get("call_idx", 0))
        sequences.setdefault(seq_key, []).append(rec.get("step_idx", 0))

    # Sort step indices within each sequence
    for sk in sequences:
        sequences[sk].sort()

    total_generated = 0   # accepted + bonus (actual tokens produced)
    total_accepted = 0    # accepted only (no bonus)
    total_steps = 0
    total_time_ms = 0.0
    per_step_log = []

    for seq_key, step_indices in sorted(sequences.items()):
        req_id, call_idx = seq_key
        if not step_indices:
            continue

        max_pos = max(step_indices)
        first_rec = record_index.get((req_id, call_idx, step_indices[0]))
        if not first_rec:
            continue
        # Detect data format to compute total sequence length:
        # NEW format (gt = tokens[pos:]): seq_len = pos + len(gt)
        # OLD format (gt = tokens[pos+1:]): seq_len = pos + 1 + len(gt)
        # Detection: OLD format has gt_len=1 at last step, NEW has gt_len>=2
        last_rec = record_index.get((req_id, call_idx, step_indices[-1]))
        last_gt_len = len(last_rec.get("ground_truth_future", [])) if last_rec else 1
        first_gt_len = len(first_rec.get("ground_truth_future", []))
        if last_gt_len <= 1:
            seq_len = step_indices[0] + 1 + first_gt_len
        else:
            seq_len = step_indices[0] + first_gt_len

        pos = step_indices[0]  # start from first available position
        step_set = set(step_indices)

        while pos <= max_pos and pos in step_set:
            rec = record_index.get((req_id, call_idx, pos))
            if rec is None:
                # No record at this position, advance by 1 (vanilla)
                total_generated += 1
                total_steps += 1
                total_time_ms += vanilla_latency_ms
                pos += 1
                continue

            # Select tree and compute acceptance
            if method == "eu":
                accepted = _eu_step(rec, budget, p_t_key)
            elif method.startswith("eu_pair:"):
                names = method.split(":", 1)[1].split(",")
                accepted = _eu_step(rec, budget, p_t_key,
                                    allowed_proposers=set(names))
            elif method.startswith("hybrid_e3:"):
                threshold = float(method.split(":", 1)[1])
                accepted = _hybrid_step(rec, budget, threshold, fallback="eagle3")
            elif method.startswith("hybrid_dm:"):
                threshold = float(method.split(":", 1)[1])
                accepted = _hybrid_step(rec, budget, threshold, fallback="draft_model")
            elif method.startswith("single:"):
                proposer_name = method.split(":", 1)[1]
                accepted = _single_proposer_step(rec, budget, proposer_name)
            elif method.startswith("subset:"):
                names = method.split(":", 1)[1].split(",")
                accepted = _subset_step(rec, budget, names)
            else:  # choose_one
                accepted = _choose_one_step(rec, budget)

            advance = accepted + 1  # accepted + bonus token
            total_generated += advance
            total_accepted += accepted
            total_steps += 1
            total_time_ms += verify_latency_ms

            per_step_log.append({
                "pos": pos,
                "accepted": accepted,
                "advance": advance,
            })

            pos += advance

        # Remaining tokens after last speculative step (vanilla decode)
        remaining = seq_len - pos
        if remaining > 0:
            total_generated += remaining
            total_steps += remaining
            total_time_ms += remaining * vanilla_latency_ms

    vanilla_time_ms = total_generated * vanilla_latency_ms
    speedup = vanilla_time_ms / total_time_ms if total_time_ms > 0 else 1.0
    mat = total_accepted / total_steps if total_steps > 0 else 0.0

    return {
        "total_generated": total_generated,  # accepted + bonus
        "total_accepted": total_accepted,    # accepted only
        "total_steps": total_steps,
        "total_time_ms": total_time_ms,
        "vanilla_time_ms": vanilla_time_ms,
        "speedup": speedup,
        "mat": mat,  # Mean Accepted Tokens (bonus excluded)
    }


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
    proposer_pairs: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Precompute EU Oracle results for all budgets in one DP pass per record.

    Stores results in rec["_eu_cache"][cache_key][budget] = accepted_tokens.
    This avoids running DP 9x per record (once per budget).
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

        # EU pair variants
        if proposer_pairs:
            for a, b_name in proposer_pairs:
                pair_key = f"{a},{b_name}"
                filtered = _filter_union_trie(rec, {a, b_name}, p_t_key)
                if filtered is None:
                    cache[pair_key] = {b: 0 for b in budgets}
                    continue
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
                cache[pair_key] = budget_accepted

        rec["_eu_cache"] = cache

        if (i + 1) % 5000 == 0 or i == n - 1:
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


def _choose_one_step(rec: dict, budget: int) -> int:
    """Choose-One Oracle: pick best proposer's full tree, return accepted tokens.

    No budget truncation — uses each proposer's entire tree and picks the
    best result. The budget parameter is ignored (kept for API compat).
    """
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0

    best_acc = 0
    for name, tree_data in rec.get("per_proposer", {}).items():
        tids = tree_data["token_ids"]
        pids = tree_data["parents"]
        acc = greedy_tree_walk(tids, pids, gt)
        best_acc = max(best_acc, acc)
    return best_acc


def _hybrid_step(rec: dict, budget: int, threshold: float,
                 fallback: str = "eagle3") -> int:
    """Hybrid: use suffix if score >= threshold, else fallback proposer.

    fallback="eagle3" → suffix + EAGLE3  (hybrid_e3)
    fallback="draft_model" → suffix + draft model  (hybrid_dm)
    """
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0

    per_proposer = rec.get("per_proposer", {})
    suffix_data = per_proposer.get("suffix")
    fallback_data = per_proposer.get(fallback)

    use_suffix = (suffix_data is not None
                  and suffix_data.get("score", 0.0) >= threshold
                  and suffix_data.get("token_ids"))

    if use_suffix:
        tids = suffix_data["token_ids"]
        pids = suffix_data["parents"]
    elif fallback_data and fallback_data.get("token_ids"):
        tids = fallback_data["token_ids"]
        pids = fallback_data["parents"]
    else:
        return 0

    return greedy_tree_walk(tids, pids, gt)


def _single_proposer_step(rec: dict, budget: int, proposer_name: str) -> int:
    """Single proposer: use full tree, return accepted tokens."""
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0

    tree_data = rec.get("per_proposer", {}).get(proposer_name)
    if not tree_data:
        return 0

    tids = tree_data["token_ids"]
    pids = tree_data["parents"]
    return greedy_tree_walk(tids, pids, gt)


def _subset_step(rec: dict, budget: int, proposer_names: List[str]) -> int:
    """Choose-One over a subset of proposers. Full trees, no truncation."""
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0

    best_acc = 0
    for name in proposer_names:
        tree_data = rec.get("per_proposer", {}).get(name)
        if not tree_data:
            continue
        tids = tree_data["token_ids"]
        pids = tree_data["parents"]
        acc = greedy_tree_walk(tids, pids, gt)
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
) -> dict:
    """Run step-by-step simulation for each budget with measured latencies.

    Returns per-budget simulation results including speedup.

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

    # Generate pairwise combinations
    pairs: List[Tuple[str, str]] = []
    for i in range(len(proposers)):
        for j in range(i + 1, len(proposers)):
            pairs.append((proposers[i], proposers[j]))

    # Precompute EU Oracle DP for all budgets in one pass (9x speedup)
    print(f"Precomputing EU Oracle DP for {len(records)} records...",
          file=sys.stderr)
    precompute_eu_results(records, budgets, p_t_key,
                          proposer_pairs=pairs if pairs else None)
    print("EU precompute done.", file=sys.stderr)

    results = {}
    for B in budgets:
        t_fwd_ms = _target_forward(B)

        # Oracle methods: all proposers draft in parallel + target verify
        oracle_step = _step_cost(proposers, B)
        eu_sim = simulate_decoding(
            records, budget=B, method="eu", p_t_key=p_t_key,
            verify_latency_ms=oracle_step,
            vanilla_latency_ms=vanilla_ms)
        c1_sim = simulate_decoding(
            records, budget=B, method="choose_one",
            verify_latency_ms=oracle_step,
            vanilla_latency_ms=vanilla_ms)

        entry = {
            "target_forward_ms": t_fwd_ms,
            "eagle3_draft_ms": _eagle3_draft(B),
            "draft_lm_tpot_ms": draft_lm_tpot,
            "eu_mat": eu_sim["mat"],
            "c1_mat": c1_sim["mat"],
            "eu_speedup": eu_sim["speedup"],
            "c1_speedup": c1_sim["speedup"],
            "eu_steps": eu_sim["total_steps"],
            "c1_steps": c1_sim["total_steps"],
            "eu_tokens": eu_sim["total_generated"],
            "c1_tokens": c1_sim["total_generated"],
        }

        # Single-proposer baselines
        for pname in proposers:
            step_ms = _step_cost([pname], B)
            sim = simulate_decoding(
                records, budget=B, method=f"single:{pname}",
                verify_latency_ms=step_ms,
                vanilla_latency_ms=vanilla_ms)
            entry[f"{pname}_mat"] = sim["mat"]
            entry[f"{pname}_speedup"] = sim["speedup"]
            entry[f"{pname}_steps"] = sim["total_steps"]

        # Pairwise combinations: Choose-One (subset) and EU Oracle (eu_pair)
        for a, b in pairs:
            pair_key = f"{a}+{b}"
            step_ms = _step_cost([a, b], B)

            # Choose-One subset
            sim = simulate_decoding(
                records, budget=B, method=f"subset:{a},{b}",
                verify_latency_ms=step_ms,
                vanilla_latency_ms=vanilla_ms)
            entry[f"{pair_key}_mat"] = sim["mat"]
            entry[f"{pair_key}_speedup"] = sim["speedup"]
            entry[f"{pair_key}_steps"] = sim["total_steps"]

            # EU Oracle subset (union trie filtered to pair)
            eu_pair_sim = simulate_decoding(
                records, budget=B, method=f"eu_pair:{a},{b}",
                verify_latency_ms=step_ms,
                vanilla_latency_ms=vanilla_ms,
                p_t_key=p_t_key)
            entry[f"eu_{pair_key}_mat"] = eu_pair_sim["mat"]
            entry[f"eu_{pair_key}_speedup"] = eu_pair_sim["speedup"]
            entry[f"eu_{pair_key}_steps"] = eu_pair_sim["total_steps"]

        # Hybrid (suffix score threshold): suffix if score >= t, else fallback
        thresholds = [1.0, 3.0, 5.0]

        # hybrid_e3: suffix + EAGLE3
        if "suffix" in proposers and "eagle3" in proposers:
            # When suffix wins → step = target_forward (no draft)
            # When eagle3 wins → step = target_forward + eagle3_draft
            # Use worst case (eagle3 fallback) for step cost
            hybrid_cost = _step_cost(["eagle3"], B)
            for t in thresholds:
                key = f"hybrid_e3_t{t:.1f}"
                sim = simulate_decoding(
                    records, budget=B, method=f"hybrid_e3:{t}",
                    verify_latency_ms=hybrid_cost,
                    vanilla_latency_ms=vanilla_ms)
                entry[f"{key}_mat"] = sim["mat"]
                entry[f"{key}_speedup"] = sim["speedup"]

        # hybrid_dm: suffix + draft model
        if "suffix" in proposers and "draft_model" in proposers:
            hybrid_cost = _step_cost(["draft_model"], B)
            for t in thresholds:
                key = f"hybrid_dm_t{t:.1f}"
                sim = simulate_decoding(
                    records, budget=B, method=f"hybrid_dm:{t}",
                    verify_latency_ms=hybrid_cost,
                    vanilla_latency_ms=vanilla_ms)
                entry[f"{key}_mat"] = sim["mat"]
                entry[f"{key}_speedup"] = sim["speedup"]

        results[B] = entry

    return results


def print_latency_summary(
    latency_results: dict,
    budgets: List[int],
    vanilla_ms: float,
):
    """Print latency-aware speedup summary."""
    # Discover proposer names from first budget entry
    first = latency_results[budgets[0]]
    proposers = sorted(k.replace("_speedup", "") for k in first
                       if k.endswith("_speedup") and k not in
                       ("eu_speedup", "c1_speedup") and "+" not in k)
    pair_keys = sorted(k.replace("_speedup", "") for k in first
                       if k.endswith("_speedup") and "+" in k
                       and not k.startswith("eu_"))

    # --- Single-proposer baselines ---
    if proposers:
        print("\n" + "=" * 85, file=sys.stderr)
        print("SINGLE-PROPOSER BASELINES", file=sys.stderr)
        print("=" * 85, file=sys.stderr)
        print(f"Vanilla TPOT: {vanilla_ms:.2f} ms/tok", file=sys.stderr)
        e3_draft = first.get("eagle3_draft_ms", 0)
        dm_tpot = first.get("draft_lm_tpot_ms", 0)
        t_fwd = first.get("target_forward_ms", vanilla_ms)
        print(f"Target forward: {t_fwd:.2f} ms | "
              f"EAGLE3 draft: {e3_draft:.2f} ms | "
              f"Draft LM TPOT: {dm_tpot:.2f} ms", file=sys.stderr)
        print("Step cost = target_forward(B) + max(draft costs); "
              "suffix=0 (CPU)", file=sys.stderr)
        print(file=sys.stderr)

        # Header
        hdr = f"{'Budget':>6} | {'T_fwd':>8} | {'E3_dft':>7}"
        for p in proposers:
            hdr += f" | {p:>10}"
        print(hdr, file=sys.stderr)
        print("-" * len(hdr), file=sys.stderr)

        best_per_proposer: Dict[str, tuple] = {p: (0, 0.0) for p in proposers}

        for B in budgets:
            r = latency_results[B]
            row = f"{B:>6} | {r['target_forward_ms']:>7.2f}ms | {r['eagle3_draft_ms']:>6.1f}ms"
            for p in proposers:
                spd = r.get(f"{p}_speedup", 0)
                mat = r.get(f"{p}_mat", 0)
                row += f" | {spd:>6.2f}x/{mat:.2f}"
                if spd > best_per_proposer[p][1]:
                    best_per_proposer[p] = (B, spd)
            print(row, file=sys.stderr)

        print(file=sys.stderr)
        for p in proposers:
            b, s = best_per_proposer[p]
            print(f"  Best {p}: budget={b}, speedup={s:.2f}x", file=sys.stderr)
        print("=" * len(hdr), file=sys.stderr)

    # --- Pairwise combinations ---
    # Discover choose-one pairs and EU pairs separately
    eu_pair_keys = sorted(k.replace("_speedup", "") for k in first
                          if k.endswith("_speedup") and k.startswith("eu_")
                          and "+" in k)

    if pair_keys:
        print("\n" + "=" * 85, file=sys.stderr)
        print("PAIRWISE COMBINATIONS (choose-one over 2 proposers)", file=sys.stderr)
        print("=" * 85, file=sys.stderr)
        print(file=sys.stderr)

        hdr = f"{'Budget':>6} | {'T_fwd':>8}"
        for pk in pair_keys:
            hdr += f" | {pk:>14}"
        print(hdr, file=sys.stderr)
        print("-" * len(hdr), file=sys.stderr)

        best_per_pair: Dict[str, tuple] = {pk: (0, 0.0) for pk in pair_keys}

        for B in budgets:
            r = latency_results[B]
            row = f"{B:>6} | {r['target_forward_ms']:>7.2f}ms"
            for pk in pair_keys:
                spd = r.get(f"{pk}_speedup", 0)
                mat = r.get(f"{pk}_mat", 0)
                row += f" | {spd:>7.2f}x/{mat:.2f}"
                if spd > best_per_pair[pk][1]:
                    best_per_pair[pk] = (B, spd)
            print(row, file=sys.stderr)

        print(file=sys.stderr)
        for pk in pair_keys:
            b, s = best_per_pair[pk]
            print(f"  Best {pk}: budget={b}, speedup={s:.2f}x", file=sys.stderr)
        print("=" * len(hdr), file=sys.stderr)

    # --- EU Oracle pairwise ---
    if eu_pair_keys:
        print("\n" + "=" * 85, file=sys.stderr)
        print("EU ORACLE PAIRWISE (union trie filtered to 2 proposers + knapsack)",
              file=sys.stderr)
        print("=" * 85, file=sys.stderr)
        print(file=sys.stderr)

        hdr = f"{'Budget':>6} | {'T_fwd':>8}"
        for pk in eu_pair_keys:
            hdr += f" | {pk:>18}"
        print(hdr, file=sys.stderr)
        print("-" * len(hdr), file=sys.stderr)

        best_eu_pair: Dict[str, tuple] = {pk: (0, 0.0) for pk in eu_pair_keys}

        for B in budgets:
            r = latency_results[B]
            row = f"{B:>6} | {r['target_forward_ms']:>7.2f}ms"
            for pk in eu_pair_keys:
                spd = r.get(f"{pk}_speedup", 0)
                mat = r.get(f"{pk}_mat", 0)
                row += f" | {spd:>10.2f}x/{mat:.2f}"
                if spd > best_eu_pair[pk][1]:
                    best_eu_pair[pk] = (B, spd)
            print(row, file=sys.stderr)

        print(file=sys.stderr)
        for pk in eu_pair_keys:
            b, s = best_eu_pair[pk]
            print(f"  Best {pk}: budget={b}, speedup={s:.2f}x", file=sys.stderr)
        print("=" * len(hdr), file=sys.stderr)

    # --- Oracle methods ---
    print("\n" + "=" * 85, file=sys.stderr)
    print("ORACLE METHODS (all proposers)", file=sys.stderr)
    print("=" * 85, file=sys.stderr)
    print(f"Vanilla: {vanilla_ms:.2f} ms/tok", file=sys.stderr)
    print(f"Step cost = target_forward(B) + max(all draft costs)", file=sys.stderr)
    print(file=sys.stderr)
    print(f"{'Budget':>6} | {'T_fwd':>8} | {'E3_dft':>7} | {'EU MAT':>7} | {'C1 MAT':>7} | "
          f"{'EU speed':>9} | {'C1 speed':>9} | {'EU steps':>9} | {'C1 steps':>9}",
          file=sys.stderr)
    print("-" * 95, file=sys.stderr)

    best_eu = (0, 0.0)
    best_c1 = (0, 0.0)

    for B in budgets:
        r = latency_results[B]
        print(f"{B:>6} | {r['target_forward_ms']:>7.2f}ms | {r['eagle3_draft_ms']:>6.1f}ms | "
              f"{r['eu_mat']:>7.2f} | {r['c1_mat']:>7.2f} | "
              f"{r['eu_speedup']:>8.2f}x | {r['c1_speedup']:>8.2f}x | "
              f"{r['eu_steps']:>8} | {r['c1_steps']:>8}",
              file=sys.stderr)
        if r['eu_speedup'] > best_eu[1]:
            best_eu = (B, r['eu_speedup'])
        if r['c1_speedup'] > best_c1[1]:
            best_c1 = (B, r['c1_speedup'])

    print(file=sys.stderr)
    print(f"Best EU Oracle:  budget={best_eu[0]}, speedup={best_eu[1]:.2f}x", file=sys.stderr)
    print(f"Best Choose-One: budget={best_c1[0]}, speedup={best_c1[1]:.2f}x", file=sys.stderr)
    if best_c1[1] > 0:
        advantage = (best_eu[1] / best_c1[1] - 1) * 100
        print(f"EU advantage: {advantage:+.1f}%", file=sys.stderr)
    print("=" * 85, file=sys.stderr)


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
        from hybrid_spec_decoding.analysis.collect_target_probs import (
            enrich_with_ground_truth_p_t,
        )
        enrich_with_ground_truth_p_t(records)
        print("Computed oracle p_t from ground truth", file=sys.stderr)

    # Evaluate Choose-One Oracle (unconstrained)
    t0 = time.time()
    choose_one = evaluate_choose_one(records)
    print(f"Choose-One Oracle: {time.time() - t0:.2f}s", file=sys.stderr)

    # Evaluate Expected-Utility Oracle (DP) across budgets
    t0 = time.time()
    eu_results = evaluate_expected_utility(records, budgets, args.p_t_key)
    print(f"Expected-Utility Oracle: {time.time() - t0:.2f}s", file=sys.stderr)

    # Evaluate Choose-One at each budget (for fair comparison)
    t0 = time.time()
    choose_one_budget = evaluate_choose_one_at_budget(records, budgets)
    print(f"Choose-One at budget: {time.time() - t0:.2f}s", file=sys.stderr)

    # Latency-aware simulation (all costs from latency_config)
    with open(args.latency_config) as f:
        latency_config = json.load(f)
    latency_results = compute_latency_speedup(
        records, budgets, latency_config, p_t_key=args.p_t_key)

    if args.print_summary:
        print_summary(choose_one, eu_results, choose_one_budget,
                      budgets, args.p_t_key)
        print_latency_summary(latency_results, budgets,
                              latency_config["vanilla_step_ms"])

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
            "expected_utility": {
                "budget_sweep": [
                    {
                        "budget": B,
                        "avg_eu": eu_results[B]["avg_eu"],
                        "total_eu": eu_results[B]["total_eu"],
                        "avg_selected_nodes": eu_results[B]["avg_selected_nodes"],
                        "choose_one_avg_acc": choose_one_budget[B]["avg_acc"],
                        "gap": eu_results[B]["avg_eu"] - choose_one_budget[B]["avg_acc"],
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
                    "verify_ms": latency_results[B]["verify_ms"],
                    "eu_speedup": latency_results[B]["eu_speedup"],
                    "c1_speedup": latency_results[B]["c1_speedup"],
                    **{
                        f"{m}_speedup": latency_results[B].get(f"{m}_speedup", 0)
                        for m in all_methods
                    },
                    **{
                        f"{m}_mat": latency_results[B].get(f"{m}_mat", 0)
                        for m in all_methods
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
