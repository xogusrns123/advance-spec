"""Tree-budget oracle simulation for heterogeneous speculative decoding.

Simulates per-step accept behavior for several method families on
per-proposer records assembled from Stage 1 (EAGLE3 oracle vanilla) and
Stage 2 (draft-model drafts) artifacts. Suffix is drawn live from a
``SuffixDecodingCache`` inside the simulator — no Stage 3a artifact.

Supported methods (50% gap goal — 2026-04-27):

* ``single:{eagle3,draft_model,suffix}`` — one proposer's tree, greedy walk.
* ``hybrid_e3:{t}`` / ``hybrid_dm:{t}`` — suffix if score ≥ t, else fall
  back to eagle3 / draft_model. Paper-faithful suffix params (F=1.0, T=0.1).
* ``extension`` / ``extension_oracle`` — EAGLE3 backbone + live suffix
  grafts at every node. Oracle variant: cost charges only accepted-in-suffix.
* ``extension_prune_pt:t`` / ``_oracle:t`` — backbone pruned by path_p_t.
* ``extension_joint_score:t`` / ``_oracle:t`` — suffix attached only when
  ``draft.score × path_p_t ≥ t`` (joint eagle3 + suffix confidence).
* ``extension_hybrid:t`` / ``_oracle:t`` — per-step suffix-only vs ext.

Forbidden methods (removed):
  - extension_oracle_path (path-only accounting unrealistic)
  - extension_hybrid_perfect_oracle* (per-step oracle gate unrealistic)
  - extension_dual_method*, extension_dmsfx*, extension_2level*,
    extension_sfx_backbone*, extension_anchor*, extension_hybrid_prune_pt*,
    extension_pure_sfx*

Usage:
    python3 -m simulation.evaluation.run_tree_oracle_sim \\
        --agent-results results/.../agent_results_eagle3.json \\
        --draft-model-drafts results/.../draft_model_drafts.jsonl \\
        --dataset data/specbench/dataset.jsonl \\
        --model Qwen/Qwen3-14B \\
        --budgets 1,2,4,8,16,32,64,128 \\
        --latency-config simulation/config/latency/qwen3_14b.json \\
        --output simulation/results/.../tree_oracle_sim.json \\
        --print-summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Module-level globals used by worker processes (set via Pool initializer)
# so workers don't need to pickle/transfer the full records list per call —
# they reference the parent's COW-shared memory after fork.
_WORKER_RECORDS = None


def _worker_init(records):
    """ProcessPoolExecutor initializer: stash records in worker globals."""
    global _WORKER_RECORDS
    _WORKER_RECORDS = records


def _worker_simulate(args):
    """Run one simulate_decoding() in a worker. Pulls records from global."""
    method_key, sim_kwargs, prefix = args
    sim_kwargs = dict(sim_kwargs)  # defensive copy
    sim_kwargs["records"] = _WORKER_RECORDS
    sim = simulate_decoding(**sim_kwargs)
    return method_key, sim, prefix


def _picklable_interp(B, table, default):
    """Module-level interp helper — picklable so it can be sent to workers
    via functools.partial. Replaces inner closure _target_forward."""
    if not table or B <= 0:
        return default
    keys_int = sorted(int(k) for k in table.keys())
    s_table = {int(k): float(v) for k, v in table.items()}
    if B in s_table:
        return s_table[B]
    if B <= keys_int[0]:
        return s_table[keys_int[0]]
    if B >= keys_int[-1]:
        return s_table[keys_int[-1]]
    for i in range(len(keys_int) - 1):
        lo, hi = keys_int[i], keys_int[i + 1]
        if lo <= B <= hi:
            frac = (B - lo) / (hi - lo)
            return s_table[lo] + frac * (s_table[hi] - s_table[lo])
    return default

import numpy as np

from simulation.evaluation.tree_knapsack import greedy_tree_walk


def print_summary(budgets: List[int]):
    """Print a header banner to stderr before the per-budget simulation loop."""
    print("\n" + "=" * 70, file=sys.stderr)
    print("TREE-BUDGET ORACLE SIMULATION RESULTS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Budgets: {budgets}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step-by-step simulation (correct skip-ahead behavior)
# ---------------------------------------------------------------------------

def simulate_decoding(
    records: List[dict],
    budget: int,
    method: str,
    *,
    vanilla_latency_ms: float,
    verify_latency_ms: float = 0.0,
    suffix_cache=None,
    draft_ratios: Optional[List[float]] = None,
    real_step_cost_ms: Optional[float] = None,
    real_step_cost_suffix_ms: Optional[float] = None,
    real_step_target_fn=None,
    real_step_draft_only_ms: Optional[float] = None,
    real_step_draft_fn=None,
    suffix_speculate_ms_param: float = 0.0,
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

    # Determine if this method has draft cost. Methods that pick suffix vs
    # eagle3 per-step (= hybrid family) need conditional draft accounting.
    # extension_hybrid* fall in the same bucket: they pick suffix-only or
    # extension fallback per step based on a score threshold.
    is_hybrid = (method.startswith("hybrid_e3:")
                 or method.startswith("hybrid_oracle:")
                 or method.startswith("hybrid_dm:")
                 or method.startswith("hybrid_dm_oracle:"))
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
    # Global tree is shared across requests within this method (so patterns
    # observed in earlier requests help later ones). Per-request LOCAL tree
    # is reset via start_request below. This prevents cross-method state
    # leakage (oracle vs realistic getting different speculate() results)
    # while preserving in-method global-tree accumulation that's essential
    # for suffix cache's purpose.
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
        # gt_len is the actual remaining-trajectory length; ground_truth_future
        # may be truncated to save memory (see assemble_records.py).
        if last_rec is not None:
            last_gt_len = last_rec.get(
                "gt_len", len(last_rec.get("ground_truth_future", [])))
        else:
            last_gt_len = 1
        first_gt_len = first_rec.get(
            "gt_len", len(first_rec.get("ground_truth_future", [])))
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
            _step_draft_ms = None  # if set, overrides real_step_draft_only_ms
            # extension_oracle:F:T — v2: per-step picker over BUDGET_GRID
            # picks B with max accept (ties: smaller B). cost = target(a+1)
            # + eagle3_draft(picked_B). Outer ``budget`` arg is ignored.
            if method == "extension_oracle" or method.startswith("extension_oracle:"):
                F, T = 4.0, 0.0  # defaults
                if ":" in method:
                    parts = method.split(":")
                    if len(parts) >= 3:
                        F = float(parts[1]); T = float(parts[2])
                _BUDGET_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
                _per_b_acc: Dict[int, int] = {}
                for _b in _BUDGET_GRID:
                    _a, _ = _extension_step(
                        rec, _b, local_cache, cache_req_id,
                        base_proposer="eagle3",
                        suffix_max_spec_factor=F,
                        suffix_min_token_prob=T,
                        suffix_max_spec_tokens=0)
                    _per_b_acc[_b] = _a
                _best_b = max(_BUDGET_GRID,
                              key=lambda b: (_per_b_acc[b], -b))
                accepted = _per_b_acc[_best_b]
                ext_size = accepted + 1  # accept-only verify
                if real_step_draft_fn is not None:
                    _step_draft_ms = real_step_draft_fn(_best_b)
            elif method == "extension" or (method.startswith("extension:")
                                           and not method.startswith("extension:by")):
                F, T = 4.0, 0.0
                if ":" in method:
                    parts = method.split(":")
                    if len(parts) >= 3:
                        F = float(parts[1]); T = float(parts[2])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    suffix_max_spec_factor=F,
                    suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_by_count:"):
                # Total tree (base + suffix grafts) capped at C = B × ratio.
                # Base stays at budget B; suffix fills up to cap.
                ratio = float(method.split(":", 1)[1])
                cap = max(1, int(round(budget * ratio)))
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3", max_count=cap)
            elif method.startswith("extension_by_score:"):
                # Only attach suffix at base nodes where suffix draft.score ≥ t.
                threshold = float(method.split(":", 1)[1])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3", score_threshold=threshold)
            elif method.startswith("extension_by_count_score:"):
                # Combined: count cap + score filter.
                _, rest = method.split(":", 1)
                r_s, t_s = rest.split(":", 1)
                cap = max(1, int(round(budget * float(r_s))))
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    max_count=cap, score_threshold=float(t_s))
            elif method.startswith("extension_prune_pt:"):
                # Prune backbone itself where path_p_t < t.
                t = float(method.split(":", 1)[1])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3", backbone_pt_threshold=t)
            # ----- draft_model-backbone extension family (parallel to eagle3) -----
            elif method == "extension_dm" or (method.startswith("extension_dm:")
                                              and not method.startswith("extension_dm_")):
                F, T = 4.0, 0.0
                if ":" in method:
                    parts = method.split(":")
                    if len(parts) >= 3:
                        F = float(parts[1]); T = float(parts[2])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model",
                    suffix_max_spec_factor=F,
                    suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method == "extension_dm_oracle" or method.startswith("extension_dm_oracle:"):
                F, T = 4.0, 0.0
                if ":" in method:
                    parts = method.split(":")
                    if len(parts) >= 3:
                        F = float(parts[1]); T = float(parts[2])
                _BUDGET_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
                _per_b_acc: Dict[int, int] = {}
                for _b in _BUDGET_GRID:
                    _a, _ = _extension_step(
                        rec, _b, local_cache, cache_req_id,
                        base_proposer="draft_model",
                        suffix_max_spec_factor=F,
                        suffix_min_token_prob=T,
                        suffix_max_spec_tokens=0)
                    _per_b_acc[_b] = _a
                _best_b = max(_BUDGET_GRID,
                              key=lambda b: (_per_b_acc[b], -b))
                accepted = _per_b_acc[_best_b]
                ext_size = accepted + 1
                if real_step_draft_fn is not None:
                    _step_draft_ms = real_step_draft_fn(_best_b)
            elif method.startswith("extension_dm_by_count:"):
                ratio = float(method.split(":", 1)[1])
                cap = max(1, int(round(budget * ratio)))
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model", max_count=cap)
            elif method.startswith("extension_dm_by_score:"):
                threshold = float(method.split(":", 1)[1])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model", score_threshold=threshold)
            elif is_hybrid:
                if method.startswith("hybrid_oracle:"):
                    # hybrid_oracle:F:T:τ — hybrid_e3 gating + accept-only
                    # verify cost. If suffix score >= τ → suffix branch
                    # (cost = target(a+1) + suffix_speculate). Else → eagle3
                    # truncated to outer budget B (cost = target(a+1) +
                    # eagle3_draft(B)).
                    parts = method.split(":")
                    F = float(parts[1]); T = float(parts[2])
                    tau = float(parts[3])
                    gt = rec.get("ground_truth_future") or []
                    if gt:
                        ctx = rec.get("context_token_ids") or []
                        sfx_tids, sfx_pids, sfx_score = _live_suffix_draft(
                            local_cache, cache_req_id, ctx,
                            max_spec_factor=F, min_token_prob=T)
                        if sfx_tids is not None and sfx_score >= tau:
                            accepted = greedy_tree_walk(
                                sfx_tids, sfx_pids, gt)
                            used_suffix = True
                            _step_draft_ms = suffix_speculate_ms_param
                        else:
                            accepted = _proposer_tree_walk(
                                rec.get("per_proposer", {}) or {},
                                "eagle3", gt, budget)
                            used_suffix = False
                            if real_step_draft_fn is not None:
                                _step_draft_ms = real_step_draft_fn(budget)
                    else:
                        accepted = 0
                    ext_size = accepted + 1  # accept-only verify
                elif method.startswith("hybrid_dm_oracle:"):
                    # hybrid_dm_oracle:F:T:τ — like hybrid_oracle but
                    # falls back to draft_model chain (capped at MAX_DRAFT_MODEL_N).
                    parts = method.split(":")
                    F = float(parts[1]); T = float(parts[2])
                    tau = float(parts[3])
                    gt = rec.get("ground_truth_future") or []
                    if gt:
                        ctx = rec.get("context_token_ids") or []
                        sfx_tids, sfx_pids, sfx_score = _live_suffix_draft(
                            local_cache, cache_req_id, ctx,
                            max_spec_factor=F, min_token_prob=T)
                        if sfx_tids is not None and sfx_score >= tau:
                            accepted = greedy_tree_walk(
                                sfx_tids, sfx_pids, gt)
                            used_suffix = True
                            _step_draft_ms = suffix_speculate_ms_param
                        else:
                            accepted = _proposer_tree_walk(
                                rec.get("per_proposer", {}) or {},
                                "draft_model", gt, budget)
                            used_suffix = False
                            # draft_model fallback: cost = draft_lm_tpot × min(B, MAX)
                            if real_step_draft_fn is not None:
                                _step_draft_ms = real_step_draft_fn(budget)
                    else:
                        accepted = 0
                    ext_size = accepted + 1  # accept-only verify
                elif method.startswith("hybrid_e3:"):
                    # hybrid_e3:F:T:t — parametric (F, T, threshold)
                    # Or legacy hybrid_e3:t — defaults F=1.0, T=0.1 (paper-faithful)
                    parts = method.split(":")
                    if len(parts) == 4:
                        F = float(parts[1]); T = float(parts[2]); threshold = float(parts[3])
                        accepted, used_suffix = _hybrid_step(
                            rec, budget, threshold, fallback="eagle3",
                            suffix_cache=local_cache,
                            cache_req_id=cache_req_id,
                            max_spec_factor=F, min_token_prob=T,
                            max_spec_tokens=None)  # unbounded
                    else:
                        threshold = float(parts[1])
                        accepted, used_suffix = _hybrid_step(
                            rec, budget, threshold, fallback="eagle3",
                            suffix_cache=local_cache,
                            cache_req_id=cache_req_id)
                elif method.startswith("hybrid_dm:"):
                    # hybrid_dm:F:T:t — gate suffix vs draft_model fallback
                    parts = method.split(":")
                    if len(parts) == 4:
                        F = float(parts[1]); T = float(parts[2]); threshold = float(parts[3])
                        accepted, used_suffix = _hybrid_step(
                            rec, budget, threshold, fallback="draft_model",
                            suffix_cache=local_cache,
                            cache_req_id=cache_req_id,
                            max_spec_factor=F, min_token_prob=T,
                            max_spec_tokens=None)
                    else:
                        threshold = float(parts[1])
                        accepted, used_suffix = _hybrid_step(
                            rec, budget, threshold, fallback="draft_model",
                            suffix_cache=local_cache,
                            cache_req_id=cache_req_id)
                else:
                    raise ValueError(f"unknown hybrid method: {method}")
                # Suffix branch: target verifies the full suffix tree (no
                # budget truncation). Re-draw with the SAME (F, T) used by
                # the gating call so the verify size matches.
                if used_suffix:
                    _base_ctx = rec.get("context_token_ids") or []
                    _F = _T = None
                    if method.startswith("hybrid_e3:"):
                        _parts = method.split(":")
                        if len(_parts) == 4:
                            _F = float(_parts[1]); _T = float(_parts[2])
                    _tids, _, _ = _live_suffix_draft(
                        local_cache, cache_req_id, _base_ctx,
                        max_spec_factor=_F, min_token_prob=_T)
                    ext_size = len(_tids) if _tids else 1
            elif method.startswith("single:"):
                # `single:suffix:F:T` (parametric) or `single:suffix` / `single:eagle3` etc.
                method_rest = method.split(":", 1)[1]
                proposer_name = method_rest.split(":", 1)[0]  # "suffix", "eagle3", ...
                _F = _T = None
                if proposer_name == "suffix" and ":" in method_rest:
                    _parts = method_rest.split(":")
                    if len(_parts) >= 3:
                        _F = float(_parts[1]); _T = float(_parts[2])
                accepted = _single_proposer_step(
                    rec, budget, proposer_name,
                    suffix_cache=local_cache, cache_req_id=cache_req_id,
                    suffix_max_spec_factor=_F,
                    suffix_min_token_prob=_T)
                if proposer_name == "suffix":
                    _base_ctx = rec.get("context_token_ids") or []
                    _tids, _, _ = _live_suffix_draft(
                        local_cache, cache_req_id, _base_ctx,
                        max_spec_factor=_F, min_token_prob=_T)
                    ext_size = len(_tids) if _tids else 1
                else:
                    tree_data = rec.get("per_proposer", {}).get(proposer_name, {})
                    tok_ids = tree_data.get("token_ids") or []
                    if not tok_ids:
                        ext_size = 1
                    else:
                        ext_size = min(budget, len(tok_ids))
            else:
                raise ValueError(f"unknown method: {method}")

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
                # hybrid's suffix branch). ext_size was set earlier in the
                # dispatch to the step's actual verified tree size.
                # real_step_draft_only_ms is the draft-only cost that doesn't
                # depend on ext_size.
                if (ext_size is not None and real_step_target_fn is not None
                        and (real_step_draft_only_ms is not None
                             or _step_draft_ms is not None)):
                    target_ms_step = real_step_target_fn(ext_size)
                    # Per-step draft-cost override (set by extension_oracle /
                    # hybrid_oracle dispatch) takes precedence.
                    _draft_ms = (_step_draft_ms if _step_draft_ms is not None
                                 else real_step_draft_only_ms)
                    step_real = target_ms_step + _draft_ms
                    total_time_real_ms += step_real
                    total_target_ms += target_ms_step
                    total_draft_ms += _draft_ms
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
                    # single:eagle3, single:draft_model — flat
                    # real_step_cost_ms (verified size == B). Approximate
                    # split from B-dependent target_forward + draft_only
                    # (not tracked directly).
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


def _live_suffix_draft(suffix_cache, cache_req_id: str, context,
                       paper_faithful: bool = False,
                       max_spec_factor: Optional[float] = None,
                       min_token_prob: Optional[float] = None,
                       max_spec_tokens: Optional[int] = None):
    """Live speculate — returns (token_ids, parents, score) or (None,None,0).

    Three regimes (priority: explicit args > paper_faithful > default):
      * Default: aggressive (F=4.0, T=0.0, N=256) — extension/ceiling
      * paper_faithful=True: F=1.0, T=0.1, N=unbounded — paper hybrid baseline
      * Explicit args: any of F/T/N override defaults
    """
    if suffix_cache is None or context is None:
        return None, None, 0.0
    try:
        ctx_np = np.asarray(context, dtype=np.int32)
        custom = (max_spec_factor is not None or min_token_prob is not None
                  or max_spec_tokens is not None)
        if custom:
            kwargs = {"use_tree_spec": True}
            kwargs["max_spec_factor"] = (max_spec_factor if max_spec_factor is not None
                                         else (1.0 if paper_faithful else 4.0))
            kwargs["min_token_prob"] = (min_token_prob if min_token_prob is not None
                                        else (0.1 if paper_faithful else 0.0))
            # max_spec_tokens: explicit positive value → cap at it.
            # max_spec_tokens=0 (or None) → UNBOUNDED (no cap).
            if max_spec_tokens is not None and max_spec_tokens > 0:
                kwargs["max_spec_tokens"] = max_spec_tokens
            # else: leave unset → ArcticInference default = unbounded
            draft = suffix_cache.speculate(cache_req_id, ctx_np, **kwargs)
        elif paper_faithful:
            draft = suffix_cache.speculate(
                cache_req_id, ctx_np,
                max_spec_factor=1.0, min_token_prob=0.1, use_tree_spec=True)
        else:
            # default (legacy): aggressive with N=256
            draft = suffix_cache.speculate(
                cache_req_id, ctx_np,
                max_spec_tokens=256, max_spec_factor=4.0,
                min_token_prob=0.0, use_tree_spec=True)
    except Exception:
        return None, None, 0.0
    if not draft.token_ids:
        return None, None, 0.0
    return list(draft.token_ids), list(draft.parents), float(
        getattr(draft, "score", 0.0))


def _hybrid_step(rec: dict, budget: int, threshold: float,
                 fallback: str = "eagle3",
                 suffix_cache=None, cache_req_id: str = "",
                 max_spec_factor: Optional[float] = None,
                 min_token_prob: Optional[float] = None,
                 max_spec_tokens: Optional[int] = None) -> tuple:
    """Hybrid: use live suffix if score >= threshold, else fallback proposer.

    Custom (max_spec_factor, min_token_prob, max_spec_tokens) override
    default aggressive suffix params — used by hybrid_e3_sfx:F:T:N:t.

    Returns (accepted_tokens, used_suffix: bool).
    """
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0, False

    per_proposer = rec.get("per_proposer", {})
    base_context = rec.get("context_token_ids") or []

    sfx_tids, sfx_pids, sfx_score = _live_suffix_draft(
        suffix_cache, cache_req_id, base_context,
        max_spec_factor=max_spec_factor,
        min_token_prob=min_token_prob,
        max_spec_tokens=max_spec_tokens)
    use_suffix = (sfx_tids is not None and sfx_score >= threshold)

    if use_suffix:
        # Suffix has no draft cost → full tree used (no budget truncation)
        return greedy_tree_walk(sfx_tids, sfx_pids, gt), True
    fallback_data = per_proposer.get(fallback)
    if fallback_data and fallback_data.get("token_ids"):
        return _proposer_tree_walk(per_proposer, fallback, gt, budget), False
    return 0, False


def _extension_step(rec: dict, budget: int, suffix_cache, cache_req_id: str,
                    base_proposer: str = "eagle3",
                    score_threshold: Optional[float] = None,
                    max_count: Optional[int] = None,
                    pathprob_threshold: Optional[float] = None,
                    pt_threshold: Optional[float] = None,
                    backbone_pt_threshold: Optional[float] = None,
                    suffix_max_spec_factor: float = 4.0,
                    suffix_min_token_prob: float = 0.0,
                    suffix_max_spec_tokens: int = 0):
    # NOTE: suffix_max_spec_tokens=0 → "unbounded" (don't pass max_spec_tokens
    # to ArcticInference). This matches the explicit value used by the bare
    # ``extension`` method dispatch, so filter variants (extension_by_count,
    # _by_score, _prune_pt) build per-graft suffix trees of the SAME size as
    # base extension. Previously the default was 256, which made filtered
    # trees not strict subsets of the base tree and produced MAT values that
    # exceeded the unfiltered base — physically impossible if the only effect
    # of filtering is to drop nodes.
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
      pt_threshold — skip suffix anchoring at base nodes whose path_p_t
                         is below t. Base node itself is kept.
      backbone_pt_threshold — prune base tree itself: remove base nodes
                         with path_p_t < t (and their subtrees). Suffix
                         extension then only attaches at surviving base
                         nodes. Since path_p_t is monotone non-increasing
                         along paths, filter keeps a valid subtree.
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

    # Backbone prune: drop base nodes whose cumulative draft probability
    # falls below the threshold. path_p_t is monotone non-increasing along
    # each root→leaf path (each step multiplies by node_p_t ≤ 1), so the
    # keep-mask defines a valid subtree (no orphan fixup needed). If
    # path_p_t is unavailable, the filter silently no-ops.
    if backbone_pt_threshold is not None and path_p_t is not None:
        keep = [path_p_t[i] >= backbone_pt_threshold for i in range(n)]
        if not all(keep):
            old_to_new: Dict[int, int] = {}
            new_tids: List[int] = []
            new_pids: List[int] = []
            new_paths: List[List[int]] = []
            new_path_p_t: List[float] = []
            new_node_p_t: List[float] = [] if node_p_t is not None else None
            for i in range(n):
                if not keep[i]:
                    continue
                po = pids[i]
                pn = old_to_new.get(po, -1) if po >= 0 else -1
                old_to_new[i] = len(new_tids)
                new_tids.append(tids[i])
                new_pids.append(pn)
                new_paths.append(paths[i])
                new_path_p_t.append(path_p_t[i])
                if new_node_p_t is not None:
                    new_node_p_t.append(node_p_t[i])
            tids = new_tids
            pids = new_pids
            ext_tids = list(new_tids)
            ext_pids = list(new_pids)
            paths = new_paths
            path_p_t = new_path_p_t
            node_p_t = new_node_p_t
            n = len(ext_tids)

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
        _spec_kwargs = dict(max_spec_factor=suffix_max_spec_factor,
                            min_token_prob=suffix_min_token_prob,
                            use_tree_spec=True)
        if suffix_max_spec_tokens > 0:
            _spec_kwargs["max_spec_tokens"] = suffix_max_spec_tokens
        # else: 0 → unbounded (don't pass)
        _root_draft = suffix_cache.speculate(
            cache_req_id,
            np.array(base_context, dtype=np.int32),
            **_spec_kwargs)
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

        try:
            _per_kwargs = dict(max_spec_factor=suffix_max_spec_factor,
                               min_token_prob=suffix_min_token_prob,
                               use_tree_spec=True)
            if suffix_max_spec_tokens > 0:
                _per_kwargs["max_spec_tokens"] = suffix_max_spec_tokens
            draft = suffix_cache.speculate(
                cache_req_id, ext_context,
                **_per_kwargs,
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
    _last_acc_base = -1   # last accepted base node (transition point)
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
            _last_acc_base = _picked
        _node = _picked

    # Realistic oracle: target verifies ENTIRE traversed graft (not just
    # the accepted prefix within it). The traversed graft = the suffix-
    # region subtree rooted at the last accepted base node. We BFS down
    # from _last_acc_base via children, counting all nodes with idx >= n
    # (= all suffix descendants of the transition point).
    _traversed_graft = 0
    if _last_acc_base >= 0:
        _stack = []
        for _c in _children.get(_last_acc_base, []):
            if _c >= n:
                _stack.append(_c)
        while _stack:
            _v = _stack.pop()
            _traversed_graft += 1
            _stack.extend(_children.get(_v, []))

    # Stash the breakdown on the function so the oracle dispatch can use
    # it (side-channel — avoids widening the return signature which many
    # existing callers unpack as a 2-tuple).
    _extension_step._last_base_size = n
    _extension_step._last_ext_size_full = len(ext_tids)
    _extension_step._last_accepted_base = _acc_base
    _extension_step._last_accepted_suffix = _acc - _acc_base
    _extension_step._last_traversed_graft_size = _traversed_graft
    return _acc, len(ext_tids)


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
                          suffix_cache=None,
                          cache_req_id: str = "",
                          suffix_max_spec_factor: Optional[float] = None,
                          suffix_min_token_prob: Optional[float] = None) -> int:
    """Single proposer: use per_proposer tree directly, truncate to budget.

    For ``suffix``, optional (F, T) suffix params control the speculate call.
    Default: aggressive (F=4.0, T=0.0, N=unbounded).
    """
    gt = rec.get("ground_truth_future", [])
    if not gt:
        return 0
    if proposer_name == "suffix":
        base_context = rec.get("context_token_ids") or []
        tids, pids, _ = _live_suffix_draft(
            suffix_cache, cache_req_id, base_context,
            max_spec_factor=suffix_max_spec_factor,
            min_token_prob=suffix_min_token_prob)
        if tids is None:
            return 0
        return greedy_tree_walk(tids, pids, gt)
    return _proposer_tree_walk(rec.get("per_proposer", {}), proposer_name, gt, budget)


def _discover_proposers(records: List[dict]) -> List[str]:
    """Find all proposer names available for this run.

    Suffix is always included (drawn live from SuffixDecodingCache inside
    simulate_decoding — no per_proposer data needed). Other proposers
    (eagle3, draft_model, mtp) show up only if their per-step tree is
    present in ``rec["per_proposer"]``.
    """
    names: set = {"suffix"}
    for rec in records:
        names.update(rec.get("per_proposer", {}).keys())
    return sorted(names)


def compute_latency_speedup(
    records: List[dict],
    budgets: List[int],
    latency_config: dict,
    topk: Optional[int] = None,
    steps: Optional[int] = None,
    method_filter: Optional[set] = None,
) -> dict:
    """Run step-by-step simulation for each budget with measured latencies.

    Returns per-budget simulation results including speedup.

    Latency config should contain decomposed costs:
        vanilla_step_ms: target TPOT with no speculation
        target_forward_ms: {B: ms} — pure target verify cost for B tokens
        eagle3_draft_ms: {B: ms} — EAGLE3 draft generation cost
        draft_lm_tpot_ms: draft model per-token cost
        suffix_speculate_ms: per-call cost of SuffixDecodingCache.speculate()

    Missing budgets in the per-B tables are linearly interpolated using the
    nearest measured bracket (and clamped at the extremes).
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
                # Clamp the extrapolation slope to be non-negative. The
                # measurement at the last two keys can be noisy enough to
                # produce a negative slope (e.g., qwen3_14b topk=4 has
                # B=32→64 dipping from 48.19→44.40 ms). Extrapolating that
                # downward beyond the table gives nonsensical negative
                # latency at large B (e.g., extension trees ≥ ~370 nodes),
                # which produced spurious 17–19× speedups in 2026-04-29
                # bfcl_v4 sweeps. Target latency MUST grow (or stay flat)
                # with verify-tree size; clamp here enforces that.
                slope = max(0.0, slope)
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
            # here previously over-charged dmsfx variants at high B by ~15×.
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
        but accounting for it here makes extension comparisons fair.

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

    def _method_allowed(method_key: str) -> bool:
        if method_filter is None:
            return True
        # Matching rules:
        #   exact:                  "extension" matches only "extension"
        #   trailing colon prefix:  "hybrid_e3:" matches all hybrid_e3:* variants
        #   trailing asterisk:      "extension*" matches all extension_* variants
        for pat in method_filter:
            if method_key == pat:
                return True
            if pat.endswith(":") and method_key.startswith(pat):
                return True
            if pat.endswith("*") and method_key.startswith(pat[:-1]):
                return True
        return False

    # Multiprocessing config: SIM_PARALLEL env var controls worker count.
    # Each worker forks the parent so 'records' is COW-shared (no pickling).
    N_WORKERS = int(os.environ.get("SIM_PARALLEL", "1"))
    _executor = None
    if N_WORKERS > 1:
        _executor = ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_worker_init,
            initargs=(records,),
        )
        print(f"Parallel mode: {N_WORKERS} workers (fork-based, COW records)",
              file=sys.stderr)

    # Pending calls accumulated for the current budget; flushed (parallel
    # or sequential) when budget loop iteration ends.
    _pending = []

    def _run(method_key, sim_fn_kwargs, prefix):
        """Queue one (method, budget) sim. Executes in parallel after budget loop."""
        if not _method_allowed(method_key):
            return
        if _executor is None:
            # Sequential path (preserve behavior)
            t0 = time.time()
            sim = simulate_decoding(**sim_fn_kwargs)
            dt = time.time() - t0
            _store_sim(entry, prefix, sim)
            print(f"    {method_key}: {dt:5.1f}s  mat={sim['mat']:.2f} spd={sim['speedup']:.2f}x",
                  file=sys.stderr)
            sys.stderr.flush()
        else:
            # Parallel: drop records (workers have it via fork) and replace
            # any closure callables with picklable equivalents (partial).
            kw = {k: v for k, v in sim_fn_kwargs.items() if k != "records"}
            # Replace _target_forward closure (if present) with a picklable partial
            if "real_step_target_fn" in kw:
                kw["real_step_target_fn"] = partial(
                    _picklable_interp,
                    table=target_fwd,
                    default=vanilla_ms,
                )
            # Replace _eagle3_draft closure with a picklable partial
            if "real_step_draft_fn" in kw:
                kw["real_step_draft_fn"] = partial(
                    _picklable_interp,
                    table=eagle3_draft,
                    default=0.0,
                )
            _pending.append((method_key, kw, prefix))

    def _flush_pending():
        """Execute all pending sims in parallel via the executor."""
        if not _pending or _executor is None:
            return
        t_b0 = time.time()
        n = len(_pending)
        # Submit all
        future_map = {}  # future -> (method_key, prefix)
        for method_key, kw, prefix in _pending:
            fut = _executor.submit(_worker_simulate, (method_key, kw, prefix))
            future_map[fut] = (method_key, prefix)
        # Collect in order of completion
        from concurrent.futures import as_completed
        n_done = 0
        for fut in as_completed(future_map):
            method_key, prefix = future_map[fut]
            try:
                _, sim, _ = fut.result()
            except Exception as e:
                print(f"    {method_key}: WORKER ERROR: {e}", file=sys.stderr)
                continue
            _store_sim(entry, prefix, sim)
            n_done += 1
            print(f"    [{n_done}/{n}] {method_key}: mat={sim['mat']:.2f} spd={sim['speedup']:.2f}x",
                  file=sys.stderr)
            sys.stderr.flush()
        _pending.clear()
        print(f"    (budget batch took {time.time() - t_b0:.1f}s)", file=sys.stderr)
        sys.stderr.flush()

    # Sentinel for simulate_decoding's suffix_cache param: non-None value
    # triggers fresh per-(req,call) SuffixDecodingCache creation inside.
    _SUFFIX_ENABLED = object()

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

        common = dict(records=records, budget=B,
                      vanilla_latency_ms=vanilla_ms,
                      draft_ratios=DRAFT_RATIOS)

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
            if pname == "suffix":
                # single:suffix uses live speculate via the simulator's
                # per-method fresh cache.
                kwargs["suffix_cache"] = _SUFFIX_ENABLED
            _run(f"single:{pname}", kwargs, pname)

            # Parametric F/T sweep for single:suffix
            if pname == "suffix":
                FT_GRID_SUFFIX = [
                    (1.0, 0.0), (1.0, 0.1),
                    (2.0, 0.0), (2.0, 0.1),
                    (4.0, 0.0), (4.0, 0.1),
                ]
                for F, T in FT_GRID_SUFFIX:
                    nm = f"single:suffix:{F}:{T}"
                    tag = f"suffix_f{F}_t{T}"
                    sub_kwargs = {**common, "method": nm,
                                  "real_step_cost_ms": _real_cost([pname], B),
                                  "real_step_target_fn": _target_forward,
                                  "real_step_draft_only_ms": draft_only,
                                  "suffix_cache": _SUFFIX_ENABLED}
                    _run(nm, sub_kwargs, tag)

        # Hybrid (suffix score threshold): suffix if score >= t, else fallback.
        # Hybrid threshold grid (smaller — only the most informative)
        hybrid_thresholds = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

        if "suffix" in proposers and "eagle3" in proposers:
            e3_cost = _real_cost(["eagle3"], B)
            suffix_only_cost = _target_forward(B) + suffix_speculate_ms
            # PARAMETRIC hybrid: F/T sweep × threshold sweep, N unbounded
            # (6 F/T pairs × 6 thresholds = 36 variants)
            FT_GRID = [
                (1.0, 0.0), (1.0, 0.1),
                (2.0, 0.0), (2.0, 0.1),
                (4.0, 0.0), (4.0, 0.1),
            ]
            for F, T in FT_GRID:
                for t in hybrid_thresholds:
                    nm = f"hybrid_e3:{F}:{T}:{t}"
                    tag = f"hybrid_e3_f{F}_t{T}_th{t:.1f}"
                    _run(nm,
                         {**common, "method": nm,
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": e3_cost,
                          "real_step_cost_suffix_ms": suffix_only_cost,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": suffix_speculate_ms},
                         tag)
                    # hybrid_oracle:F:T:τ — accept-only verify cost
                    nm_o = f"hybrid_oracle:{F}:{T}:{t}"
                    tag_o = f"hybrid_oracle_f{F}_t{T}_th{t:.1f}"
                    _run(nm_o,
                         {**common, "method": nm_o,
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": e3_cost,
                          "real_step_cost_suffix_ms": suffix_only_cost,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_fn": _eagle3_draft,
                          "suffix_speculate_ms_param": suffix_speculate_ms},
                         tag_o)

        # ----- Hybrid family with draft_model fallback (parallel to eagle3) -----
        if "suffix" in proposers and "draft_model" in proposers:
            dm_cost = _real_cost(["draft_model"], B)
            suffix_only_cost = _target_forward(B) + suffix_speculate_ms
            FT_GRID = [
                (1.0, 0.0), (1.0, 0.1),
                (2.0, 0.0), (2.0, 0.1),
                (4.0, 0.0), (4.0, 0.1),
            ]
            def _dm_draft_cost(b):
                # draft_model fallback cost = TPOT × min(B, MAX)
                return min(b, MAX_DRAFT_MODEL_N) * draft_lm_tpot
            for F, T in FT_GRID:
                for t in hybrid_thresholds:
                    nm = f"hybrid_dm:{F}:{T}:{t}"
                    tag = f"hybrid_dm_f{F}_t{T}_th{t:.1f}"
                    _run(nm,
                         {**common, "method": nm,
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": dm_cost,
                          "real_step_cost_suffix_ms": suffix_only_cost,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": suffix_speculate_ms},
                         tag)
                    nm_o = f"hybrid_dm_oracle:{F}:{T}:{t}"
                    tag_o = f"hybrid_dm_oracle_f{F}_t{T}_th{t:.1f}"
                    _run(nm_o,
                         {**common, "method": nm_o,
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": dm_cost,
                          "real_step_cost_suffix_ms": suffix_only_cost,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_fn": _dm_draft_cost,
                          "suffix_speculate_ms_param": suffix_speculate_ms},
                         tag_o)

        # Extension: base tree + suffix extension at every node.
        # Real cost per step = target_forward(actual_ext_tree_size)
        #                    + max(base_draft, node_count × suffix_speculate_ms)
        # The target cost is the expensive part and it scales with the full
        # extended tree (not the EAGLE3 base budget B), because target
        # verifies every node. We pass _target_forward as a per-step callable
        # so the simulator can interpolate for any ext_size.
        # (`suffix_cache` sentinel defined at function top — signals
        #  simulate_decoding to instantiate a fresh cache internally)
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
            # PARAMETRIC F/T sweep for extension and extension_oracle.
            # N is always unbounded. 5-pair F/T grid.
            FT_GRID = [
                (1.0, 0.0), (1.0, 0.1),
                (2.0, 0.0), (2.0, 0.1),
                (4.0, 0.0), (4.0, 0.1),
            ]
            for F, T in FT_GRID:
                tag = f"f{F}_t{T}"
                _run(f"extension:{F}:{T}",
                     {**common, "method": f"extension:{F}:{T}",
                      "suffix_cache": _SUFFIX_ENABLED,
                      "real_step_cost_ms": ext_cost_fallback,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": ext_draft_only},
                     f"extension_{tag}")
                # extension_oracle (v2): per-step budget picker, accept-only
                # verify. Outer ``budget`` is ignored — enroll only at the
                # max budget so result appears once per (FT, reslice).
                if B == budgets[-1]:
                    _run(f"extension_oracle:{F}:{T}",
                         {**common, "method": f"extension_oracle:{F}:{T}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_fn": _eagle3_draft,
                          "real_step_draft_only_ms": ext_draft_only},
                         f"extension_oracle_{tag}")

            # Filter variants — count cap (total tree ≤ B × ratio) and
            # score threshold (only graft suffix at high-score base nodes).
            # Combined too. All deployable (no oracle accounting).
            for r in [1.0, 2.0, 4.0]:
                _run(f"extension_by_count:{r}",
                     {**common, "method": f"extension_by_count:{r}",
                      "suffix_cache": _SUFFIX_ENABLED,
                      "real_step_cost_ms": ext_cost_fallback,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": ext_draft_only},
                     f"extension_by_count_r{r}")
            for t in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
                _run(f"extension_by_score:{t}",
                     {**common, "method": f"extension_by_score:{t}",
                      "suffix_cache": _SUFFIX_ENABLED,
                      "real_step_cost_ms": ext_cost_fallback,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": ext_draft_only},
                     f"extension_by_score_t{t:.1f}")
            # extension_by_count_score (combined filter) — disabled this run

            # p_t–based extension filters (eagle3 base only — require
            # EAGLE3 draft-side path probabilities, captured at Stage 1 by
            # the oracle_patch organize_draft_results tracer).
            # Legacy artifacts that predate path_draft_p_t capture simply
            # don't carry it → skip these methods.
            has_draft_p_t = any(
                (rec.get("per_proposer", {})
                    .get("eagle3", {}) or {}).get("path_draft_p_t") is not None
                for rec in records)
            if not has_draft_p_t:
                print("NOTE: no path_draft_p_t available — skipping "
                      "ptopk/product/pathprob/topp/dynsfx methods",
                      file=sys.stderr)

            # prune_pt: remove base nodes with path_p_t < t
            # (their suffix grafts also pruned). Cuts verify cost.
            if has_draft_p_t:
                for t in [0.001, 0.01, 0.1]:
                    _run(f"extension_prune_pt:{t}",
                         {**common, "method": f"extension_prune_pt:{t}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": ext_draft_only},
                         f"extension_prune_pt_t{t}")

        # ----- Extension family with draft_model backbone -----
        # Mirrors the eagle3-base extension family but uses the draft_model
        # linear chain (capped at MAX_DRAFT_MODEL_N=16) as the base tree.
        # prune_pt is omitted — draft_model has no path_draft_p_t signal.
        if "suffix" in proposers and "draft_model" in proposers:
            dm_k = min(B, MAX_DRAFT_MODEL_N)  # base chain length
            # draft-only cost: HF/server forwards for dm chain + per-node suffix
            # speculate (overlapped → max).
            dm_draft_only = max(dm_k * draft_lm_tpot,
                                dm_k * suffix_speculate_ms)
            dm_cost_fallback = _target_forward(B) + dm_draft_only
            for F, T in FT_GRID:
                tag = f"f{F}_t{T}"
                _run(f"extension_dm:{F}:{T}",
                     {**common, "method": f"extension_dm:{F}:{T}",
                      "suffix_cache": _SUFFIX_ENABLED,
                      "real_step_cost_ms": dm_cost_fallback,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": dm_draft_only},
                     f"extension_dm_{tag}")
                if B == budgets[-1]:
                    _run(f"extension_dm_oracle:{F}:{T}",
                         {**common, "method": f"extension_dm_oracle:{F}:{T}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": dm_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_fn": _eagle3_draft,  # dm-oracle still uses target-only verify; draft cost ≈ draft_lm_tpot × picked_B
                          "real_step_draft_only_ms": dm_draft_only},
                         f"extension_dm_oracle_{tag}")
            for r in [1.0, 2.0, 4.0]:
                _run(f"extension_dm_by_count:{r}",
                     {**common, "method": f"extension_dm_by_count:{r}",
                      "suffix_cache": _SUFFIX_ENABLED,
                      "real_step_cost_ms": dm_cost_fallback,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": dm_draft_only},
                     f"extension_dm_by_count_r{r}")
            for t in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
                _run(f"extension_dm_by_score:{t}",
                     {**common, "method": f"extension_dm_by_score:{t}",
                      "suffix_cache": _SUFFIX_ENABLED,
                      "real_step_cost_ms": dm_cost_fallback,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": dm_draft_only},
                     f"extension_dm_by_score_t{t:.1f}")

        # Parallel mode: dispatch all queued (method, budget) pairs
        _flush_pending()

        results[B] = entry
        b_dt = time.time() - b_t0
        print(f"[{b_idx+1}/{total_budgets}] Budget={B} done in {b_dt:.1f}s",
              file=sys.stderr)
        sys.stderr.flush()

    if _executor is not None:
        _executor.shutdown(wait=True)
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
    # like "eagle3", "suffix", "hybrid_e3_t5.0", "extension".
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
    # Input: Stage 1 agent_results + optional Stage 2 draft-model drafts.
    # Records are assembled on the fly; suffix is drawn live in-sim.
    parser.add_argument("--agent-results", required=True,
                        help="Stage 1 EAGLE3 agent_results_eagle3.json")
    parser.add_argument("--draft-model-drafts", default=None,
                        help="Stage 2 per-step draft-model JSONL")
    parser.add_argument("--dataset", default=None,
                        help="dataset.jsonl for BFCL/SpecBench prompt "
                             "reconstruction")
    parser.add_argument("--responses", default=None,
                        help="agent_results_responses.json (BFCL only)")
    parser.add_argument("--model", default=None,
                        help="Target model name for tokenizer")
    parser.add_argument("--exclude", default=None,
                        help="Exclude-ids file")
    parser.add_argument("--output", default=None,
                        help="Output JSON for simulation results")
    parser.add_argument("--budgets", default="1,2,4,8,16,32,64",
                        help="Comma-separated budget values for sweep")
    parser.add_argument("--latency-config", default=None,
                        help="Path to latency_config.json. "
                             "When omitted, MAT / accept-rate stats are still "
                             "reported but latency-aware speedup numbers are skipped.")
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
    parser.add_argument("--reslice-steps", type=int, default=None,
                        help="When set together with --reslice-topk, the per-step "
                             "EAGLE3 tree is rebuilt from the captured full pool "
                             "(SGLANG_CAPTURE_FULL_POOL=1) at this depth (s'). "
                             "Requires --capture-steps and --capture-topk to "
                             "describe the original (S, K) used at capture time.")
    parser.add_argument("--reslice-topk", type=int, default=None,
                        help="Per-parent topk (k') for the resliced tree.")
    parser.add_argument("--capture-steps", type=int, default=None,
                        help="Original S used during Stage 1 capture (e.g. 8). "
                             "Required when --reslice-steps is set.")
    parser.add_argument("--capture-topk", type=int, default=None,
                        help="Original K used during Stage 1 capture (e.g. 16). "
                             "Required when --reslice-steps is set.")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated method names/prefixes to run; "
                             "omit to run all. Examples: "
                             "'single:eagle3,single:suffix,hybrid_e3:1.0,"
                             "extension,extension_oracle'. Use a bare prefix "
                             "like 'extension' to match all extension_* variants.")
    args = parser.parse_args()

    if not args.output and not args.print_summary:
        parser.error("At least one of --output or --print-summary required")

    eagle3_reslice = None
    if args.reslice_steps is not None or args.reslice_topk is not None:
        if not (args.reslice_steps and args.reslice_topk
                and args.capture_steps and args.capture_topk):
            parser.error("--reslice-steps, --reslice-topk, --capture-steps, "
                         "--capture-topk must all be set together.")
        eagle3_reslice = (args.capture_steps, args.capture_topk,
                          args.reslice_steps, args.reslice_topk)

    budgets = [int(b) for b in args.budgets.split(",")]

    from simulation.pipeline.assemble_records import (
        assemble_records_from_artifacts,
    )
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
    input_source = args.agent_results

    # Per-position accept rates per proposer. Independent of method/budget —
    # purely a property of the draft tree vs ground-truth future.
    # Aggregated once over all records; consumed via the output JSON's
    # "position_accepts" field. Accept rate at position d:
    #   seq_accept[d-1] / depth_ge[d-1]   (sequential — requires positions
    #                                       1..d to all match in greedy walk)
    #   ind_accept[d-1] / depth_ge[d-1]   (independent — any node at depth d
    #                                       matches gt[d-1] regardless of
    #                                       ancestors)
    # Cap of 64 covers eagle3 (≤8 reslice depth), draft_model (≤16 chain) and
    # the SuffixDecodingCache's max_tree_depth=64 (live-suffix pre-pass below).
    # extension is NOT measured here — it's a per-step synthesis from
    # eagle3+suffix at sim time, not a single tree.
    POSITION_ACCEPT_MAX = 64
    from simulation.evaluation.tree_knapsack import position_accept_rates
    position_accepts: dict[str, dict[str, list[int]]] = {}

    def _accumulate(prop_name: str, tids, pids, gt):
        if not gt:
            return
        seq, ind, denom_depth = position_accept_rates(
            tids or [], pids or [], gt, POSITION_ACCEPT_MAX)
        if denom_depth <= 0:
            return
        stats = position_accepts.setdefault(prop_name, {
            "seq_accept": [0] * POSITION_ACCEPT_MAX,
            "ind_accept": [0] * POSITION_ACCEPT_MAX,
            "depth_ge": [0] * POSITION_ACCEPT_MAX,
        })
        # depth_ge counts ALL positions up to denom_depth, regardless of
        # whether this step's tree was deep enough to draft at position d.
        # This avoids the "deep-tree steps inflate deep-position accept rate"
        # bias that variable-depth proposers (suffix / EAGLE3 with reslice
        # shorter than tree) would otherwise introduce.
        for d in range(denom_depth):
            stats["depth_ge"][d] += 1
            stats["seq_accept"][d] += seq[d]
            stats["ind_accept"][d] += ind[d]

    # Pre-pass A: stored proposers in records["per_proposer"]. Restricted to
    # the canonical basic set {eagle3, draft_model}. mtp is skipped per
    # current spec; suffix always comes from live pre-pass B below to ensure
    # uniform method-independent measurement (ground-truth-fed cache).
    _PA_PROPOSERS = {"eagle3", "draft_model"}
    for rec in records:
        gt = rec.get("ground_truth_future") or []
        if not gt:
            continue
        for prop_name, prop in (rec.get("per_proposer") or {}).items():
            if prop_name not in _PA_PROPOSERS:
                continue
            _accumulate(prop_name, prop.get("token_ids"),
                        prop.get("parents"), gt)

    # Pre-pass B: live suffix. Suffix tree is generated at sim time from a
    # SuffixDecodingCache fed with ground-truth tokens (method-independent
    # ceiling). Mirrors the cache feed pattern simulate_decoding uses.
    try:
        import numpy as _np
        from hybrid_spec_decoding.suffix_decoding.suffix_tree import (
            SuffixDecodingCache as _PA_Cache,
        )
        _suffix_cache = _PA_Cache(
            max_tree_depth=POSITION_ACCEPT_MAX,
            max_cached_requests=100000,
        )

        # Group records by (request_id, call_idx) and order by step_idx.
        from collections import defaultdict as _dd
        _by_seq: dict = _dd(list)
        for rec in records:
            _by_seq[(rec["request_id"], rec.get("call_idx", 0))].append(rec)
        for _k in _by_seq:
            _by_seq[_k].sort(key=lambda r: r.get("step_idx", 0))

        for (_rid, _cid), _seq in _by_seq.items():
            if not _seq:
                continue
            _cache_req_id = f"{_rid}_{_cid}"
            _prompt = _seq[0].get("context_token_ids") or []
            _suffix_cache.start_request(
                _cache_req_id, _np.asarray(_prompt, dtype=_np.int32))
            for _i, rec in enumerate(_seq):
                _ctx = rec.get("context_token_ids") or []
                _gt = rec.get("ground_truth_future") or []
                if _ctx and _gt:
                    try:
                        _draft = _suffix_cache.speculate(
                            _cache_req_id,
                            _np.asarray(_ctx, dtype=_np.int32),
                            max_spec_factor=4.0,
                            min_token_prob=0.0,
                            max_spec_tokens=256,
                            use_tree_spec=True,
                        )
                        if _draft.token_ids:
                            _accumulate(
                                "suffix",
                                list(_draft.token_ids),
                                list(_draft.parents),
                                _gt,
                            )
                    except Exception:
                        pass
                # Advance: feed gt tokens up to the next step's offset.
                if _i < len(_seq) - 1:
                    _adv = (_seq[_i + 1].get("step_idx", 0)
                            - rec.get("step_idx", 0))
                else:
                    _adv = 1
                if _gt and _adv > 0:
                    _suffix_cache.add_active_response(
                        _cache_req_id, list(_gt[:_adv]))
            _suffix_cache.stop_request(_cache_req_id)
    except Exception as _e:
        import sys as _sys
        print(f"WARN: suffix position-accept pre-pass skipped: {_e}",
              file=_sys.stderr)

    # Latency-aware simulation. When --latency-config is missing, feed a
    # stub config (vanilla_step_ms=1.0, empty per-budget tables); speedup
    # numbers become placeholders but MAT is unaffected.
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

    method_filter = None
    if args.methods:
        method_filter = set(m.strip() for m in args.methods.split(",")
                            if m.strip())

    latency_results = compute_latency_speedup(
        records, budgets, latency_config,
        topk=args.topk, steps=args.steps,
        method_filter=method_filter)

    if args.print_summary:
        print_summary(budgets)
        try:
            print_latency_summary(latency_results, budgets,
                                  latency_config["vanilla_step_ms"])
            if not have_latency:
                print("(WARNING: speedup columns above use stub latency; "
                      "only MAT is meaningful)", file=sys.stderr)
        except Exception as e:
            print(f"Warning: print_latency_summary failed: {e}",
                  file=sys.stderr)

    if args.output:
        output = {
            "metadata": {
                "input_source": input_source,
                "n_steps": len(records),
                "budgets": budgets,
            },
            "position_accepts": {
                "max_position": POSITION_ACCEPT_MAX,
                "by_proposer": position_accepts,
                "_doc": (
                    "Per-position draft-token accept counts (depth=position). "
                    "seq_accept[d-1] = #steps where position d accepted via "
                    "greedy walk (requires positions 1..d all match). "
                    "ind_accept[d-1] = #steps where ANY node at depth d "
                    "matches ground_truth[d-1] regardless of ancestors. "
                    "depth_ge[d-1] = denominator: #steps where the draft tree "
                    "has depth ≥ d AND ground_truth has length ≥ d. "
                    "Coverage: eagle3 + draft_model from records, plus "
                    "suffix from a live SuffixDecodingCache fed with the "
                    "ground-truth trajectory (method-independent ceiling). "
                    "mtp + extension are NOT measured."
                ),
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
