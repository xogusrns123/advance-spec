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
* ``extension_joint_score:t`` / ``_oracle:t`` — suffix attached only when
  ``draft.score × path_p_t ≥ t`` (joint eagle3 + suffix confidence).
* ``extension_hybrid:t`` / ``_oracle:t`` — per-step suffix-only vs ext.

Forbidden methods (removed):
  - extension_oracle_path (path-only accounting unrealistic)
  - extension_hybrid_perfect_oracle* (per-step oracle gate unrealistic)
  - extension_dual_method*, extension_dmsfx*, extension_2level*,
    extension_sfx_backbone*, extension_anchor*, extension_hybrid_prune_pt*,
    extension_pure_sfx*
  - extension_prune_pt, extension_by_joint (backbone pruning dropped)
  - extension_by_count, extension_dm_by_count, extension_by_count_score
    (only score-based filter retained)

Usage:
    python3 -m simulation.evaluation.run_tree_oracle_sim \\
        --agent-trajectory results/.../agent_results_eagle3.json \\
        --draft-model-drafts results/.../draft_model_drafts.jsonl \\
        --dataset data/specbench/dataset.jsonl \\
        --model Qwen/Qwen3-14B \\
        --budgets 1,2,4,8,16,32,64,128 \\
        --latency-data simulation/config/latency/qwen3_14b.json \\
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


def _const_draft(B, val):
    """Picklable constant-draft helper. Used when FIXED_DRAFT_MS overrides the
    per-step draft cost (depth-independent draft model) for methods that call
    real_step_draft_fn per-step in parallel mode (extension_oracle etc.)."""
    return val


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

from simulation.evaluation.tree_knapsack import greedy_tree_walk, greedy_tree_walk_path


# BNS branching/score-distribution trace (env-gated).
# Populated inside _extension_step when bns_enabled and BNS_TRACE_OUT env is set.
# Cleared after each method's sim by simulate_decoding (just before return).
_BNS_TRACE = {
    'enabled': os.environ.get('BNS_TRACE_OUT') is not None,
    'max_records': 50000,
    'records': [],
    'agg': {
        'parent_count': 0,
        'total_orig_kids': 0,
        'total_kept_kids': 0,
        'top_R_sum': 0.0,
        'orig_kids_hist': {},  # {n: count}
        'kept_kids_hist': {},  # {n: count}
    },
}

# BNS score-calibration trace (env-gated).
# For each (parent on greedy accept path, child) pair on the *full* (pre-prune)
# extended tree, dump (norm_R, accepted, k_siblings, parent_depth) so we can
# bin norm_R and measure P(accept | norm_R) — i.e. whether normalized R is a
# calibrated predictor of acceptance among siblings.
_BNS_CALIB = {
    'enabled': os.environ.get('BNS_CALIB_OUT') is not None,
    'max_records': 5_000_000,
    'records': [],  # list of [norm_R, accepted, k, depth]
}


# === Online per-edge accept-rate calibration (extension_calib_*) ===
# Ridge-regularized OLS that maps per-edge features → conditional accept rate
#   c(e) = P(edge accepted | parent reached).
# Two INDEPENDENT models — eagle3 backbone edges vs suffix-decoding graft edges
# — so the count-based suffix scale and the softmax-based eagle scale are each
# brought onto the common "accept rate" scale where DNS/topk compare them. This
# generalizes the single ``bns_lambda_suffix`` scalar: λ is the degenerate case
# "suffix model = intercept only". Suffix-decoding drafts only exist at sim time
# (generated by speculate()), so the calibration is fit ONLINE during the sim:
# running sufficient statistics (XtX, Xty) updated each step from that step's
# full-tree accept path, used causally on subsequent steps. Closed-form solve,
# no RNG / learning rate → identical results under SIM_PARALLEL fork workers.
_CALIB_FEAT_DIM = 6


def _calib_features(edge_prob, depth, n_desc, n_sib, match_len):
    """Per-edge feature vector. Counts are log/scaled so a single ridge term
    keeps the normal-equations matrix well-conditioned across feature scales."""
    import math
    return [
        1.0,                                # intercept
        math.log(max(edge_prob, 1e-6)),     # raw per-edge prob (log)
        depth / 10.0,                       # tree depth
        math.log1p(max(n_desc, 0)),         # subtree size (trie hub-ness)
        math.log1p(max(n_sib, 0)),          # sibling fan-out at parent
        match_len / 10.0,                   # suffix match length (0 for backbone)
    ]


class _OnlineEdgeCalibrator:
    """Per-group (eagle/suffix) online linear-probability calibrator.

    Accumulates XtX/Xty; solves β = (XtX + ridge·I)⁻¹ Xty lazily. ``predict``
    returns the calibrated conditional accept rate clipped to (0, 1]; until a
    group has ``min_samples`` observations it returns the supplied raw-prob
    fallback (= λ=1 behavior), so the cold-start window degrades gracefully.
    """

    def __init__(self, ridge: float = 1.0, min_samples: int = 300):
        self.ridge = ridge
        self.min_samples = min_samples
        d = _CALIB_FEAT_DIM
        self._S = {'eagle': np.zeros((d, d)), 'suffix': np.zeros((d, d))}
        self._b = {'eagle': np.zeros(d), 'suffix': np.zeros(d)}
        self._n = {'eagle': 0, 'suffix': 0}
        self._beta = {'eagle': None, 'suffix': None}
        self._dirty = {'eagle': False, 'suffix': False}

    def update(self, group: str, x, y: float) -> None:
        xv = np.asarray(x, dtype=np.float64)
        self._S[group] += np.outer(xv, xv)
        self._b[group] += xv * y
        self._n[group] += 1
        self._dirty[group] = True

    def _solve(self, group: str):
        if self._dirty[group] or self._beta[group] is None:
            S = self._S[group] + self.ridge * np.eye(_CALIB_FEAT_DIM)
            try:
                self._beta[group] = np.linalg.solve(S, self._b[group])
            except np.linalg.LinAlgError:
                self._beta[group] = None
            self._dirty[group] = False
        return self._beta[group]

    def predict(self, group: str, x, fallback: float) -> float:
        if self._n[group] < self.min_samples:
            return fallback
        beta = self._solve(group)
        if beta is None:
            return fallback
        val = float(np.dot(beta, np.asarray(x, dtype=np.float64)))
        if val < 1e-6:
            return 1e-6
        if val > 1.0:
            return 1.0
        return val


class _CalibSampleCollector:
    """FIT-pass sample collector for offline isotonic calibration
    (``extension_isofit_*``).

    ``predict`` returns the raw fallback, so the fit pass selects trees
    exactly like the λ=1 variant of the same selector while ``update``
    records one (edge_prob, depth, accept) sample per evaluable full-tree
    edge. ``dump`` writes one JSON line per simulate_decoding call to a
    worker-unique file (``<SIM_ISO_COLLECT_OUT>.b<budget>.<pid>.part``) so
    SIM_PARALLEL workers never interleave writes.
    """
    wants_shrunk_edge_probs = (os.environ.get("SIM_SUFFIX_SHRINK", "1") == "1")

    def __init__(self):
        self.samples = {'eagle': [], 'suffix': []}

    def update(self, group: str, x, y: float) -> None:
        import math
        # x[1] = log(max(edge_prob, 1e-6)), x[2] = depth/10 (see _calib_features).
        self.samples[group].append(
            (round(math.exp(x[1]), 8), x[2] * 10.0, float(y)))

    def predict(self, group: str, x, fallback: float) -> float:
        return fallback

    def dump(self, path_prefix: str, meta: dict) -> None:
        out = f"{path_prefix}.b{meta.get('budget', 0)}.{os.getpid()}.part"
        with open(out, "a") as f:
            f.write(json.dumps({"meta": meta, "samples": self.samples}) + "\n")


class _FrozenIsoCalibrator:
    """EVAL-pass frozen isotonic map raw_edge_prob → P(accept) per group
    (``extension_iso_*``).

    Loaded from the JSON written by
    simulation/scripts/experiments/fit_iso_calibration.py (fitted on the
    TRAIN request split). ``predict`` reads only x[1] (log edge prob —
    already Jeffreys-shrunk for suffix edges when the map was fitted with
    shrinkage, mirrored here via ``wants_shrunk_edge_probs``) and does a
    step-function lookup. ``update`` is a no-op: the map never trains on
    eval-split data.
    """

    def __init__(self, blob: dict):
        self._maps = {}
        for grp, m in blob["groups"].items():
            self._maps[grp] = (np.asarray(m["x"], dtype=np.float64),
                               np.asarray(m["y"], dtype=np.float64))
        self.wants_shrunk_edge_probs = bool(
            blob.get("meta", {}).get("shrink"))

    @classmethod
    def load(cls, path: str) -> "_FrozenIsoCalibrator":
        with open(path) as f:
            return cls(json.load(f))

    def update(self, group: str, x, y: float) -> None:
        pass

    def predict(self, group: str, x, fallback: float) -> float:
        import math
        m = self._maps.get(group)
        if m is None:
            return fallback
        xs, ys = m
        p = math.exp(x[1])
        i = int(np.searchsorted(xs, p, side="right")) - 1
        if i < 0:
            i = 0
        v = float(ys[i])
        return v if v > 1e-6 else 1e-6


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

    # Request-level train/test split (offline calibration experiments).
    # SIM_REQ_SPLIT="train:0.5" keeps the first ceil(50%) of the sorted
    # unique request ids; "test:0.5" keeps the complement. The env applies
    # to EVERY method in the invocation, so baselines and calibrated
    # methods are scored on the same request set.
    _split_env = os.environ.get("SIM_REQ_SPLIT")
    if _split_env:
        _part, _frac_s = _split_env.split(":")
        if _part not in ("train", "test"):
            raise ValueError(f"bad SIM_REQ_SPLIT: {_split_env!r}")
        _req_ids = sorted({sk[0] for sk in sequences})
        _n_train = int(round(len(_req_ids) * float(_frac_s)))
        _train_set = set(_req_ids[:_n_train])
        sequences = {
            sk: v for sk, v in sequences.items()
            if (sk[0] in _train_set) == (_part == "train")
        }

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

    # Optional per-method budget breakdown (backbone vs extension graft).
    # Gated on EXTENSION_BREAKDOWN=1 to avoid overhead in normal sweeps.
    _BREAKDOWN_ON = os.environ.get("EXTENSION_BREAKDOWN") == "1"
    bd_base_size = 0
    bd_graft_size = 0
    bd_accepted_base = 0
    bd_accepted_suffix = 0

    # Optional per-step JSONL dump (one row per step). Gated on env var
    # SIM_PER_STEP_JSONL=<path>. Opens in append mode so multiple
    # simulate_decoding calls (e.g. across budgets) accumulate.
    _per_step_path = os.environ.get("SIM_PER_STEP_JSONL")
    _per_step_fh = None
    if _per_step_path:
        _per_step_fh = open(_per_step_path, "a")

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
        # enable_undo=True powers the per-anchor temporary_extension flow
        # in _extension_step (paths[i] is appended to BOTH local & global
        # trees, speculate runs, then pop reverses both bit-exactly).
        local_cache = _FreshCache(
            max_tree_depth=64, max_cached_requests=100000,
            enable_undo=True,
        )
    else:
        local_cache = None

    # Online accept-rate calibrator (extension_calib_*). One instance per
    # simulate_decoding call (= per method, per worker process); its running
    # OLS state accumulates across this method's steps. None for all other
    # methods, leaving their λ-based selection untouched.
    if method.startswith("extension_calib_"):
        _calib = _OnlineEdgeCalibrator()
    elif method.startswith("extension_isofit_"):
        # Offline-isotonic FIT pass: λ=1 selection + full-tree label collection.
        _calib = _CalibSampleCollector()
    elif method.startswith("extension_iso_"):
        # Offline-isotonic EVAL pass: frozen train-split map, no updates.
        _calib = _FrozenIsoCalibrator.load(os.environ["SIM_ISO_CALIB"])
    else:
        _calib = None

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
                # Env override: EXTENSION_ORACLE_BUDGET_GRID="1,2,4,8,..."
                _grid_env = os.environ.get("EXTENSION_ORACLE_BUDGET_GRID")
                if _grid_env:
                    _BUDGET_GRID = tuple(int(x) for x in _grid_env.split(","))
                else:
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
            elif method.startswith("extension_gd:"):
                # extension_gd:D[:F:T] — graft suffix ONLY at anchor depth D
                # (D=0 = root hybrid). Independent single-depth-graft config.
                parts = method.split(":")
                D = int(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0, only_depth=D)
            elif method.startswith("extension_stem:"):
                # extension_stem:D[:F:T] — D backbone tokens THEN suffix stem
                # only (backbone truncated to depth<D, so beyond the anchor
                # there is NO backbone; only the suffix graft at depth D).
                # Measured over ALL steps. D=0 = pure suffix-only (no backbone).
                parts = method.split(":")
                D = int(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0, only_depth=D, stem_depth=D)
            elif method.startswith("extension_cumd:"):
                # extension_cumd:D[:F:T] — cumulative graft at anchor depths
                # 0..D (D=0 = root hybrid; D=8 ~ full extension).
                parts = method.split(":")
                D = int(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0, cum_depth=D)
            elif method.startswith("extension_by_score:"):
                # extension_by_score:t[:F:T] — score threshold + optional FT.
                parts = method.split(":")
                threshold = float(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3", score_threshold=threshold,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_by_pt_alpha:"):
                # extension_by_pt_alpha:alpha:threshold[:F:T] — anchor skip if
                # (path_p_t[node] ** alpha) < threshold. Base tree unchanged.
                parts = method.split(":")
                alpha = float(parts[1])
                pt_t = float(parts[2])
                F, T = (4.0, 0.0)
                if len(parts) >= 5:
                    F, T = float(parts[3]), float(parts[4])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    pt_threshold=pt_t, pt_alpha=alpha,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_by_combined:"):
                # extension_by_combined:alpha:pt:score[:F:T] — AND filter.
                # Anchor passes only if (path_p_t**alpha >= pt) AND (score >= score).
                parts = method.split(":")
                alpha = float(parts[1])
                pt_t = float(parts[2])
                sc_t = float(parts[3])
                F, T = (4.0, 0.0)
                if len(parts) >= 6:
                    F, T = float(parts[4]), float(parts[5])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    pt_threshold=pt_t, pt_alpha=alpha,
                    score_threshold=sc_t,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_by_product:"):
                # extension_by_product:alpha:threshold[:F:T] — multiplicative joint.
                # Anchor passes if (score * path_p_t**alpha) >= threshold.
                parts = method.split(":")
                alpha = float(parts[1])
                prod_t = float(parts[2])
                F, T = (4.0, 0.0)
                if len(parts) >= 5:
                    F, T = float(parts[3]), float(parts[4])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    product_threshold=prod_t, product_alpha=alpha,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_pt_prune:"):
                # extension_pt_prune:alpha:pt[:F:T] — BACKBONE pruning by path_p_t^α.
                # Both backbone nodes AND graft skip use the same threshold.
                parts = method.split(":")
                alpha = float(parts[1])
                pt_t = float(parts[2])
                F, T = (4.0, 0.0)
                if len(parts) >= 5:
                    F, T = float(parts[3]), float(parts[4])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    pt_threshold=pt_t, pt_alpha=alpha,           # graft side
                    backbone_pt_threshold=pt_t, backbone_pt_alpha=alpha,  # backbone side (new)
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_top1:") or method == "extension_top1":
                # extension_top1[:F:T] — graft only on root + top-1 chain through backbone.
                F, T = (4.0, 0.0)
                if ":" in method:
                    parts = method.split(":")
                    if len(parts) >= 3:
                        F, T = float(parts[1]), float(parts[2])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    top1_chain=True,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_topb_pathprob:") or method == "extension_topb_pathprob":
                # extension_topb_pathprob[:F:T] — unified token-level top-B selection.
                # Rank: path_p_t(v) for backbone, anchor_path × draft.probs[i] for grafts.
                # Selects top-budget nodes globally with ancestor closure.
                F, T = (4.0, 0.0)
                if ":" in method:
                    parts = method.split(":")
                    if len(parts) >= 3:
                        F, T = float(parts[1]), float(parts[2])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    topb_unified_rank=True,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_topb_pathprob_alpha:"):
                # extension_topb_pathprob_alpha:alpha[:F:T] — topb with α calibration on
                # extension probs. Rank: path_p_t(v) for backbone, anchor_path × draft.probs[i]^α
                # for grafts. α > 1 compresses overconfident count-based probs to balance
                # against softmax-based backbone path probs.
                parts = method.split(":")
                alpha = float(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    topb_unified_rank=True,
                    topb_alpha_ext=alpha,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_topb_pathprob_keep1:"):
                # extension_topb_pathprob_keep1:alpha[:F:T] — topb_alpha + force-preserve
                # the top-1 backbone chain (greedy argmax-child from root) regardless of
                # budget. Tests whether deep backbone chain truncation hurts acceptance.
                parts = method.split(":")
                alpha = float(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    topb_unified_rank=True,
                    topb_alpha_ext=alpha,
                    topb_keep1_chain=True,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_bns:"):
                # extension_bns:lambda_suffix:rho[:F:T] — Branch Nucleus Selection.
                # Recursive sibling pruning by subtree PathScore mass. λ_suffix
                # re-weights every suffix-sourced edge (count-based probs are
                # not calibrated against EAGLE3 softmax probs). ρ is the
                # nucleus mass threshold per sibling group. No budget cap —
                # tree size is implicitly controlled by ρ and λ_suffix.
                parts = method.split(":")
                lam = float(parts[1])
                rho = float(parts[2])
                F, T = (4.0, 0.0)
                if len(parts) >= 5:
                    F, T = float(parts[3]), float(parts[4])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    bns_enabled=True, bns_rho=rho, bns_lambda_suffix=lam,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_dns:") or method.startswith("extension_bns_depth:"):
                # extension_dns:lambda_suffix:rho[:F:T] — Depth-wise Nucleus
                # Selection (formerly BNSv2). The "extension_bns_depth:" prefix
                # is kept as a legacy alias for backward compatibility with
                # previously-collected results.
                # Three differences from BNSv1:
                #   (A) dedup boost: max(eagle3_prob, λ * suffix_prob) at any
                #       (parent, token) reached by both proposers.
                #   (B) score = per-node path prob (no subtree rollup), avoiding
                #       the "big subtree advantage" of BNSv1's R(c).
                #   (C) nucleus is per-DEPTH (not per-parent): all depth-d
                #       candidates compete in one pool.
                parts = method.split(":")
                lam = float(parts[1])
                rho = float(parts[2])
                F, T = (4.0, 0.0)
                if len(parts) >= 5:
                    F, T = float(parts[3]), float(parts[4])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    dns_enabled=True, bns_rho=rho, bns_lambda_suffix=lam,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_dnsv2:"):
                # extension_dnsv2:lambda_suffix:rho[:F:T] — DNS without
                # candidate-pool normalization. Cumulative sum uses raw
                # path-prob; ρ is an absolute mass threshold.
                parts = method.split(":")
                lam = float(parts[1])
                rho = float(parts[2])
                F, T = (4.0, 0.0)
                if len(parts) >= 5:
                    F, T = float(parts[3]), float(parts[4])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    dnsv2_enabled=True, bns_rho=rho, bns_lambda_suffix=lam,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_topk:"):
                # extension_topk:lambda_suffix:k[:F:T] — per-depth top-k.
                # Same effective-prob (dedup-boost + λ-scaled suffix) and
                # path-prob scoring as DNS, but the per-depth selection
                # rule is "keep first k by path-prob" instead of "cumulative
                # ≥ ρ nucleus". Tree size bounded per depth at k.
                parts = method.split(":")
                lam = float(parts[1])
                k = int(parts[2])
                F, T = (4.0, 0.0)
                if len(parts) >= 5:
                    F, T = float(parts[3]), float(parts[4])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    topk_depth_enabled=True, topk_depth_k=k,
                    bns_lambda_suffix=lam,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_calib_dns:"):
                # extension_calib_dns:rho[:F:T] — DNS (depth-wise nucleus)
                # selection where the per-edge λ_suffix re-weighting is
                # REPLACED by the online accept-rate calibrator (`_calib`).
                # eagle3 and suffix edges are each mapped to a calibrated
                # conditional accept rate, so λ is no longer a free knob.
                parts = method.split(":")
                rho = float(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    dns_enabled=True, bns_rho=rho,
                    calib=_calib,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_calib_topk:"):
                # extension_calib_topk:k[:F:T] — per-depth top-k with the
                # online accept-rate calibrator in place of λ_suffix.
                parts = method.split(":")
                k = int(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    topk_depth_enabled=True, topk_depth_k=k,
                    calib=_calib,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif (method.startswith("extension_isofit_topk:")
                  or method.startswith("extension_iso_topk:")):
                # Offline isotonic calibration, both passes share the per-depth
                # top-k selector:
                #   extension_isofit_topk:k[:F:T] — FIT: collector predict()
                #     returns the raw (count-shrunk) edge prob, so selection
                #     behaves like extension_topk:1.0:k while full-tree accept
                #     labels are logged (train split via SIM_REQ_SPLIT).
                #   extension_iso_topk:k[:F:T] — EVAL: suffix edge probs are
                #     Jeffreys count-shrunk, then BOTH proposers' edge probs
                #     map through the frozen per-group isotonic accept-rate
                #     curves (SIM_ISO_CALIB) — λ_suffix has no role.
                parts = method.split(":")
                k = int(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    topk_depth_enabled=True, topk_depth_k=k,
                    calib=_calib,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_by_match_len:"):
                # extension_by_match_len:M[:F:T] — match-length gated basic
                # extension. Only attach suffix grafts (root + per-anchor)
                # whose speculate() returned match_len >= M. Idea: short
                # matches yield noisy, low-confidence drafts that pollute
                # the verify budget; gating them out should let the rest
                # of the budget cover higher-quality candidates.
                parts = method.split(":")
                M = int(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    match_len_threshold=M,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_topk_match:"):
                # extension_topk_match:lambda_suffix:k:M[:F:T] — topk per-depth
                # selection combined with match_len gate. The gate filters
                # at attach-time; per-depth selection then prunes the
                # remaining grafts. Stacking two orthogonal signals.
                parts = method.split(":")
                lam = float(parts[1])
                k = int(parts[2])
                M = int(parts[3])
                F, T = (4.0, 0.0)
                if len(parts) >= 6:
                    F, T = float(parts[4]), float(parts[5])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    topk_depth_enabled=True, topk_depth_k=k,
                    bns_lambda_suffix=lam,
                    match_len_threshold=M,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_descrank:"):
                # extension_descrank:lambda_suffix:k:alpha[:F:T] — per-depth
                # top-k by composite path_prob × (1 + alpha · log(1+n_desc)).
                # alpha=0 reduces to extension_topk exactly (sanity baseline).
                parts = method.split(":")
                lam = float(parts[1])
                k = int(parts[2])
                alpha = float(parts[3])
                F, T = (4.0, 0.0)
                if len(parts) >= 6:
                    F, T = float(parts[4]), float(parts[5])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    descrank_enabled=True, descrank_alpha=alpha,
                    topk_depth_k=k,
                    bns_lambda_suffix=lam,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_depth_cap:"):
                # extension_depth_cap:D[:F:T] — basic extension + hard depth
                # cap. Per-node analysis: ~20% of suffix-tree budget lives at
                # depth > 10 but contains <3% of accepted nodes. Cap frees
                # budget for shallow candidates where accepts actually cluster.
                parts = method.split(":")
                D = int(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    depth_cap=D,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_topk_cap:"):
                # extension_topk_cap:lambda_suffix:k:D[:F:T] — per-depth topk
                # selection stacked with depth cap. Removes deep-tail waste
                # AND keeps per-depth selection at the surviving levels.
                parts = method.split(":")
                lam = float(parts[1])
                k = int(parts[2])
                D = int(parts[3])
                F, T = (4.0, 0.0)
                if len(parts) >= 6:
                    F, T = float(parts[4]), float(parts[5])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="eagle3",
                    topk_depth_enabled=True, topk_depth_k=k,
                    bns_lambda_suffix=lam,
                    depth_cap=D,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
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
            elif method.startswith("extension_dm_by_score:"):
                parts = method.split(":")
                threshold = float(parts[1])
                F, T = (4.0, 0.0)
                if len(parts) >= 4:
                    F, T = float(parts[2]), float(parts[3])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model", score_threshold=threshold,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_dm_by_pt_alpha:"):
                # draft_model backbone version of extension_by_pt_alpha.
                parts = method.split(":")
                alpha = float(parts[1])
                pt_t = float(parts[2])
                F, T = (4.0, 0.0)
                if len(parts) >= 5:
                    F, T = float(parts[3]), float(parts[4])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model",
                    pt_threshold=pt_t, pt_alpha=alpha,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_dm_by_combined:"):
                parts = method.split(":")
                alpha = float(parts[1])
                pt_t = float(parts[2])
                sc_t = float(parts[3])
                F, T = (4.0, 0.0)
                if len(parts) >= 6:
                    F, T = float(parts[4]), float(parts[5])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model",
                    pt_threshold=pt_t, pt_alpha=alpha,
                    score_threshold=sc_t,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
            elif method.startswith("extension_dm_by_product:"):
                parts = method.split(":")
                alpha = float(parts[1])
                prod_t = float(parts[2])
                F, T = (4.0, 0.0)
                if len(parts) >= 5:
                    F, T = float(parts[3]), float(parts[4])
                accepted, ext_size = _extension_step(
                    rec, budget, local_cache, cache_req_id,
                    base_proposer="draft_model",
                    product_threshold=prod_t, product_alpha=alpha,
                    suffix_max_spec_factor=F, suffix_min_token_prob=T,
                    suffix_max_spec_tokens=0)
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
            # Force advance=1 to ensure every captured step yields one dump
            # row (gated by env var; default behavior unchanged).
            if os.environ.get("SIM_FORCE_ADVANCE_1") == "1":
                advance = 1
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
                    # Per-step breakdown (only for extension family — side-channel
                    # set inside _extension_step). Oracle's last call is for the
                    # last budget tested, not the picked one — slightly off; we
                    # only track for non-oracle methods anyway.
                    if _BREAKDOWN_ON and not (method == "extension_oracle"
                                              or method.startswith("extension_oracle:")
                                              or method == "extension_dm_oracle"
                                              or method.startswith("extension_dm_oracle:")):
                        try:
                            # graft_size = ALL grafts attached at any anchor
                            # (= ext_size_full - base_size), which matches the
                            # tokens actually paid for in the target verify.
                            bd_base_size += _extension_step._last_base_size
                            bd_graft_size += (_extension_step._last_ext_size_full
                                              - _extension_step._last_base_size)
                            bd_accepted_base += _extension_step._last_accepted_base
                            bd_accepted_suffix += _extension_step._last_accepted_suffix
                        except AttributeError:
                            pass
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

            # Per-step JSONL dump (one row per simulated step).
            if _per_step_fh is not None:
                _row = {
                    "method": method,
                    "budget": budget,
                    "request_id": req_id,
                    "call_idx": call_idx,
                    "step_id": pos,
                    "accepted": int(accepted),
                    "ext_size": (int(ext_size)
                                 if ext_size is not None else None),
                }
                # Per-step real-cost breakdown (only available when dynamic
                # target_fn path was taken). Compute fields from locals if
                # they're in scope; otherwise null.
                _row["target_ms"] = (locals().get("target_ms_step")
                                     if ext_size is not None else None)
                _row["draft_ms"] = (
                    locals().get("_draft_ms")
                    if ext_size is not None else
                    real_step_cost_ms if not is_hybrid else None)
                _row["total_step_ms"] = (
                    locals().get("step_real")
                    if ext_size is not None else
                    real_step_cost_ms if not is_hybrid else None)
                # Tree topology arrays. Extension family uses _extension_step
                # side-channel; single:* uses _proposer_tree_walk side-channel.
                _tids = getattr(_extension_step, "_last_ext_tids", None)
                _pids = getattr(_extension_step, "_last_ext_pids", None)
                _nb = getattr(_extension_step, "_last_ext_n_base", None)
                _ap = getattr(_extension_step, "_last_accepted_path", None)
                if _tids is not None and _pids is not None and _nb is not None:
                    _row["tree_token_ids"] = list(_tids)
                    _row["tree_parents"] = list(_pids)
                    _row["tree_n_base"] = int(_nb)
                    _accepted_set = set(_ap or [])
                    _row["tree_is_accepted"] = [
                        bool(i in _accepted_set) for i in range(len(_tids))]
                    _row["tree_source"] = [
                        "eagle" if i < _nb else "suffix"
                        for i in range(len(_tids))]
                    # Raw path_prob + anchor_node_id propagation (added for
                    # bfcl full-analysis xlsx). Only populated for extension
                    # family; None / -2 fallback otherwise.
                    _pp = getattr(_extension_step,
                                  "_last_ext_path_prob", None)
                    _ai = getattr(_extension_step,
                                  "_last_ext_anchor_id", None)
                    _sf = getattr(_extension_step,
                                  "_last_ext_suffix_freq", None)
                    _sc = getattr(_extension_step,
                                  "_last_ext_suffix_cum_prob", None)
                    _ml = getattr(_extension_step,
                                  "_last_ext_match_len", None)
                    if _pp is not None and len(_pp) == len(_tids):
                        _row["tree_path_prob"] = list(_pp)
                    if _ai is not None and len(_ai) == len(_tids):
                        _row["tree_anchor_node_id"] = list(_ai)
                    if _sf is not None and len(_sf) == len(_tids):
                        _row["tree_suffix_freq"] = list(_sf)
                    if _sc is not None and len(_sc) == len(_tids):
                        _row["tree_suffix_cum_prob"] = list(_sc)
                    if _ml is not None and len(_ml) == len(_tids):
                        _row["tree_match_len"] = list(_ml)
                    # Ground truth info from this captured step
                    _gt = rec.get("ground_truth_future") or []
                    if _gt:
                        _row["ground_truth_token"] = int(_gt[0])
                        # cap future to 32 tokens to keep cell small
                        _row["ground_truth_future"] = [
                            int(t) for t in _gt[:32]]
                    _extension_step._last_ext_tids = None
                    _extension_step._last_ext_pids = None
                    _extension_step._last_ext_n_base = None
                    _extension_step._last_accepted_path = None
                    _extension_step._last_ext_path_prob = None
                    _extension_step._last_ext_anchor_id = None
                    _extension_step._last_ext_suffix_freq = None
                    _extension_step._last_ext_suffix_cum_prob = None
                    _extension_step._last_ext_match_len = None
                else:
                    # single:* methods leave data on _proposer_tree_walk.
                    _ptids = getattr(_proposer_tree_walk,
                                     "_last_tids", None)
                    _ppids = getattr(_proposer_tree_walk,
                                     "_last_pids", None)
                    _pap = getattr(_proposer_tree_walk,
                                   "_last_accepted_path", None)
                    if _ptids is not None and _ppids is not None:
                        _row["tree_token_ids"] = list(_ptids)
                        _row["tree_parents"] = list(_ppids)
                        # For single:* methods the tree is the entire base
                        # proposer's tree — all nodes are "eagle" / "draft"
                        # source. Default to method-name suffix.
                        _src = (method.split(":", 1)[1]
                                if ":" in method else "eagle")
                        _row["tree_n_base"] = len(_ptids)
                        _row["tree_source"] = [_src] * len(_ptids)
                        _accepted_set = set(_pap or [])
                        _row["tree_is_accepted"] = [
                            bool(i in _accepted_set)
                            for i in range(len(_ptids))]
                        _proposer_tree_walk._last_tids = None
                        _proposer_tree_walk._last_pids = None
                        _proposer_tree_walk._last_accepted_path = None
                if os.environ.get("SIM_PER_STEP_SLIM") == "1":
                    # Slim row (no tree arrays) — tiny, so concurrent O_APPEND
                    # writes stay atomic under SIM_PARALLEL. Enough for survival.
                    _per_step_fh.write(json.dumps(
                        {"method": method, "request_id": req_id,
                         "call_idx": call_idx, "step_id": pos,
                         "accepted": int(accepted)},
                        separators=(",", ":")) + "\n")
                else:
                    _per_step_fh.write(json.dumps(
                        _row, separators=(",", ":")) + "\n")

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

    # Close per-step dump file
    if _per_step_fh is not None:
        try:
            _per_step_fh.close()
        except Exception:
            pass

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
    if _BREAKDOWN_ON:
        result["budget_breakdown"] = {
            "base_size": bd_base_size,
            "graft_size": bd_graft_size,
            "accepted_base": bd_accepted_base,
            "accepted_suffix": bd_accepted_suffix,
            "wasted_base": bd_base_size - bd_accepted_base,
            "wasted_suffix": bd_graft_size - bd_accepted_suffix,
        }
    if total_time_real_always_ms is not None:
        result["speedup_real_always"] = (vanilla_time_ms / total_time_real_always_ms
                                         if total_time_real_always_ms > 0 else 1.0)

    # Offline-isotonic FIT pass: persist the collected (edge_prob, depth,
    # accept) samples for fit_iso_calibration.py.
    if isinstance(_calib, _CalibSampleCollector):
        _iso_out = os.environ.get("SIM_ISO_COLLECT_OUT")
        if _iso_out:
            _calib.dump(_iso_out, {"method": method, "budget": budget,
                                   "n_steps": total_steps})
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
                    pt_alpha: float = 1.0,
                    product_threshold: Optional[float] = None,
                    product_alpha: float = 1.0,
                    backbone_pt_threshold: Optional[float] = None,
                    backbone_pt_alpha: float = 1.0,
                    top1_chain: bool = False,
                    topb_unified_rank: bool = False,
                    topb_alpha_ext: float = 1.0,
                    topb_keep1_chain: bool = False,
                    bns_enabled: bool = False,
                    dns_enabled: bool = False,
                    dnsv2_enabled: bool = False,
                    bns_rho: float = 0.90,
                    bns_lambda_suffix: float = 1.0,
                    topk_depth_enabled: bool = False,
                    topk_depth_k: int = 1,
                    descrank_enabled: bool = False,
                    descrank_alpha: float = 0.5,
                    match_len_threshold: int = 0,
                    depth_cap: int = 0,
                    calib: "Optional[_OnlineEdgeCalibrator]" = None,
                    suffix_max_spec_factor: float = 4.0,
                    suffix_min_token_prob: float = 0.0,
                    suffix_max_spec_tokens: int = 0,
                    only_depth: Optional[int] = None,
                    cum_depth: Optional[int] = None,
                    stem_depth: Optional[int] = None):
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
    # When topb_unified_rank is set, we DON'T pre-truncate the base tree
    # — top-B selection happens after all extensions.
    # BNS, however, MUST respect the B-cap on backbone: the intended pipeline is
    #   1) truncate backbone to B → 2) attach extensions (= basic's full tree)
    #   → 3) BNS pruning. So BNS's tree must be a subset of basic at same B.
    if topb_unified_rank:
        n = len(tids)
    else:
        n = min(budget, len(tids))
    tids = tids[:n]
    pids = pids[:n]
    pids = [p if p < n else -1 for p in pids]

    # EXT_STEM_DEPTH=D: truncate the BACKBONE to tree-depth < D (keep the first
    # D backbone tokens) BEFORE grafting. Done early (not after) so the suffix
    # graft at depth D-1 has no deeper backbone to dedup-merge into — otherwise
    # a suffix token coinciding with a deep backbone token would merge into that
    # backbone node and be lost when deep backbone is removed. D=0 → no backbone
    # at all (only the vroot graft survives) = pure suffix-only.
    import os as _os_sd
    if stem_depth is None:
        _env_sd0 = _os_sd.environ.get("EXT_STEM_DEPTH")
        stem_depth = int(_env_sd0) if _env_sd0 not in (None, "") else None
    if stem_depth is not None:
        _sd_depth = [0] * len(tids)
        for _i in range(len(tids)):
            _dep, _nn = 0, pids[_i]
            while _nn >= 0:
                _dep += 1; _nn = pids[_nn]
            _sd_depth[_i] = _dep
        _sd_keep = [_i for _i in range(len(tids)) if _sd_depth[_i] < stem_depth]
        _sd_o2n = {_o: _k for _k, _o in enumerate(_sd_keep)}
        tids = [tids[_i] for _i in _sd_keep]
        pids = [(_sd_o2n.get(pids[_i], -1) if pids[_i] >= 0 else -1)
                for _i in _sd_keep]
        n = len(tids)

    # Build extended tree
    ext_tids = list(tids)
    ext_pids = list(pids)
    # Per-node raw draft cumulative path probability (parallel to ext_tids).
    # Backbone init from EAGLE3 capture's path_draft_p_t; suffix grafts
    # are appended in the graft loops below. None when missing.
    _e3 = rec.get("per_proposer", {}).get("eagle3", {}) or {}
    _path_draft = _e3.get("path_draft_p_t") or []
    ext_path_probs = [
        float(_path_draft[i]) if (i < len(_path_draft)
                                  and _path_draft[i] is not None)
        else None
        for i in range(len(tids))
    ]
    # Per-node anchor id: -2 = backbone (not a graft), -1 = vroot suffix,
    # >=0 = backbone-anchored suffix. Backbone nodes get -2.
    ext_anchor_ids = [-2] * len(tids)
    # Suffix-specific raw features (None for backbone):
    #   ext_suffix_freq[i]      — Arctic count for the suffix node
    #   ext_suffix_cum_prob[i]  — cumulative prob within graft (raw draft.probs[j])
    #   ext_match_len[i]        — match_len of the speculate() call producing node i
    ext_suffix_freq = [None] * len(tids)
    ext_suffix_cum_prob = [None] * len(tids)
    ext_match_len = [None] * len(tids)
    # Per-node global reach probability (= cumulative path prob from root)
    # Only populated when topb_unified_rank is set; used for top-B selection.
    node_ranks: list = [] if topb_unified_rank else None

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

    # Initial rank for backbone base nodes = their path_p_t.
    if node_ranks is not None:
        if path_p_t is not None:
            node_ranks.extend(path_p_t[:n])
        else:
            node_ranks.extend([1.0] * n)

    # === BNS bookkeeping ===
    # For Branch Nucleus Selection we track each tree node's per-edge prob
    # (unadjusted; λ_suffix is applied later when computing adjusted scores)
    # and a flag marking whether the incoming edge is suffix-sourced. Both
    # arrays grow in lockstep with ext_tids/ext_pids.
    # `dedup_extra[i]` records the largest raw suffix-edge prob that was
    # merged into node i via the dedup branch (when a graft proposes a
    # (parent, token) pair that already exists). 0.0 when no dedup happened.
    # The BNS effective prob is max(λ-adjusted edge_prob, λ * dedup_extra),
    # which corresponds to "max(eagle3_prob, λ * suffix_prob)" at deduped sites.
    _bns_track = bns_enabled or dns_enabled or dnsv2_enabled or topk_depth_enabled or descrank_enabled
    if _bns_track:
        edge_prob: list = []
        is_suffix: list = []
        dedup_extra: list = []
        # Per-node suffix match_len (0 for backbone); a feature for `calib`.
        match_len_arr: list = []
        for _i in range(n):
            edge_prob.append(node_p_t[_i] if node_p_t is not None else 1.0)
            is_suffix.append(False)
            dedup_extra.append(0.0)
            match_len_arr.append(0)
    else:
        edge_prob = None
        is_suffix = None
        dedup_extra = None
        match_len_arr = None

    allowed_nodes = None

    # === Backbone pruning (extension_pt_prune) ===
    # Remove base nodes (and their subtrees) where path_p_t^α < threshold.
    # This rebuilds tids/pids/n with only surviving nodes — graft attaches
    # later will operate on the smaller base tree.
    if backbone_pt_threshold is not None and path_p_t is not None and n > 1:
        # Mark pruned nodes (failing threshold OR parent pruned).
        # Iterate in topological order (BFS = pids[i] < i for i > 0).
        pruned = set()
        for i in range(n):
            if i == 0:
                # Root: never prune (always keep root in tree).
                if (path_p_t[i] ** backbone_pt_alpha) < backbone_pt_threshold:
                    pruned.add(i)
                continue
            parent_idx = pids[i]
            if parent_idx in pruned:
                pruned.add(i); continue
            if (path_p_t[i] ** backbone_pt_alpha) < backbone_pt_threshold:
                pruned.add(i)
        # Always preserve root (index 0)
        pruned.discard(0)
        if pruned:
            # Rebuild tree with only surviving nodes (re-index).
            old_to_new = {}
            new_tids, new_pids, new_path_p_t, new_node_p_t = [], [], [], []
            for old_i in range(n):
                if old_i in pruned: continue
                new_i = len(new_tids)
                old_to_new[old_i] = new_i
                new_tids.append(tids[old_i])
                new_parent = pids[old_i]
                new_pids.append(old_to_new[new_parent] if new_parent in old_to_new else -1)
                new_path_p_t.append(path_p_t[old_i])
                new_node_p_t.append(node_p_t[old_i] if node_p_t is not None else 1.0)
            tids = new_tids; pids = new_pids
            n = len(tids); ext_tids = list(tids); ext_pids = list(pids)
            path_p_t = new_path_p_t
            node_p_t = new_node_p_t if node_p_t is not None else None
            # Recompute paths[] for new indices.
            paths = [None] * n
            for i in range(n):
                path = []; node = i
                while node >= 0:
                    path.append(tids[node]); node = pids[node]
                path.reverse(); paths[i] = path
            # Rebuild children index.
            children = {}
            for i in range(len(ext_tids)):
                children.setdefault(ext_pids[i], {})[ext_tids[i]] = i

    # === Top-1 chain anchor selection (extension_top1) ===
    # Restrict graft attachment to root + the top-1 chain (root → highest
    # node_p_t child → ... → depth s). Base tree itself stays full.
    if top1_chain and node_p_t is not None and n > 1:
        children_of_idx = {}
        for i in range(n):
            children_of_idx.setdefault(pids[i], []).append(i)
        chain = set()
        current = -1
        while True:
            kids = children_of_idx.get(current, [])
            if not kids: break
            best = max(kids, key=lambda j: node_p_t[j])
            chain.add(best); current = best
        allowed_nodes = chain  # used by per-node graft loop below

    # Trie-invariant children index: maps parent_idx → {token_id: child_idx}.
    # Populated with the base tree first; suffix extensions then merge
    # into this structure so that a (parent, token) pair never occurs
    # twice in the extended tree (deduplicates base/suffix overlap).
    children = {}
    for i in range(len(ext_tids)):
        p = ext_pids[i]
        tok = ext_tids[i]
        children.setdefault(p, {})[tok] = i

    # EXT_ONLY_DEPTH=d (analysis knob): graft the suffix at a SINGLE anchor
    # depth only. d=0 → keep only the virtual-root graft (root hybrid). d>=1 →
    # keep only grafts anchored at backbone nodes at tree-depth d-1 (i.e. after
    # exactly d accepted backbone tokens). Default (unset) = graft everywhere.
    import os as _os_ed
    if only_depth is not None:
        _only_depth = only_depth
    else:
        _env_od = _os_ed.environ.get("EXT_ONLY_DEPTH")
        _only_depth = int(_env_od) if _env_od not in (None, "") else None
    # Cumulative variant: graft at ALL anchor depths 0..cum_depth (vroot always
    # kept; per-node grafts up to tree-depth cum_depth-1). cum_depth=8 ≈ full
    # extension. Mutually exclusive with only_depth.
    if cum_depth is not None:
        _cum_depth = cum_depth
    else:
        _env_cd = _os_ed.environ.get("EXT_CUM_DEPTH")
        _cum_depth = int(_env_cd) if _env_cd not in (None, "") else None

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
    # match_len gate: skip the root-level graft if the suffix tree didn't
    # match deep enough into context. match_len is the length of the context
    # suffix that matched a known pattern; longer = more confidence.
    if (_root_draft is not None and match_len_threshold > 0
            and int(getattr(_root_draft, "match_len", 0)) < match_len_threshold):
        _root_draft = None
    if _only_depth is not None and _only_depth != 0:
        _root_draft = None          # EXT_ONLY_DEPTH: vroot graft only for d==0
    if _root_draft is not None and _root_draft.token_ids:
        _root_probs = list(_root_draft.probs) if hasattr(_root_draft, 'probs') and _root_draft.probs else None
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
                # Dedup: another proposer already created this (parent, token).
                # Capture incoming graft's raw edge prob so BNS can combine
                # via max(existing-effective, λ × incoming) downstream.
                if _bns_track:
                    if _pid == -1:
                        _ep_in = float(_root_probs[_j]) if (_root_probs and _j < len(_root_probs)) else 0.0
                    else:
                        _pc = float(_root_probs[_pid]) if (_root_probs and _pid < len(_root_probs)) else 0.0
                        _cc = float(_root_probs[_j]) if (_root_probs and _j < len(_root_probs)) else 0.0
                        _ep_in = (_cc / _pc) if _pc > 1e-12 else 0.0
                    if _ep_in > dedup_extra[_existing]:
                        dedup_extra[_existing] = _ep_in
                _root_local[_j] = _existing
                continue
            if max_count is not None and len(ext_tids) >= max_count:
                break
            _new_idx = len(ext_tids)
            ext_tids.append(_tid)
            ext_pids.append(_tparent)
            # Vroot suffix: anchor = -1, path_prob = root_probs[j] (graft is
            # rooted at virtual root with anchor_pp=1.0).
            ext_anchor_ids.append(-1)
            _vr_pp = (float(_root_probs[_j])
                      if (_root_probs and _j < len(_root_probs))
                      else None)
            ext_path_probs.append(_vr_pp)
            # Suffix-specific raw features for vroot graft.
            _root_counts = (list(_root_draft.counts)
                            if hasattr(_root_draft, 'counts')
                               and _root_draft.counts else None)
            ext_suffix_freq.append(
                int(_root_counts[_j])
                if (_root_counts and _j < len(_root_counts))
                else None)
            ext_suffix_cum_prob.append(_vr_pp)
            ext_match_len.append(
                int(getattr(_root_draft, "match_len", 0)))
            children.setdefault(_tparent, {})[_tid] = _new_idx
            _root_local[_j] = _new_idx
            # Track rank: anchor=virtual root → path=1.0, multiply by graft cumulative.
            if node_ranks is not None:
                _gp = float(_root_probs[_j]) if (_root_probs and _j < len(_root_probs)) else 0.0
                node_ranks.append(_gp)
            # BNS: per-edge prob from this graft's local parent. probs[] is
            # cumulative within the graft, so per-edge = cum[j]/cum[parent_j]
            # (or just cum[j] when parent_j == -1 i.e. this is the graft root).
            if _bns_track:
                if _pid == -1:
                    _ep = float(_root_probs[_j]) if (_root_probs and _j < len(_root_probs)) else 0.0
                else:
                    _pc = float(_root_probs[_pid]) if (_root_probs and _pid < len(_root_probs)) else 0.0
                    _cc = float(_root_probs[_j]) if (_root_probs and _j < len(_root_probs)) else 0.0
                    _ep = (_cc / _pc) if _pc > 1e-12 else 0.0
                edge_prob.append(_ep)
                is_suffix.append(True)
                dedup_extra.append(0.0)
                match_len_arr.append(int(getattr(_root_draft, "match_len", 0)))

    for node_idx in range(n):
        if max_count is not None and len(ext_tids) >= max_count:
            break  # hit the overall tree-size cap before iterating this node

        if allowed_nodes is not None and node_idx not in allowed_nodes:
            continue  # ptopk filter

        if _only_depth is not None:
            # EXT_ONLY_DEPTH: graft only at backbone nodes after exactly
            # _only_depth accepted tokens (tree-depth == _only_depth-1). d==0
            # keeps no per-node grafts (vroot graft only, handled above).
            if _only_depth == 0 or (len(paths[node_idx]) - 1) != _only_depth - 1:
                continue
        if _cum_depth is not None:
            # EXT_CUM_DEPTH: cumulative — graft at every anchor depth up to
            # _cum_depth (tree-depth 0.._cum_depth-1). vroot (depth 0) kept
            # above; _cum_depth==0 → vroot only (= root hybrid).
            if _cum_depth == 0 or (len(paths[node_idx]) - 1) > _cum_depth - 1:
                continue

        ext_context = np.array(base_context + paths[node_idx], dtype=np.int32)

        # AB-TEST KNOB: set BENCH_NO_TEMP_EXT=1 to skip the temporary_extension
        # wrapper (mimics pre-pop behavior — speculate against tree state
        # that has NOT seen paths[node_idx]). Default behavior uses temp.
        import os as _os
        _no_temp = _os.environ.get("BENCH_NO_TEMP_EXT") == "1"
        try:
            _per_kwargs = dict(max_spec_factor=suffix_max_spec_factor,
                               min_token_prob=suffix_min_token_prob,
                               use_tree_spec=True)
            if suffix_max_spec_tokens > 0:
                _per_kwargs["max_spec_tokens"] = suffix_max_spec_tokens
            if _no_temp:
                draft = suffix_cache.speculate(
                    cache_req_id, ext_context,
                    **_per_kwargs,
                )
            else:
                with suffix_cache.temporary_extension(
                        cache_req_id, paths[node_idx]):
                    draft = suffix_cache.speculate(
                        cache_req_id, ext_context,
                        **_per_kwargs,
                    )
        except Exception:
            continue

        if not draft.token_ids:
            continue
        # match_len gate (same as root): skip grafting at anchors where the
        # suffix-tree match was shallower than match_len_threshold tokens.
        if match_len_threshold > 0:
            _ml = int(getattr(draft, "match_len", 0))
            if _ml < match_len_threshold:
                continue
        draft_score = float(getattr(draft, "score", 0.0))
        if score_threshold is not None and draft_score < score_threshold:
            continue
        if pathprob_threshold is not None and path_p_t is not None:
            if draft_score * path_p_t[node_idx] < pathprob_threshold:
                continue
        if pt_threshold is not None and path_p_t is not None:
            # EAGLE3 path_p_t with optional alpha exponent. pt_alpha=1.0 (default)
            # reduces to "path_p_t[node] < pt_threshold" comparison; alpha != 1
            # adjusts the depth penalty: alpha<1 softens (deeper still pass),
            # alpha>1 amplifies (deep nodes drop faster).
            if (path_p_t[node_idx] ** pt_alpha) < pt_threshold:
                continue
        if product_threshold is not None and path_p_t is not None:
            # Multiplicative joint: keep if draft.score * path_p_t[node]**alpha
            # >= product_threshold. Combines suffix quality (score) with
            # backbone reach probability into a single signal — one strong axis
            # can compensate for the other (unlike AND, which requires both).
            if draft_score * (path_p_t[node_idx] ** product_alpha) < product_threshold:
                continue

        # Attach suffix chain with dedup. Each draft token is checked
        # against the current children[tree_parent] map; if the same
        # token already exists under that parent (backbone or previously-
        # merged suffix), reuse it — otherwise append. local_to_tree
        # threads parent-index resolution for multi-token chains.
        # Assumes draft.parents is topologically ordered (parent idx <
        # child idx) — sglang's SuffixDecodingCache returns BFS.
        local_to_tree = {}
        _graft_probs = list(draft.probs) if hasattr(draft, 'probs') and draft.probs else None
        _anchor_path = path_p_t[node_idx] if path_p_t is not None else 1.0
        for j, (tid, pid) in enumerate(zip(draft.token_ids, draft.parents)):
            if pid == -1:
                tree_parent = node_idx
            else:
                tree_parent = local_to_tree.get(pid)
                if tree_parent is None:
                    break  # malformed draft — abort this chain
            existing = children.get(tree_parent, {}).get(tid)
            if existing is not None:
                # Dedup boost: capture incoming graft's raw edge prob.
                if _bns_track:
                    if pid == -1:
                        _ep_in = float(_graft_probs[j]) if (_graft_probs and j < len(_graft_probs)) else 0.0
                    else:
                        _pc = float(_graft_probs[pid]) if (_graft_probs and pid < len(_graft_probs)) else 0.0
                        _cc = float(_graft_probs[j]) if (_graft_probs and j < len(_graft_probs)) else 0.0
                        _ep_in = (_cc / _pc) if _pc > 1e-12 else 0.0
                    if _ep_in > dedup_extra[existing]:
                        dedup_extra[existing] = _ep_in
                local_to_tree[j] = existing  # merge into existing node
                continue
            if max_count is not None and len(ext_tids) >= max_count:
                break  # cap reached — stop adding new nodes
            new_idx = len(ext_tids)
            ext_tids.append(tid)
            ext_pids.append(tree_parent)
            # Per-anchor suffix: anchor = node_idx (backbone index),
            # path_prob = anchor_path_prob × cumulative_within_graft.
            ext_anchor_ids.append(node_idx)
            _anchor_pp_self = (ext_path_probs[node_idx]
                               if node_idx < len(ext_path_probs)
                               else None)
            _gp_cum = (float(_graft_probs[j])
                       if (_graft_probs and j < len(_graft_probs))
                       else None)
            if _anchor_pp_self is not None and _gp_cum is not None:
                ext_path_probs.append(float(_anchor_pp_self) * _gp_cum)
            else:
                ext_path_probs.append(None)
            # Suffix raw features for per-anchor graft.
            _graft_counts = (list(draft.counts)
                             if hasattr(draft, 'counts') and draft.counts
                             else None)
            ext_suffix_freq.append(
                int(_graft_counts[j])
                if (_graft_counts and j < len(_graft_counts))
                else None)
            ext_suffix_cum_prob.append(_gp_cum)
            ext_match_len.append(
                int(getattr(draft, "match_len", 0)))
            children.setdefault(tree_parent, {})[tid] = new_idx
            local_to_tree[j] = new_idx
            # Track rank: anchor's backbone path × (graft cumulative prob)^alpha.
            # topb_alpha_ext exponent calibrates count-based suffix probs against
            # softmax-based backbone probs (default 1.0 = no calibration).
            if node_ranks is not None:
                _gp = float(_graft_probs[j]) if (_graft_probs and j < len(_graft_probs)) else 0.0
                if topb_alpha_ext != 1.0 and _gp > 0:
                    _gp = _gp ** topb_alpha_ext
                node_ranks.append(_anchor_path * _gp)
            # BNS: per-edge prob within graft (cum[j]/cum[pid], or cum[j] when
            # pid==-1 so this is the graft root attached at anchor).
            if _bns_track:
                if pid == -1:
                    _ep = float(_graft_probs[j]) if (_graft_probs and j < len(_graft_probs)) else 0.0
                else:
                    _pc = float(_graft_probs[pid]) if (_graft_probs and pid < len(_graft_probs)) else 0.0
                    _cc = float(_graft_probs[j]) if (_graft_probs and j < len(_graft_probs)) else 0.0
                    _ep = (_cc / _pc) if _pc > 1e-12 else 0.0
                edge_prob.append(_ep)
                is_suffix.append(True)
                dedup_extra.append(0.0)
                match_len_arr.append(int(getattr(draft, "match_len", 0)))

    # === SUFFIX COUNT SHRINKAGE (Jeffreys) ===
    # Suffix edge probs are trie count ratios c/n — high-variance at small
    # counts (a 2/2 edge reads 1.0, same as a 500/500 edge). When the active
    # calibrator was fitted on shrunk probs (wants_shrunk_edge_probs), replace
    # each suffix edge prob with the Jeffreys posterior mean (c+0.5)/(n+1),
    # recovering n from n = c/p. Backbone (softmax) probs and λ-based methods
    # are untouched. Must run BEFORE the calib block below so predictions and
    # labels live in the shrunk feature space.
    if (calib is not None
            and getattr(calib, "wants_shrunk_edge_probs", False)
            and edge_prob is not None):
        for _i in range(len(ext_tids)):
            if not is_suffix[_i] or _i >= len(ext_suffix_freq):
                continue
            _c = ext_suffix_freq[_i]
            _p = edge_prob[_i]
            if _c is None or _c <= 0 or _p <= 0.0:
                continue
            _nn = max(float(_c) / _p, float(_c))
            edge_prob[_i] = (float(_c) + 0.5) / (_nn + 1.0)

    # === ONLINE ACCEPT-RATE CALIBRATION (extension_calib_*) ===
    # When `calib` is supplied, replace the λ-scaled `_eff` used by the
    # nucleus/top-k selectors with a per-edge calibrated conditional accept
    # rate. Features are computed on the FULL pre-prune tree; the per-edge
    # accept LABELS are read from the full-tree greedy path and fed back into
    # the running OLS so later steps use a fitted model (this step is scored
    # with the model fit on PRIOR steps → causal, no train/test leakage).
    _eff_override = None
    if calib is not None and edge_prob is not None and len(ext_tids) > 1:
        _N = len(ext_tids)
        # Structural features. BFS order ⇒ ext_pids[i] < i for i>0, so depth
        # is a forward pass and n_descendants a reverse pass.
        _c_depth = [0] * _N
        for _i in range(_N):
            _p = ext_pids[_i]
            _c_depth[_i] = 1 if _p < 0 else _c_depth[_p] + 1
        _c_ndesc = [0] * _N
        for _i in range(_N - 1, -1, -1):
            _p = ext_pids[_i]
            if _p >= 0:
                _c_ndesc[_p] += 1 + _c_ndesc[_i]
        _c_kcount: dict = {}
        for _i in range(_N):
            _c_kcount[ext_pids[_i]] = _c_kcount.get(ext_pids[_i], 0) + 1
        # 1) Predict calibrated edge value (uses β fit on PRIOR steps).
        _c_feat = [None] * _N
        _eff_override = [0.0] * _N
        for _i in range(_N):
            _ml = match_len_arr[_i] if match_len_arr is not None else 0
            _nsib = _c_kcount.get(ext_pids[_i], 1)
            _x = _calib_features(edge_prob[_i], _c_depth[_i], _c_ndesc[_i], _nsib, _ml)
            _c_feat[_i] = _x
            _grp = 'suffix' if is_suffix[_i] else 'eagle'
            _v = calib.predict(_grp, _x, edge_prob[_i])
            # Dedup site (both proposers reached this node): take the stronger
            # of the eagle prediction and the suffix prediction on the merged
            # suffix edge — mirrors the λ formula's max(eagle, λ·suffix).
            if dedup_extra[_i] > 0.0:
                _xd = _calib_features(dedup_extra[_i], _c_depth[_i], _c_ndesc[_i], _nsib, _ml)
                _vd = calib.predict('suffix', _xd, dedup_extra[_i])
                if _vd > _v:
                    _v = _vd
            _eff_override[_i] = _v
        # 2) Conditional-accept labels on the FULL tree → update running OLS.
        # An edge (parent→child) is evaluable iff its parent was reached
        # (virtual root, or a node on the greedy accept path); label = whether
        # the child continues that accept path.
        _apath = set(greedy_tree_walk_path(ext_tids, ext_pids, gt))
        for _i in range(_N):
            _p = ext_pids[_i]
            if _p < 0 or _p in _apath:
                _y = 1.0 if _i in _apath else 0.0
                calib.update('suffix' if is_suffix[_i] else 'eagle',
                             _c_feat[_i], _y)

    # === BRANCH NUCLEUS SELECTION (BNS) ===
    # Recursive top-down sibling pruning. At each parent we compute each
    # child's subtree contribution R(c) = Σ_{z∈T_c} PathScore(z), normalize
    # among siblings, sort desc, and keep the smallest prefix whose
    # cumulative normalized contribution reaches bns_rho. Pruned children
    # (and their entire subtrees) are discarded. No budget cap.
    if bns_enabled and edge_prob is not None and len(ext_tids) > 1:
        # Adjusted edge prob: suffix edges multiplied by bns_lambda_suffix.
        # When a node was created by both a base proposer AND a deduped suffix
        # graft (dedup_extra[i] > 0), take max(base λ-adjusted, λ × dedup_extra).
        # This applies "max(eagle3_prob, λ × suffix_prob)" at deduped sites.
        if _eff_override is not None:
            _adj = list(_eff_override)
        else:
            _lam = bns_lambda_suffix
            _adj = [
                max(_lam * ep if isfx else ep, _lam * dx)
                for ep, isfx, dx in zip(edge_prob, is_suffix, dedup_extra)
            ]
        # PathScore(z) = product of adjusted edges along root→z. BFS order
        # of ext_pids guarantees ext_pids[i] < i for i>0 so a single
        # forward pass suffices.
        _ps = [0.0] * len(ext_tids)
        for i in range(len(ext_tids)):
            p = ext_pids[i]
            if p == -1:
                _ps[i] = _adj[i]
            else:
                _ps[i] = _ps[p] * _adj[i]
        # R[i] = Σ PathScore over subtree(i). Reverse-BFS post-order roll-up.
        _children_of: dict = {}
        for i, p in enumerate(ext_pids):
            _children_of.setdefault(p, []).append(i)
        _R = list(_ps)
        for i in range(len(ext_tids) - 1, -1, -1):
            p = ext_pids[i]
            if p >= 0:
                _R[p] += _R[i]
        # === BNS calibration trace (env-gated, pre-prune) ===
        # For each parent on the greedy accept path (incl. virtual root -1),
        # dump (norm_R, accepted, k_siblings, parent_depth, subtree_mat)
        # for every child.
        #   accepted        = 1 iff this child is on the greedy accept path of
        #                     the full pre-prune tree (binary).
        #   subtree_mat     = number of accept-path tokens contributed BY this
        #                     branch — i.e. (len(accept_path) - parent_depth)
        #                     if c is on the path (since accept-path is
        #                     contiguous from root → c → deeper); 0 otherwise.
        #                     This is the "subtree value" R(c) actually wants
        #                     to predict, beyond binary "this node accepts".
        if _BNS_CALIB['enabled'] and len(_BNS_CALIB['records']) < _BNS_CALIB['max_records']:
            _accept_path = greedy_tree_walk_path(ext_tids, ext_pids, gt)
            _on_path = set(_accept_path)
            _ap_len = len(_accept_path)
            # virtual root + each accepted node is a "reached parent"
            for _depth, _parent in enumerate([-1] + _accept_path):
                _kids = _children_of.get(_parent, [])
                if not _kids:
                    continue
                _total_R = 0.0
                for _c in _kids:
                    _total_R += _R[_c]
                if _total_R <= 0:
                    continue
                # children of parent at parent_depth=_depth live at depth _depth+1.
                # subtree_mat = # path tokens from this child onwards (inclusive)
                #             = _ap_len - _depth  (if c on path)
                _subtree_mat_if_on = _ap_len - _depth
                for _c in _kids:
                    _norm = _R[_c] / _total_R
                    _acc = 1 if _c in _on_path else 0
                    _smat = _subtree_mat_if_on if _acc else 0
                    _BNS_CALIB['records'].append([
                        round(_norm, 6), _acc, len(_kids), _depth, _smat,
                    ])
                    if len(_BNS_CALIB['records']) >= _BNS_CALIB['max_records']:
                        break
                if len(_BNS_CALIB['records']) >= _BNS_CALIB['max_records']:
                    break

        # Recursive BNS starting from virtual root (-1). Tree depth is
        # small (≤ ~30) so recursion is safe.
        _keep_set: set = set()
        def _bns_pick(parent_idx):
            kids = _children_of.get(parent_idx, [])
            if not kids:
                return
            total = 0.0
            for c in kids:
                total += _R[c]
            if total <= 0:
                # No signal — keep every child (degenerate; rare).
                for c in kids:
                    _keep_set.add(c)
                    _bns_pick(c)
                return
            kids_sorted = sorted(kids, key=lambda c: -_R[c])
            cumsum = 0.0
            for c in kids_sorted:
                _keep_set.add(c)
                _bns_pick(c)
                cumsum += _R[c] / total
                if cumsum >= bns_rho:
                    break
        _bns_pick(-1)
        # === BNS branching/score-distribution trace (env-gated) ===
        if _BNS_TRACE['enabled']:
            _agg = _BNS_TRACE['agg']
            for _parent, _kids in _children_of.items():
                if not _kids: continue
                _total_R = sum(_R[c] for c in _kids)
                if _total_R <= 0: continue
                _norm_R = sorted([_R[c]/_total_R for c in _kids], reverse=True)
                _kept_count = sum(1 for c in _kids if c in _keep_set)
                _agg['parent_count'] += 1
                _agg['total_orig_kids'] += len(_kids)
                _agg['total_kept_kids'] += _kept_count
                _agg['top_R_sum'] += _norm_R[0]
                _agg['orig_kids_hist'][len(_kids)] = _agg['orig_kids_hist'].get(len(_kids), 0) + 1
                _agg['kept_kids_hist'][_kept_count] = _agg['kept_kids_hist'].get(_kept_count, 0) + 1
                if len(_BNS_TRACE['records']) < _BNS_TRACE['max_records']:
                    _BNS_TRACE['records'].append({
                        'p': int(_parent),
                        'k': len(_kids),
                        'kp': _kept_count,
                        'r': [round(r, 4) for r in _norm_R[:10]],
                    })
        # Rebuild tree from kept indices. base_kept goes first so the
        # downstream "n = base count" invariant holds.
        ORIG_N = n
        base_kept = sorted(i for i in _keep_set if i < ORIG_N)
        graft_kept = sorted(i for i in _keep_set if i >= ORIG_N)
        emit_order = base_kept + graft_kept
        old_to_new = {old: new for new, old in enumerate(emit_order)}
        new_tids = [ext_tids[i] for i in emit_order]
        new_pids = [
            (old_to_new[ext_pids[i]] if ext_pids[i] >= 0 and ext_pids[i] in old_to_new else -1)
            for i in emit_order
        ]
        ext_tids = new_tids; ext_pids = new_pids
        n = len(base_kept)

    # === DNS: DEPTH-WISE NUCLEUS SELECTION (extension_dns) ===
    # Formerly known as BNSv2. Differences from BNSv1:
    #   (A) Dedup-boost: already incorporated via dedup_extra[] regardless of
    #       variant — no extra work here.
    #   (B) Score = path-prob (cumulative product down to the node) instead of
    #       R(c) = subtree-summed PathScore. Removes the "big subtree = high
    #       score" bias.
    #   (C) Nucleus selection is per-DEPTH, not per-parent. At each depth d,
    #       all candidates (= depth-d nodes with kept parent) compete in a
    #       single pool; we keep the smallest top-by-path-prob prefix whose
    #       cumulative normalized share reaches bns_rho.
    # Top-down iteration keeps the kept set closed under ancestor by
    # construction (a child is only a candidate if its parent is in keep_set).
    if dns_enabled and edge_prob is not None and len(ext_tids) > 1:
        if _eff_override is not None:
            _eff = list(_eff_override)
        else:
            _lam = bns_lambda_suffix
            # 1. Effective per-node prob (dedup-boost incorporated, λ-scaled).
            _eff = [
                max(_lam * ep if isfx else ep, _lam * dx)
                for ep, isfx, dx in zip(edge_prob, is_suffix, dedup_extra)
            ]
        # 2. Cumulative path prob (root → node). No subtree rollup.
        _ps = [0.0] * len(ext_tids)
        for i in range(len(ext_tids)):
            p = ext_pids[i]
            _ps[i] = _eff[i] if p == -1 else _ps[p] * _eff[i]
        # 3. Per-node depth (BFS order of ext_pids gives p < i so single pass).
        _dep = [0] * len(ext_tids)
        for i in range(len(ext_tids)):
            p = ext_pids[i]
            _dep[i] = 1 if p == -1 else _dep[p] + 1
        _max_d = max(_dep) if _dep else 0
        # Group node indices by depth.
        _by_depth: dict = {}
        for i, d in enumerate(_dep):
            _by_depth.setdefault(d, []).append(i)
        # 4. Top-down per-depth nucleus.
        _keep_set: set = set()
        for d in range(1, _max_d + 1):
            _cands = [i for i in _by_depth.get(d, [])
                      if ext_pids[i] == -1 or ext_pids[i] in _keep_set]
            if not _cands:
                break  # no candidates at depth d → no descendants either
            _total = 0.0
            for c in _cands:
                _total += _ps[c]
            if _total <= 0:
                # All-zero pool (degenerate). Keep everything — rare.
                for c in _cands:
                    _keep_set.add(c)
                continue
            _cands.sort(key=lambda c: -_ps[c])
            _cumsum = 0.0
            for c in _cands:
                _keep_set.add(c)
                _cumsum += _ps[c] / _total
                if _cumsum >= bns_rho:
                    break
        # 5. Rebuild tree from kept indices (same pattern as BNSv1).
        ORIG_N = n
        base_kept = sorted(i for i in _keep_set if i < ORIG_N)
        graft_kept = sorted(i for i in _keep_set if i >= ORIG_N)
        emit_order = base_kept + graft_kept
        old_to_new = {old: new for new, old in enumerate(emit_order)}
        new_tids = [ext_tids[i] for i in emit_order]
        new_pids = [
            (old_to_new[ext_pids[i]] if ext_pids[i] >= 0 and ext_pids[i] in old_to_new else -1)
            for i in emit_order
        ]
        ext_tids = new_tids; ext_pids = new_pids
        n = len(base_kept)

    # === DNSv2: DEPTH-WISE NUCLEUS, NO NORMALIZE (extension_dnsv2) ===
    # Same as DNS, but the cumulative sum is over RAW path-probs (not
    # normalized by the candidate pool's total mass). The threshold bns_rho
    # is interpreted as an absolute path-prob mass.
    # Because path_prob shrinks with depth, the same ρ at deeper levels
    # implicitly keeps more nodes (often all): at depth d=5, total mass
    # might be 1e-3 while ρ=0.1 → cumsum never reaches ρ → all kept.
    # Conversely, at depth d=1, ρ=0.1 means "keep until absolute cumulative
    # path-prob reaches 0.1" — generally aggressive.
    if dnsv2_enabled and edge_prob is not None and len(ext_tids) > 1:
        if _eff_override is not None:
            _eff = list(_eff_override)
        else:
            _lam = bns_lambda_suffix
            _eff = [
                max(_lam * ep if isfx else ep, _lam * dx)
                for ep, isfx, dx in zip(edge_prob, is_suffix, dedup_extra)
            ]
        _ps = [0.0] * len(ext_tids)
        for i in range(len(ext_tids)):
            p = ext_pids[i]
            _ps[i] = _eff[i] if p == -1 else _ps[p] * _eff[i]
        _dep = [0] * len(ext_tids)
        for i in range(len(ext_tids)):
            p = ext_pids[i]
            _dep[i] = 1 if p == -1 else _dep[p] + 1
        _max_d = max(_dep) if _dep else 0
        _by_depth: dict = {}
        for i, d in enumerate(_dep):
            _by_depth.setdefault(d, []).append(i)
        _keep_set: set = set()
        for d in range(1, _max_d + 1):
            _cands = [i for i in _by_depth.get(d, [])
                      if ext_pids[i] == -1 or ext_pids[i] in _keep_set]
            if not _cands:
                break
            _cands.sort(key=lambda c: -_ps[c])
            _cumsum = 0.0
            for c in _cands:
                _keep_set.add(c)
                _cumsum += _ps[c]  # RAW path-prob (no /_total)
                if _cumsum >= bns_rho:
                    break
        # Rebuild tree (identical to DNS rebuild).
        ORIG_N = n
        base_kept = sorted(i for i in _keep_set if i < ORIG_N)
        graft_kept = sorted(i for i in _keep_set if i >= ORIG_N)
        emit_order = base_kept + graft_kept
        old_to_new = {old: new for new, old in enumerate(emit_order)}
        new_tids = [ext_tids[i] for i in emit_order]
        new_pids = [
            (old_to_new[ext_pids[i]] if ext_pids[i] >= 0 and ext_pids[i] in old_to_new else -1)
            for i in emit_order
        ]
        ext_tids = new_tids; ext_pids = new_pids
        n = len(base_kept)

    # === TOP-K DEPTH SELECTION (extension_topk) ===
    # Per-depth top-k by path-prob. Effective per-node prob (dedup-boost
    # + λ-scaled suffix) and path-prob scoring identical to BNSv2 — the
    # only difference is the selection rule: "keep first k by rank"
    # instead of "cumulative ≥ ρ nucleus". Tree size bounded per depth
    # at k, so total nodes ≤ k × max_depth.
    if topk_depth_enabled and edge_prob is not None and len(ext_tids) > 1:
        if _eff_override is not None:
            _eff = list(_eff_override)
        else:
            _lam = bns_lambda_suffix
            # 1. Effective per-node prob (dedup-boost incorporated, λ-scaled).
            _eff = [
                max(_lam * ep if isfx else ep, _lam * dx)
                for ep, isfx, dx in zip(edge_prob, is_suffix, dedup_extra)
            ]
        # 2. Cumulative path prob (no subtree rollup — same as DNS).
        _ps = [0.0] * len(ext_tids)
        for i in range(len(ext_tids)):
            p = ext_pids[i]
            _ps[i] = _eff[i] if p == -1 else _ps[p] * _eff[i]
        # 3. Per-node depth.
        _dep = [0] * len(ext_tids)
        for i in range(len(ext_tids)):
            p = ext_pids[i]
            _dep[i] = 1 if p == -1 else _dep[p] + 1
        _max_d = max(_dep) if _dep else 0
        _by_depth: dict = {}
        for i, d in enumerate(_dep):
            _by_depth.setdefault(d, []).append(i)
        # 4. Top-down per-depth TOP-K (only difference from DNS).
        _keep_set: set = set()
        for d in range(1, _max_d + 1):
            _cands = [i for i in _by_depth.get(d, [])
                      if ext_pids[i] == -1 or ext_pids[i] in _keep_set]
            if not _cands:
                break
            _cands.sort(key=lambda c: -_ps[c])
            for c in _cands[:topk_depth_k]:
                _keep_set.add(c)
        # 5. Rebuild tree (identical to DNS rebuild).
        ORIG_N = n
        base_kept = sorted(i for i in _keep_set if i < ORIG_N)
        graft_kept = sorted(i for i in _keep_set if i >= ORIG_N)
        emit_order = base_kept + graft_kept
        old_to_new = {old: new for new, old in enumerate(emit_order)}
        new_tids = [ext_tids[i] for i in emit_order]
        new_pids = [
            (old_to_new[ext_pids[i]] if ext_pids[i] >= 0 and ext_pids[i] in old_to_new else -1)
            for i in emit_order
        ]
        ext_tids = new_tids; ext_pids = new_pids
        n = len(base_kept)

    # === DESCRANK SELECTION (extension_descrank) ===
    # Per-depth top-k by composite score:
    #   score[i] = path_prob[i] × (1 + alpha · log(1 + n_descendants[i]))
    # n_descendants is the post-tree subtree size — captures whether this
    # node is a "trie hub" (where many continuations agree). Per-node data
    # shows n_descendants has 5.4× precision lift on suffix accepts vs
    # path_prob's 1.8×, so this is the strongest unused signal. alpha=0
    # degenerates exactly to topk (sanity check).
    if descrank_enabled and edge_prob is not None and len(ext_tids) > 1:
        import math
        if _eff_override is not None:
            _eff = list(_eff_override)
        else:
            _lam = bns_lambda_suffix
            # 1. Effective per-node prob (identical to topk).
            _eff = [
                max(_lam * ep if isfx else ep, _lam * dx)
                for ep, isfx, dx in zip(edge_prob, is_suffix, dedup_extra)
            ]
        # 2. Cumulative path prob (identical to topk).
        _ps = [0.0] * len(ext_tids)
        for i in range(len(ext_tids)):
            p = ext_pids[i]
            _ps[i] = _eff[i] if p == -1 else _ps[p] * _eff[i]
        # 3. Per-node depth.
        _dep = [0] * len(ext_tids)
        for i in range(len(ext_tids)):
            p = ext_pids[i]
            _dep[i] = 1 if p == -1 else _dep[p] + 1
        _max_d = max(_dep) if _dep else 0
        # 4. NEW: n_descendants via reverse-BFS post-order. BFS ordering
        # guarantees ext_pids[i] < i for i > 0, so descending i visits
        # children before parents.
        _ndesc = [0] * len(ext_tids)
        for i in range(len(ext_tids) - 1, -1, -1):
            p = ext_pids[i]
            if p >= 0:
                _ndesc[p] += 1 + _ndesc[i]
        # 5. Composite score. alpha=0 → score == path_prob → topk-equivalent.
        _score = [
            _ps[i] * (1.0 + descrank_alpha * math.log(1.0 + _ndesc[i]))
            for i in range(len(ext_tids))
        ]
        # 6. Per-depth top-k by score (closure preserved by top-down order).
        _by_depth: dict = {}
        for i, d in enumerate(_dep):
            _by_depth.setdefault(d, []).append(i)
        _keep_set: set = set()
        for d in range(1, _max_d + 1):
            _cands = [i for i in _by_depth.get(d, [])
                      if ext_pids[i] == -1 or ext_pids[i] in _keep_set]
            if not _cands:
                break
            _cands.sort(key=lambda c: -_score[c])
            for c in _cands[:topk_depth_k]:
                _keep_set.add(c)
        # 7. Rebuild tree (identical to topk rebuild).
        ORIG_N = n
        base_kept = sorted(i for i in _keep_set if i < ORIG_N)
        graft_kept = sorted(i for i in _keep_set if i >= ORIG_N)
        emit_order = base_kept + graft_kept
        old_to_new = {old: new for new, old in enumerate(emit_order)}
        new_tids = [ext_tids[i] for i in emit_order]
        new_pids = [
            (old_to_new[ext_pids[i]] if ext_pids[i] >= 0 and ext_pids[i] in old_to_new else -1)
            for i in emit_order
        ]
        ext_tids = new_tids; ext_pids = new_pids
        n = len(base_kept)

    # === DEPTH CAP ===
    # Drop nodes at depth > depth_cap. Motivated by per-node analysis on
    # bfcl_v4: ~20% of suffix budget lives at depth > 10 but contains
    # ~2% of accepted nodes (and 0% past depth 12). Capping frees budget
    # at zero accept-quality cost.
    if depth_cap > 0 and len(ext_tids) > 1:
        _dep = [0] * len(ext_tids)
        for i in range(len(ext_tids)):
            p = ext_pids[i]
            _dep[i] = 1 if p == -1 else _dep[p] + 1
        kept_idx = [i for i, d in enumerate(_dep) if d <= depth_cap]
        if len(kept_idx) < len(ext_tids):
            base_kept = [i for i in kept_idx if i < n]
            graft_kept = [i for i in kept_idx if i >= n]
            emit_order = base_kept + graft_kept
            old_to_new = {old: new for new, old in enumerate(emit_order)}
            new_tids = [ext_tids[i] for i in emit_order]
            new_pids = [
                (old_to_new[ext_pids[i]] if ext_pids[i] >= 0 and ext_pids[i] in old_to_new else -1)
                for i in emit_order
            ]
            ext_tids = new_tids; ext_pids = new_pids
            n = len(base_kept)

    # === TOP-B UNIFIED RANK SELECTION ===
    # When topb_unified_rank is set, prune the extended tree to keep only
    # the top-`budget` nodes ranked by their cumulative reach probability
    # (= path_p_t for backbone, anchor_path × graft_prob for grafts).
    # Since ranks are cumulative-product, parent rank ≥ child rank, so
    # sorting desc places ancestors before descendants → closure preserved.
    if topb_unified_rank and node_ranks is not None and len(ext_tids) > budget:
        assert len(node_ranks) == len(ext_tids), \
            f"rank/tids mismatch: {len(node_ranks)} vs {len(ext_tids)}"
        ORIG_N = n
        # Force-keep set: top-1 backbone chain (greedy argmax-child from root)
        # is preserved regardless of budget. Tests whether deep backbone chain
        # truncation by extension is hurting acceptance.
        force_keep = set()
        if topb_keep1_chain and ORIG_N > 0:
            _bb_children = {}
            for _i in range(ORIG_N):
                _bb_children.setdefault(ext_pids[_i], []).append(_i)
            _cur = -1
            while _cur in _bb_children:
                _kids = _bb_children[_cur]
                # Greedy: child with highest rank (path_p_t)
                _nxt = max(_kids, key=lambda c: node_ranks[c])
                force_keep.add(_nxt)
                _cur = _nxt
        # Sort indices by rank desc, ties broken by original index (parents
        # naturally come first in BFS within same rank).
        sorted_idx = sorted(range(len(ext_tids)),
                            key=lambda i: (-node_ranks[i], i))
        # If forcing top-1 chain, allocate budget after subtracting forced nodes
        if force_keep:
            keep_set = set(force_keep)
            remaining_budget = max(0, budget - len(force_keep))
            for i in sorted_idx:
                if len(keep_set) >= len(force_keep) + remaining_budget:
                    break
                keep_set.add(i)
        else:
            keep_set = set(sorted_idx[:budget])
        # Safety: enforce closure (ancestors of any kept node must be kept).
        # Rank monotonicity makes this no-op except in ties — small over-budget
        # acceptable.
        for i in list(keep_set):
            p = ext_pids[i]
            while p >= 0 and p not in keep_set:
                keep_set.add(p); p = ext_pids[p]
        # Emit base nodes first (preserving relative order = BFS = parent-first),
        # then graft nodes (also BFS). This preserves global parent-before-child
        # ordering AND keeps n = number of base tokens at the prefix.
        base_kept = sorted(i for i in keep_set if i < ORIG_N)
        graft_kept = sorted(i for i in keep_set if i >= ORIG_N)
        emit_order = base_kept + graft_kept
        old_to_new = {old: new for new, old in enumerate(emit_order)}
        new_tids = [ext_tids[i] for i in emit_order]
        new_pids = [
            (old_to_new[ext_pids[i]] if ext_pids[i] >= 0 and ext_pids[i] in old_to_new else -1)
            for i in emit_order
        ]
        ext_tids = new_tids; ext_pids = new_pids
        n = len(base_kept)

    # (EXT_STEM_DEPTH backbone truncation is applied EARLY, before grafting —
    # see top of function — so suffix never merges into removed deep backbone.)

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

    # Tree topology side-channel for per-step dump.
    # Backbone nodes occupy indices [0, n); suffix grafts occupy [n, len(ext_tids)).
    _extension_step._last_ext_tids = list(ext_tids)
    _extension_step._last_ext_pids = list(ext_pids)
    _extension_step._last_ext_n_base = n
    # Greedy accepted path indices (re-run lightweight walk).
    _acc_path: List[int] = []
    _node = -1
    for _t in gt:
        _picked = None
        for _c in _children.get(_node, []):
            if ext_tids[_c] == _t:
                _picked = _c
                break
        if _picked is None:
            break
        _acc_path.append(_picked)
        _node = _picked
    _extension_step._last_accepted_path = _acc_path

    # Defensive resize guard: ext_path_probs/ext_anchor_ids should always
    # equal len(ext_tids). If a code path (e.g., dedup branch that takes
    # the `continue` before our append) caused them to fall out of sync,
    # pad/truncate to maintain invariant. Better to surface as None than
    # to corrupt downstream index alignment.
    while len(ext_path_probs) < len(ext_tids):
        ext_path_probs.append(None)
    while len(ext_anchor_ids) < len(ext_tids):
        ext_anchor_ids.append(-2)
    while len(ext_suffix_freq) < len(ext_tids):
        ext_suffix_freq.append(None)
    while len(ext_suffix_cum_prob) < len(ext_tids):
        ext_suffix_cum_prob.append(None)
    while len(ext_match_len) < len(ext_tids):
        ext_match_len.append(None)
    ext_path_probs = ext_path_probs[:len(ext_tids)]
    ext_anchor_ids = ext_anchor_ids[:len(ext_tids)]
    ext_suffix_freq = ext_suffix_freq[:len(ext_tids)]
    ext_suffix_cum_prob = ext_suffix_cum_prob[:len(ext_tids)]
    ext_match_len = ext_match_len[:len(ext_tids)]
    _extension_step._last_ext_path_prob = list(ext_path_probs)
    _extension_step._last_ext_anchor_id = list(ext_anchor_ids)
    _extension_step._last_ext_suffix_freq = list(ext_suffix_freq)
    _extension_step._last_ext_suffix_cum_prob = list(ext_suffix_cum_prob)
    _extension_step._last_ext_match_len = list(ext_match_len)

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

    # Side-channel: expose tree topology + accepted path for per-step dump.
    _proposer_tree_walk._last_tids = list(tids)
    _proposer_tree_walk._last_pids = list(pids)
    # Greedy walk path for accepted_set
    _children: Dict[int, List[int]] = {}
    for _i, _p in enumerate(pids):
        _children.setdefault(_p, []).append(_i)
    _acc_path: List[int] = []
    _node = -1
    for _t in gt:
        _picked = None
        for _c in _children.get(_node, []):
            if tids[_c] == _t:
                _picked = _c
                break
        if _picked is None:
            break
        _acc_path.append(_picked)
        _node = _picked
    _proposer_tree_walk._last_accepted_path = _acc_path

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
    latency_data: dict,
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
    vanilla_ms = latency_data["vanilla_step_ms"]
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
    tfwd_by_topk = latency_data.get("target_forward_ms_by_topk", {}) or {}
    e3draft_by_ts = latency_data.get("eagle3_draft_ms_by_topk_steps", {}) or {}

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
        target_fwd = dict(latency_data.get("target_forward_ms", {}))

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
        eagle3_draft = dict(latency_data.get("eagle3_draft_ms", {}))

    legacy_verify = latency_data.get("verify_latencies_ms",
                                       latency_data.get("eagle3_step_ms", {}))

    if not target_fwd and legacy_verify:
        # Derive from legacy: target_forward ≈ vanilla, eagle3_draft = remainder
        for b_str, step in legacy_verify.items():
            target_fwd[b_str] = vanilla_ms
            eagle3_draft[b_str] = max(float(step) - vanilla_ms, 0.0)

    # Per-proposer draft costs (non-EAGLE3)
    draft_lm_tpot = float(latency_data.get("draft_lm_tpot_ms", 0.0) or 0.0)
    suffix_speculate_ms = float(
        latency_data.get("suffix_speculate_ms", 0.0) or 0.0)
    # Draft-model chain length cap. Stage 3b (collect_draft_model.py) hard-codes
    # --max-draft-tokens=16; anything above that is filled by other proposers,
    # not by more draft forwards.
    MAX_DRAFT_MODEL_N = int(latency_data.get("max_draft_model_n", 16))

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

    _fixed_draft_ms = os.environ.get("FIXED_DRAFT_MS")
    _fixed_draft_ms = float(_fixed_draft_ms) if _fixed_draft_ms is not None else None

    def _eagle3_draft(B: int) -> float:
        """EAGLE3 draft generation cost for budget B.

        FIXED_DRAFT_MS env var overrides the measured table with a constant
        (depth-independent) draft cost, modeling a "systemically solved" draft
        path (overlap with verify, distilled head, or MTP where draft folds
        into target_forward). FIXED_DRAFT_MS=0 = draft-free upper bound; =8 ≈
        canonical steps=4 cost decoupled from actual tree depth.
        """
        if _fixed_draft_ms is not None:
            return _fixed_draft_ms
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
        # Optional budget breakdown (backbone vs extension graft accept/waste).
        # Flatten the nested dict to flat columns.
        if "budget_breakdown" in sim:
            for bk, bv in sim["budget_breakdown"].items():
                entry[f"{prefix}_bd_{bk}"] = bv

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
            _spd_real = sim.get('speedup_real')
            _spd_str = (f"spd_real={_spd_real:.2f}x" if _spd_real is not None
                        else f"spd_proxy={sim['speedup']:.2f}x")
            print(f"    {method_key}: {dt:5.1f}s  mat={sim['mat']:.2f} {_spd_str}",
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
            # Replace _eagle3_draft closure with a picklable partial. Honor the
            # FIXED_DRAFT_MS override here too — otherwise per-step draft_fn
            # callers (extension_oracle / hybrid_oracle) silently read the raw
            # measured table and ignore the ablation.
            if "real_step_draft_fn" in kw:
                if _fixed_draft_ms is not None:
                    kw["real_step_draft_fn"] = partial(
                        _const_draft, val=_fixed_draft_ms)
                else:
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
            _spd_real = sim.get('speedup_real')
            _spd_str = (f"spd_real={_spd_real:.2f}x" if _spd_real is not None
                        else f"spd_proxy={sim['speedup']:.2f}x")
            print(f"    [{n_done}/{n}] {method_key}: mat={sim['mat']:.2f} {_spd_str}",
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
                    (1.0, 0.0),
                    (2.0, 0.0),
                    (4.0, 0.0),
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
        hybrid_thresholds = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]

        if "suffix" in proposers and "eagle3" in proposers:
            e3_cost = _real_cost(["eagle3"], B)
            suffix_only_cost = _target_forward(B) + suffix_speculate_ms
            # PARAMETRIC hybrid: F/T sweep × threshold sweep, N unbounded
            # Reduced 3-pair grid (T=0.0 only) for speed.
            FT_GRID = [
                (1.0, 0.0),
                (2.0, 0.0),
                (4.0, 0.0),
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
                (1.0, 0.0),
                (2.0, 0.0),
                (4.0, 0.0),
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
            # N is always unbounded. Reduced 3-pair grid (T=0.0 only) for speed.
            FT_GRID = [
                (1.0, 0.0),
                (2.0, 0.0),
                (4.0, 0.0),
            ]
            # path_draft_p_t availability — needed by prune_pt variant.
            has_draft_p_t = any(
                (rec.get("per_proposer", {})
                    .get("eagle3", {}) or {}).get("path_draft_p_t") is not None
                for rec in records)
            if not has_draft_p_t:
                print("NOTE: no path_draft_p_t available — skipping "
                      "ptopk/product/pathprob/topp/dynsfx methods",
                      file=sys.stderr)

            for F, T in FT_GRID:
                tag = f"f{F}_t{T}"
                _run(f"extension:{F}:{T}",
                     {**common, "method": f"extension:{F}:{T}",
                      "suffix_cache": _SUFFIX_ENABLED,
                      "real_step_cost_ms": ext_cost_fallback,
                      "real_step_target_fn": _target_forward,
                      "real_step_draft_only_ms": ext_draft_only},
                     f"extension_{tag}")
                # extension_gd:D — graft suffix only at anchor depth D
                # (independent single-depth-graft config; D=0 = root hybrid).
                # extension_cumd:D — cumulative graft at depths 0..D.
                for D in range(9):
                    _run(f"extension_gd:{D}:{F}:{T}",
                         {**common, "method": f"extension_gd:{D}:{F}:{T}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": ext_draft_only},
                         f"extension_gd{D}_{tag}")
                    _run(f"extension_cumd:{D}:{F}:{T}",
                         {**common, "method": f"extension_cumd:{D}:{F}:{T}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": ext_draft_only},
                         f"extension_cumd{D}_{tag}")
                    # extension_stem:D — D backbone tokens then suffix stem
                    # only (backbone truncated beyond D), over all steps.
                    _run(f"extension_stem:{D}:{F}:{T}",
                         {**common, "method": f"extension_stem:{D}:{F}:{T}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": ext_draft_only},
                         f"extension_stem{D}_{tag}")
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

                # Score filter is the only kept extension filter variant.
                for thresh in [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]:
                    _run(f"extension_by_score:{thresh}:{F}:{T}",
                         {**common, "method": f"extension_by_score:{thresh}:{F}:{T}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": ext_draft_only},
                         f"extension_by_score_t{thresh:.1f}_{tag}")
                # BOUNDARY-EDGE EXTENSION BATCH (temporary — only new combos).
                # Existing canonical already covers α∈{0.5,1,2}, pt∈{0.001,0.01,0.1,0.5}.
                # New boundary edges only:
                if has_draft_p_t:
                    pt_alpha_new = [
                        (3.0, 0.001), (5.0, 0.001),   # α UPPER
                        (2.0, 0.0001),                # pt LOWER (at best α)
                        (3.0, 0.0001), (5.0, 0.0001), # corner: both extended
                        (3.0, 0.01), (3.0, 0.1),      # α=3 × mid thresholds
                        (3.0, 0.00001),               # α=3 × very low threshold
                        (1.0, 0.001), (1.0, 0.01), (1.0, 0.1), (1.0, 0.5),  # α=1 baseline at k=16
                        # canonical α × pt grid for sweep plots
                        (0.5, 0.001), (0.5, 0.01), (0.5, 0.1), (0.5, 0.5),
                        (2.0, 0.001), (2.0, 0.01), (2.0, 0.1), (2.0, 0.5),
                    ]
                    for alpha, pt in pt_alpha_new:
                        _run(f"extension_by_pt_alpha:{alpha}:{pt}:{F}:{T}",
                             {**common,
                              "method": f"extension_by_pt_alpha:{alpha}:{pt}:{F}:{T}",
                              "suffix_cache": _SUFFIX_ENABLED,
                              "real_step_cost_ms": ext_cost_fallback,
                              "real_step_target_fn": _target_forward,
                              "real_step_draft_only_ms": ext_draft_only},
                             f"extension_by_pt_alpha_a{alpha}_t{pt}_{tag}")
                # AND filter — BOUNDARY-EDGE: α∈{3,5}, pt=0.0001, only at score∈{1,3} (best score area).
                if has_draft_p_t:
                    combined_new = [
                        # α UPPER × current best pt (0.001) × low scores
                        (3.0, 0.001, 1.0), (3.0, 0.001, 3.0),
                        (5.0, 0.001, 1.0), (5.0, 0.001, 3.0),
                        # pt LOWER × current best α (2.0) × low scores
                        (2.0, 0.0001, 1.0), (2.0, 0.0001, 3.0),
                        # corner
                        (3.0, 0.0001, 1.0), (3.0, 0.0001, 3.0),
                        (5.0, 0.0001, 1.0), (5.0, 0.0001, 3.0),
                    ]
                    for alpha, pt, sc in combined_new:
                        _run(f"extension_by_combined:{alpha}:{pt}:{sc}:{F}:{T}",
                             {**common,
                              "method": f"extension_by_combined:{alpha}:{pt}:{sc}:{F}:{T}",
                              "suffix_cache": _SUFFIX_ENABLED,
                              "real_step_cost_ms": ext_cost_fallback,
                              "real_step_target_fn": _target_forward,
                              "real_step_draft_only_ms": ext_draft_only},
                             f"extension_by_combined_a{alpha}_pt{pt}_s{sc:.1f}_{tag}")
                # Multiplicative — BOUNDARY-EDGE: thr LOWER (0.025, 0.01) × best+neighbor α.
                if has_draft_p_t:
                    product_new = [
                        (0.5, 0.025), (0.5, 0.01),
                        (1.0, 0.025), (1.0, 0.01),
                        (2.0, 0.025), (2.0, 0.01),
                    ]
                    for alpha, prod_t in product_new:
                        _run(f"extension_by_product:{alpha}:{prod_t}:{F}:{T}",
                             {**common,
                              "method": f"extension_by_product:{alpha}:{prod_t}:{F}:{T}",
                              "suffix_cache": _SUFFIX_ENABLED,
                              "real_step_cost_ms": ext_cost_fallback,
                              "real_step_target_fn": _target_forward,
                              "real_step_draft_only_ms": ext_draft_only},
                             f"extension_by_product_a{alpha}_t{prod_t}_{tag}")
                # NEW: extension_pt_prune (backbone prune + graft filter, same threshold).
                # Sweep parallel to by_pt_alpha.
                if has_draft_p_t:
                    for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
                        for pt in [0.0001, 0.001, 0.01, 0.1, 0.5]:
                            _run(f"extension_pt_prune:{alpha}:{pt}:{F}:{T}",
                                 {**common,
                                  "method": f"extension_pt_prune:{alpha}:{pt}:{F}:{T}",
                                  "suffix_cache": _SUFFIX_ENABLED,
                                  "real_step_cost_ms": ext_cost_fallback,
                                  "real_step_target_fn": _target_forward,
                                  "real_step_draft_only_ms": ext_draft_only},
                                 f"extension_pt_prune_a{alpha}_t{pt}_{tag}")
                # NEW: extension_top1 (per-depth top-1 chain graft, backbone full).
                if has_draft_p_t:
                    _run(f"extension_top1:{F}:{T}",
                         {**common,
                          "method": f"extension_top1:{F}:{T}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": ext_draft_only},
                         f"extension_top1_{tag}")
                # NEW: extension_topb_pathprob (token-level top-B by unified rank).
                if has_draft_p_t:
                    _run(f"extension_topb_pathprob:{F}:{T}",
                         {**common,
                          "method": f"extension_topb_pathprob:{F}:{T}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": ext_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": ext_draft_only},
                         f"extension_topb_pathprob_{tag}")
                # NEW: extension_topb_pathprob_alpha (topb with α calibration on extension probs).
                if has_draft_p_t:
                    for alpha in (2.0, 4.0, 8.0):
                        _run(f"extension_topb_pathprob_alpha:{alpha}:{F}:{T}",
                             {**common,
                              "method": f"extension_topb_pathprob_alpha:{alpha}:{F}:{T}",
                              "suffix_cache": _SUFFIX_ENABLED,
                              "real_step_cost_ms": ext_cost_fallback,
                              "real_step_target_fn": _target_forward,
                              "real_step_draft_only_ms": ext_draft_only},
                             f"extension_topb_pathprob_alpha_a{alpha}_{tag}")
                # NEW: extension_topb_pathprob_keep1 (topb_alpha + force-preserve top-1 backbone chain).
                if has_draft_p_t:
                    for alpha in (1.0, 2.0):
                        _run(f"extension_topb_pathprob_keep1:{alpha}:{F}:{T}",
                             {**common,
                              "method": f"extension_topb_pathprob_keep1:{alpha}:{F}:{T}",
                              "suffix_cache": _SUFFIX_ENABLED,
                              "real_step_cost_ms": ext_cost_fallback,
                              "real_step_target_fn": _target_forward,
                              "real_step_draft_only_ms": ext_draft_only},
                             f"extension_topb_pathprob_keep1_a{alpha}_{tag}")
                # NEW: extension_bns (Branch Nucleus Selection). Tree size is set
                # by ρ + λ_suffix (not B), but draft-only cost still varies with
                # B so we enroll at every budget. F=1.0 is too restrictive for
                # BNS to be informative — skip it.
                # Expanded grid: 5×5=25 combos. Initial 3×3 sweep showed best at
                # upper edge (λ=1.0, ρ=0.8/0.9) so we extend both up (λ=1.5/2.0,
                # ρ=0.95/0.98) to find the actual optimum.
                if has_draft_p_t and F >= 2.0:
                    for lam in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                        for rho in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98):
                            _run(f"extension_bns:{lam}:{rho}:{F}:{T}",
                                 {**common,
                                  "method": f"extension_bns:{lam}:{rho}:{F}:{T}",
                                  "suffix_cache": _SUFFIX_ENABLED,
                                  "real_step_cost_ms": ext_cost_fallback,
                                  "real_step_target_fn": _target_forward,
                                  "real_step_draft_only_ms": ext_draft_only},
                                 f"extension_bns_l{lam}_r{rho}_{tag}")
                # DNS (extension_dns, formerly BNSv2): depth-nucleus +
                # node-prob score + dedup-boost. Compact grid; expand once
                # results justify.
                if has_draft_p_t and F >= 2.0:
                    for lam in (0.5, 1.0, 2.0):
                        for rho in (0.50, 0.70, 0.80, 0.90, 0.98):
                            _run(f"extension_dns:{lam}:{rho}:{F}:{T}",
                                 {**common,
                                  "method": f"extension_dns:{lam}:{rho}:{F}:{T}",
                                  "suffix_cache": _SUFFIX_ENABLED,
                                  "real_step_cost_ms": ext_cost_fallback,
                                  "real_step_target_fn": _target_forward,
                                  "real_step_draft_only_ms": ext_draft_only},
                                 f"extension_dns_l{lam}_r{rho}_{tag}")
                # extension_topk: per-depth top-k variant (no nucleus mass
                # thresholding). Same dedup + λ + path-prob as DNS; only
                # the per-depth selection rule changes. k bounds the tree
                # per depth.
                if has_draft_p_t and F >= 2.0:
                    for lam in (0.5, 1.0, 2.0):
                        for k in (2, 4, 8, 16, 32):
                            _run(f"extension_topk:{lam}:{k}:{F}:{T}",
                                 {**common,
                                  "method": f"extension_topk:{lam}:{k}:{F}:{T}",
                                  "suffix_cache": _SUFFIX_ENABLED,
                                  "real_step_cost_ms": ext_cost_fallback,
                                  "real_step_target_fn": _target_forward,
                                  "real_step_draft_only_ms": ext_draft_only},
                                 f"extension_topk_l{lam}_k{k}_{tag}")
                # extension_calib_dns / extension_calib_topk: drop the λ_suffix
                # axis entirely — the online accept-rate calibrator learns the
                # eagle3↔suffix scale per-edge during the sim. Sweep only the
                # selection knob (ρ for DNS, k for topk); same grids as their
                # λ-based counterparts so the comparison is argmax-vs-argmax.
                if has_draft_p_t and F >= 2.0:
                    for rho in (0.50, 0.70, 0.80, 0.90, 0.98):
                        _run(f"extension_calib_dns:{rho}:{F}:{T}",
                             {**common,
                              "method": f"extension_calib_dns:{rho}:{F}:{T}",
                              "suffix_cache": _SUFFIX_ENABLED,
                              "real_step_cost_ms": ext_cost_fallback,
                              "real_step_target_fn": _target_forward,
                              "real_step_draft_only_ms": ext_draft_only},
                             f"extension_calib_dns_r{rho}_{tag}")
                    for k in (2, 4, 8, 16, 32):
                        _run(f"extension_calib_topk:{k}:{F}:{T}",
                             {**common,
                              "method": f"extension_calib_topk:{k}:{F}:{T}",
                              "suffix_cache": _SUFFIX_ENABLED,
                              "real_step_cost_ms": ext_cost_fallback,
                              "real_step_target_fn": _target_forward,
                              "real_step_draft_only_ms": ext_draft_only},
                             f"extension_calib_topk_k{k}_{tag}")
                # Offline isotonic calibration (train/test split workflow).
                # FIT methods need SIM_ISO_COLLECT_OUT, EVAL methods need
                # SIM_ISO_CALIB — env-gated so default sweeps (no --methods
                # filter) never enroll them.
                if (has_draft_p_t and F >= 2.0
                        and os.environ.get("SIM_ISO_COLLECT_OUT")):
                    for k in (2, 4, 8, 16, 32):
                        _run(f"extension_isofit_topk:{k}:{F}:{T}",
                             {**common,
                              "method": f"extension_isofit_topk:{k}:{F}:{T}",
                              "suffix_cache": _SUFFIX_ENABLED,
                              "real_step_cost_ms": ext_cost_fallback,
                              "real_step_target_fn": _target_forward,
                              "real_step_draft_only_ms": ext_draft_only},
                             f"extension_isofit_topk_k{k}_{tag}")
                if (has_draft_p_t and F >= 2.0
                        and os.environ.get("SIM_ISO_CALIB")):
                    for k in (2, 4, 8, 16, 32):
                        _run(f"extension_iso_topk:{k}:{F}:{T}",
                             {**common,
                              "method": f"extension_iso_topk:{k}:{F}:{T}",
                              "suffix_cache": _SUFFIX_ENABLED,
                              "real_step_cost_ms": ext_cost_fallback,
                              "real_step_target_fn": _target_forward,
                              "real_step_draft_only_ms": ext_draft_only},
                             f"extension_iso_topk_k{k}_{tag}")
                # extension_by_match_len: basic + match_len gate. Filters out
                # suffix grafts whose context-suffix match was shallow (low-
                # confidence patterns). Sweep M={1,2,4,8,16}; M=1 keeps every
                # graft (essentially basic), M=16 is the strictest (cache's
                # max_tree_depth=64, so 16 = matched ≥1/4 of the trie depth).
                if has_draft_p_t and F >= 2.0:
                    for M in (1, 2, 4, 8, 16):
                        _run(f"extension_by_match_len:{M}:{F}:{T}",
                             {**common,
                              "method": f"extension_by_match_len:{M}:{F}:{T}",
                              "suffix_cache": _SUFFIX_ENABLED,
                              "real_step_cost_ms": ext_cost_fallback,
                              "real_step_target_fn": _target_forward,
                              "real_step_draft_only_ms": ext_draft_only},
                             f"extension_by_match_len_m{M}_{tag}")
                # extension_topk_match: stack the match_len gate on top of
                # per-depth topk selection. λ/k mirror topk's best region
                # (from prior sweep λ=2.0 k=16). Sweep M to find the gate
                # threshold where short-match drafts hurt vs help.
                if has_draft_p_t and F >= 2.0:
                    for lam in (1.0, 2.0):
                        for k in (8, 16):
                            for M in (2, 4, 8):
                                _run(f"extension_topk_match:{lam}:{k}:{M}:{F}:{T}",
                                     {**common,
                                      "method": f"extension_topk_match:{lam}:{k}:{M}:{F}:{T}",
                                      "suffix_cache": _SUFFIX_ENABLED,
                                      "real_step_cost_ms": ext_cost_fallback,
                                      "real_step_target_fn": _target_forward,
                                      "real_step_draft_only_ms": ext_draft_only},
                                     f"extension_topk_match_l{lam}_k{k}_m{M}_{tag}")
                # extension_descrank: per-depth top-k by composite
                # path_prob × (1 + alpha · log(1+n_descendants)). The
                # n_descendants term is the strongest single signal in the
                # per-node data (5.4× precision lift on suffix accepts vs
                # path_prob's 1.8×). alpha=0 → identical to topk (baseline).
                if has_draft_p_t and F >= 2.0:
                    for lam in (1.0, 2.0):
                        for k in (8, 16):
                            for alpha in (0.0, 0.3, 0.5, 1.0, 2.0):
                                _run(f"extension_descrank:{lam}:{k}:{alpha}:{F}:{T}",
                                     {**common,
                                      "method": f"extension_descrank:{lam}:{k}:{alpha}:{F}:{T}",
                                      "suffix_cache": _SUFFIX_ENABLED,
                                      "real_step_cost_ms": ext_cost_fallback,
                                      "real_step_target_fn": _target_forward,
                                      "real_step_draft_only_ms": ext_draft_only},
                                     f"extension_descrank_l{lam}_k{k}_a{alpha}_{tag}")
                # extension_depth_cap: basic + hard depth cap. Per-node data
                # shows ~7% of suffix budget at depth > 12 has 0 accepted
                # nodes; ~20% at depth > 10 has only ~2% accepts. Free budget
                # by truncating impossible-to-accept deep tail.
                if F >= 2.0:
                    for D in (6, 8, 10, 12, 16):
                        _run(f"extension_depth_cap:{D}:{F}:{T}",
                             {**common,
                              "method": f"extension_depth_cap:{D}:{F}:{T}",
                              "suffix_cache": _SUFFIX_ENABLED,
                              "real_step_cost_ms": ext_cost_fallback,
                              "real_step_target_fn": _target_forward,
                              "real_step_draft_only_ms": ext_draft_only},
                             f"extension_depth_cap_d{D}_{tag}")
                # extension_topk_cap: topk per-depth selection + depth cap.
                # Stacks two complementary prunings.
                if has_draft_p_t and F >= 2.0:
                    for lam in (1.0, 2.0):
                        for k in (8, 16):
                            for D in (6, 8, 10, 12):
                                _run(f"extension_topk_cap:{lam}:{k}:{D}:{F}:{T}",
                                     {**common,
                                      "method": f"extension_topk_cap:{lam}:{k}:{D}:{F}:{T}",
                                      "suffix_cache": _SUFFIX_ENABLED,
                                      "real_step_cost_ms": ext_cost_fallback,
                                      "real_step_target_fn": _target_forward,
                                      "real_step_draft_only_ms": ext_draft_only},
                                     f"extension_topk_cap_l{lam}_k{k}_d{D}_{tag}")
                # extension_dnsv2: DNS without candidate-pool normalization
                # (raw-cumsum path-prob threshold). ρ grid spans absolute
                # mass thresholds; tiny ρ keeps many deep-d nodes (where
                # absolute path-prob is small), big ρ aggressive at shallow.
                if has_draft_p_t and F >= 2.0:
                    for lam in (0.5, 1.0, 2.0):
                        for rho in (0.001, 0.01, 0.1, 0.3, 0.5):
                            _run(f"extension_dnsv2:{lam}:{rho}:{F}:{T}",
                                 {**common,
                                  "method": f"extension_dnsv2:{lam}:{rho}:{F}:{T}",
                                  "suffix_cache": _SUFFIX_ENABLED,
                                  "real_step_cost_ms": ext_cost_fallback,
                                  "real_step_target_fn": _target_forward,
                                  "real_step_draft_only_ms": ext_draft_only},
                                 f"extension_dnsv2_l{lam}_r{rho}_{tag}")
                # Breakdown-best enrollment (env-driven, additive).
                # BREAKDOWN_BEST_CONFIGS=/path/to/best_configs.json enrolls only those.
                _BD_BEST_FILE = os.environ.get("BREAKDOWN_BEST_CONFIGS")
                if has_draft_p_t and _BD_BEST_FILE and os.path.exists(_BD_BEST_FILE):
                    import json as _json
                    with open(_BD_BEST_FILE) as _f:
                        _bd_cfgs = _json.load(_f)
                    for _cfg in _bd_cfgs:
                        if abs(_cfg.get("F", -1) - F) > 1e-9 or abs(_cfg.get("T", -1) - T) > 1e-9:
                            continue
                        _fam = _cfg["family"]
                        if _fam == "extension_by_product":
                            _a, _t = _cfg["alpha"], _cfg["threshold"]
                            _run(f"extension_by_product:{_a}:{_t}:{F}:{T}",
                                 {**common, "method": f"extension_by_product:{_a}:{_t}:{F}:{T}",
                                  "suffix_cache": _SUFFIX_ENABLED,
                                  "real_step_cost_ms": ext_cost_fallback,
                                  "real_step_target_fn": _target_forward,
                                  "real_step_draft_only_ms": ext_draft_only},
                                 f"extension_by_product_a{_a}_t{_t}_{tag}")
                        elif _fam == "extension_by_combined":
                            _a, _pt, _sc = _cfg["alpha"], _cfg["pt"], _cfg["score"]
                            _run(f"extension_by_combined:{_a}:{_pt}:{_sc}:{F}:{T}",
                                 {**common, "method": f"extension_by_combined:{_a}:{_pt}:{_sc}:{F}:{T}",
                                  "suffix_cache": _SUFFIX_ENABLED,
                                  "real_step_cost_ms": ext_cost_fallback,
                                  "real_step_target_fn": _target_forward,
                                  "real_step_draft_only_ms": ext_draft_only},
                                 f"extension_by_combined_a{_a}_pt{_pt}_s{_sc:.1f}_{tag}")
                        elif _fam == "extension_by_pt_alpha":
                            _a, _pt = _cfg["alpha"], _cfg["pt"]
                            _run(f"extension_by_pt_alpha:{_a}:{_pt}:{F}:{T}",
                                 {**common, "method": f"extension_by_pt_alpha:{_a}:{_pt}:{F}:{T}",
                                  "suffix_cache": _SUFFIX_ENABLED,
                                  "real_step_cost_ms": ext_cost_fallback,
                                  "real_step_target_fn": _target_forward,
                                  "real_step_draft_only_ms": ext_draft_only},
                                 f"extension_by_pt_alpha_a{_a}_t{_pt}_{tag}")
                        # extension_by_score / extension already enrolled above for this F/T.

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

                for thresh in [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]:
                    _run(f"extension_dm_by_score:{thresh}:{F}:{T}",
                         {**common, "method": f"extension_dm_by_score:{thresh}:{F}:{T}",
                          "suffix_cache": _SUFFIX_ENABLED,
                          "real_step_cost_ms": dm_cost_fallback,
                          "real_step_target_fn": _target_forward,
                          "real_step_draft_only_ms": dm_draft_only},
                         f"extension_dm_by_score_t{thresh:.1f}_{tag}")
                # Alpha-weighted backbone-prob anchor-skip filter (dm backbone).
                if has_draft_p_t:
                    for alpha in [0.5, 1.0, 2.0]:
                        for pt in [0.001, 0.01, 0.1, 0.5]:
                            _run(f"extension_dm_by_pt_alpha:{alpha}:{pt}:{F}:{T}",
                                 {**common,
                                  "method": f"extension_dm_by_pt_alpha:{alpha}:{pt}:{F}:{T}",
                                  "suffix_cache": _SUFFIX_ENABLED,
                                  "real_step_cost_ms": dm_cost_fallback,
                                  "real_step_target_fn": _target_forward,
                                  "real_step_draft_only_ms": dm_draft_only},
                                 f"extension_dm_by_pt_alpha_a{alpha}_t{pt}_{tag}")
                if has_draft_p_t:
                    for alpha in [1.0]:
                        for pt in [0.01, 0.1]:
                            for sc in [3.0, 10.0, 15.0]:
                                _run(f"extension_dm_by_combined:{alpha}:{pt}:{sc}:{F}:{T}",
                                     {**common,
                                      "method": f"extension_dm_by_combined:{alpha}:{pt}:{sc}:{F}:{T}",
                                      "suffix_cache": _SUFFIX_ENABLED,
                                      "real_step_cost_ms": dm_cost_fallback,
                                      "real_step_target_fn": _target_forward,
                                      "real_step_draft_only_ms": dm_draft_only},
                                     f"extension_dm_by_combined_a{alpha}_pt{pt}_s{sc:.1f}_{tag}")
                if has_draft_p_t:
                    for alpha in [0.5, 1.0, 2.0]:
                        for prod_t in [0.5, 2.0, 5.0]:
                            _run(f"extension_dm_by_product:{alpha}:{prod_t}:{F}:{T}",
                                 {**common,
                                  "method": f"extension_dm_by_product:{alpha}:{prod_t}:{F}:{T}",
                                  "suffix_cache": _SUFFIX_ENABLED,
                                  "real_step_cost_ms": dm_cost_fallback,
                                  "real_step_target_fn": _target_forward,
                                  "real_step_draft_only_ms": dm_draft_only},
                                 f"extension_dm_by_product_a{alpha}_t{prod_t}_{tag}")

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
    # Input: Stage 1 agent_trajectory + optional Stage 2 draft-model drafts.
    # Records are assembled on the fly; suffix is drawn live in-sim.
    parser.add_argument("--agent-trajectory", required=True,
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
    parser.add_argument("--latency-data", default=None,
                        help="Path to latency_data.json. "
                             "When omitted, MAT / accept-rate stats are still "
                             "reported but latency-aware speedup numbers are skipped.")
    parser.add_argument("--topk", type=int, default=None,
                        help="EAGLE3 topk used for this Stage 1 run. "
                             "When set, latency lookups pull from the "
                             "per-topk tables in latency_data.json "
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
    parser.add_argument("--proposer-label", default="eagle3",
                        choices=["eagle3", "mtp"],
                        help="Display label for the eagle3-family proposer in "
                             "output JSON. Use 'mtp' for Qwen3.5-9B captures "
                             "(the proposer is actually MTP — see "
                             "project_eagle_label_means_mtp memory). Renames "
                             "eagle3* → label*, hybrid_e3_* → hybrid_label_*, "
                             "hybrid_oracle_* → hybrid_oracle_label_*. Internal "
                             "sim logic keeps using 'eagle3'; only the on-disk "
                             "JSON keys/strings are rewritten.")
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
        agent_trajectory_path=args.agent_trajectory,
        suffix_drafts_path=None,
        draft_model_drafts_path=args.draft_model_drafts,
        mtp_agent_trajectory_path=None,
        exclude_path=args.exclude,
        model=args.model,
        dataset_path=args.dataset,
        responses_path=args.responses,
        eagle3_reslice=eagle3_reslice,
    )
    input_source = args.agent_trajectory

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
        seq, ind, cond_acc, cond_dn, denom_depth = position_accept_rates(
            tids or [], pids or [], gt, POSITION_ACCEPT_MAX)
        if denom_depth <= 0:
            return
        stats = position_accepts.setdefault(prop_name, {
            "seq_accept": [0] * POSITION_ACCEPT_MAX,
            "ind_accept": [0] * POSITION_ACCEPT_MAX,
            "depth_ge": [0] * POSITION_ACCEPT_MAX,
            "cond_accept": [0] * POSITION_ACCEPT_MAX,
            "cond_denom": [0] * POSITION_ACCEPT_MAX,
        })
        # depth_ge counts ALL positions up to denom_depth, regardless of
        # whether this step's tree was deep enough to draft at position d.
        # This avoids the "deep-tree steps inflate deep-position accept rate"
        # bias that variable-depth proposers (suffix / EAGLE3 with reslice
        # shorter than tree) would otherwise introduce.
        # cond_denom only increments when prev position was accepted —
        # cond_rate[d] = cond_accept[d] / cond_denom[d] = P(accept[d] |
        # accept[d-1]).
        for d in range(denom_depth):
            stats["depth_ge"][d] += 1
            stats["seq_accept"][d] += seq[d]
            stats["ind_accept"][d] += ind[d]
            stats["cond_accept"][d] += cond_acc[d]
            stats["cond_denom"][d] += cond_dn[d]

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

    # Latency-aware simulation. When --latency-data is missing, feed a
    # stub config (vanilla_step_ms=1.0, empty per-budget tables); speedup
    # numbers become placeholders but MAT is unaffected.
    have_latency = bool(args.latency_data)
    if have_latency:
        with open(args.latency_data) as f:
            latency_data = json.load(f)
    else:
        print("NOTE: --latency-data not provided; MAT is still reported "
              "but speedup numbers will be stub values (not measured).",
              file=sys.stderr)
        latency_data = {
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
        records, budgets, latency_data,
        topk=args.topk, steps=args.steps,
        method_filter=method_filter)

    if args.print_summary:
        print_summary(budgets)
        try:
            print_latency_summary(latency_results, budgets,
                                  latency_data["vanilla_step_ms"])
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
            "vanilla_step_ms": latency_data["vanilla_step_ms"],
            "proposers": proposers,
            "pairs": pairs,
            "has_latency_data": have_latency,
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
                "latency_data not provided; MAT values are accurate but "
                "speedup_* columns use stub latencies (not meaningful)")

        _rename_proposer_keys_inplace(output, args.proposer_label)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Output: {args.output}", file=sys.stderr)

    # BNS trace dump (env-gated)
    if _BNS_TRACE['enabled']:
        out_path = os.environ['BNS_TRACE_OUT']
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({
                'aggregates': _BNS_TRACE['agg'],
                'sample_records': _BNS_TRACE['records'],
            }, f, indent=2)
        print(f"BNS trace: {_BNS_TRACE['agg']['parent_count']} parents → {out_path}", file=sys.stderr)

    # BNS calibration dump (env-gated): per-branch (norm_R, accepted) records
    # for parents on the greedy accept path of the full pre-prune tree.
    if _BNS_CALIB['enabled']:
        out_path = os.environ['BNS_CALIB_OUT']
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({
                'schema': ['norm_R', 'accepted', 'k_siblings',
                           'parent_depth', 'subtree_mat'],
                'n_records': len(_BNS_CALIB['records']),
                'records': _BNS_CALIB['records'],
            }, f)
        print(f"BNS calib: {len(_BNS_CALIB['records'])} branch records → {out_path}", file=sys.stderr)


def _rename_proposer_keys_inplace(output: dict, label: str) -> None:
    """Rewrite eagle3-family keys/strings in the assembled output dict to use
    a different proposer label (e.g. ``'mtp'``). Internal sim logic still
    uses ``'eagle3'`` everywhere; only the on-disk JSON changes. Applied
    once, just before ``json.dump``.

    For Qwen3.5-9B captures the proposer is actually MTP (built-in head),
    not real EAGLE3 — see project_eagle_label_means_mtp memory. Pass
    ``--proposer-label mtp`` so the output JSON reflects that distinction.

    Renames (when ``label != 'eagle3'``):
      * ``eagle3``           → ``label``                       (proposers list, dict keys)
      * ``eagle3_*``         → ``label_*``                     (e.g. eagle3_mat → mtp_mat)
      * ``eagle3+...``       → ``label+...``                   (pairs entries)
      * ``hybrid_e3``        → ``hybrid_label``
      * ``hybrid_e3_*``      → ``hybrid_label_*``              (e.g. hybrid_e3_f1.0_t0.0_th3.0 → hybrid_mtp_f1.0_t0.0_th3.0)
      * ``hybrid_oracle``    → ``hybrid_oracle_label``
      * ``hybrid_oracle_*``  → ``hybrid_oracle_label_*``       (eagle3-backed variant)
    """
    if label == "eagle3":
        return

    EXACT = {
        "eagle3":         label,
        "hybrid_e3":      f"hybrid_{label}",
        "hybrid_oracle":  f"hybrid_oracle_{label}",
    }
    PREFIXES = [
        ("eagle3+",        f"{label}+"),
        ("eagle3_",        f"{label}_"),
        ("hybrid_e3_",     f"hybrid_{label}_"),
        ("hybrid_oracle_", f"hybrid_oracle_{label}_"),
    ]

    def _rn(s: str) -> str:
        if s in EXACT:
            return EXACT[s]
        for old, new in PREFIXES:
            if s.startswith(old):
                return new + s[len(old):]
        return s

    def _walk(obj):
        # Returns a renamed copy. Strings inside dicts (values) are NOT
        # renamed — only dict keys and items in lists-of-strings (proposers,
        # pairs) get rewritten.
        if isinstance(obj, dict):
            return {_rn(k): _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_rn(x) if isinstance(x, str) else _walk(x) for x in obj]
        return obj

    new = _walk(output)
    output.clear()
    output.update(new)


if __name__ == "__main__":
    main()
