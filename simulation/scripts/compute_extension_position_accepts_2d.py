"""Per-(backbone, extension) position accept-rate stats for the
extension method.

Each step's draft tree is rebuilt as a NO-DEDUP extension tree:
  * EAGLE3 backbone (truncated to budget B), each node carrying coord
    (b = backbone depth, e = 0).
  * Root-level suffix graft (anchor = virtual root): every suffix node
    carries coord (0, e ≥ 1). This is the suffix-only column.
  * Per-base-node suffix graft (anchor = base node X at depth d_X):
    every suffix node carries coord (d_X, e ≥ 1) — the genuine
    "extension" cells.

NO-DEDUP: each suffix graft is appended as fresh nodes even if they
duplicate tokens already present elsewhere. The all-paths verifier in
``position_accept_rates_2d`` OR-aggregates per cell so duplicates do
not double-count.

Scoring: every node's root→node path is checked against the ground
truth in topological order. A cell is sequentially accepted iff some
node at that coord has its FULL path matched; conditionally evaluable
iff some node at that coord has its parent's path matched.

Output schema (mirrors compute_position_accepts.py for 1-D):

  {
    "metadata": {...},
    "position_accepts_2d": {
      "max_b": 8,
      "max_e": 16,
      "by_proposer": {
        "extension": {
          "seq_accept": [[..max_e+1..]] * (max_b+1),
          "ind_accept": [[..]],
          "depth_ge":   [[..]],
          "cond_accept":[[..]],
          "cond_denom": [[..]]
        }
      }
    }
  }

Suffix cache update: same bonus-only policy as the 1-D collection —
exactly one ground-truth token is fed to the cache per recorded step.
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

from simulation.evaluation.tree_knapsack import position_accept_rates_2d  # noqa: E402
from simulation.pipeline.assemble_records import (  # noqa: E402
    assemble_records_from_artifacts,
)


SUFFIX_FACTOR = 4.0
SUFFIX_MIN_PROB = 0.0


def _empty_grid(rows: int, cols: int) -> List[List[int]]:
    return [[0] * cols for _ in range(rows)]


def _empty_stats(rows: int, cols: int) -> Dict[str, List[List[int]]]:
    return {
        # Standard (no-dedup tree, all cells).
        "seq_accept":  _empty_grid(rows, cols),
        "ind_accept":  _empty_grid(rows, cols),
        "depth_ge":    _empty_grid(rows, cols),
        "cond_accept": _empty_grid(rows, cols),
        "cond_denom":  _empty_grid(rows, cols),
        # Reachable: pure-extension cells (b ≥ 1, e ≥ 1) only, on a
        # DEDUPED tree (so duplicates with eagle3 backbone tokens are
        # excluded — only genuine suffix-extension proposals count),
        # AND only when the cell's anchor backbone is accepted by
        # eagle3 alone (eagle3.chain ≥ b).
        "seq_accept_reachable":  _empty_grid(rows, cols),
        "ind_accept_reachable":  _empty_grid(rows, cols),
        "depth_ge_reachable":    _empty_grid(rows, cols),
        "cond_accept_reachable": _empty_grid(rows, cols),
        "cond_denom_reachable":  _empty_grid(rows, cols),
    }


def _add_grids(dst: List[List[int]], src: List[List[int]]) -> None:
    rows = len(dst)
    cols = len(dst[0]) if rows else 0
    for b in range(rows):
        for e in range(cols):
            dst[b][e] += src[b][e]


def _build_extension_tree(
    base_tids: List[int],
    base_pids: List[int],
    base_context: List[int],
    suffix_cache,
    cache_req_id: str,
    budget: int,
    max_b: int,
    max_e: int,
    dedup: bool = False,
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """Build extension tree with per-node (b, e) coords.

    By default NO-DEDUP (duplicate (parent, token) pairs allowed). Set
    ``dedup=True`` to merge duplicates: when a suffix graft would add
    a (parent, token) that already exists in the tree, skip it (uses
    the existing node instead, like the simulator's _extension_step).

    Returns (token_ids, parents, coords). Tree is topologically
    ordered (parent_idx < child_idx).
    """
    n = min(budget, len(base_tids))
    tids = list(base_tids[:n])
    pids = [p if (p < n) else -1 for p in base_pids[:n]]

    # BFS depth per base node — roots (parent = -1) have depth 1.
    base_depth = [0] * n
    for i in range(n):
        if pids[i] < 0:
            base_depth[i] = 1
        else:
            base_depth[i] = base_depth[pids[i]] + 1

    ext_tids: List[int] = list(tids)
    ext_pids: List[int] = list(pids)
    ext_coords: List[Tuple[int, int]] = [(base_depth[i], 0) for i in range(n)]

    # Children index for dedup mode: parent_idx → {token: child_idx}.
    children: Dict[int, Dict[int, int]] = {}
    if dedup:
        for i in range(len(ext_tids)):
            p = ext_pids[i]
            children.setdefault(p, {})[int(ext_tids[i])] = i

    # Path tokens root→node for context construction.
    paths: List[List[int]] = []
    for i in range(n):
        path: List[int] = []
        node = i
        while node >= 0:
            path.append(tids[node])
            node = pids[node]
        path.reverse()
        paths.append(path)

    def _graft(anchor_global_idx: int, anchor_b: int, ctx_arr: np.ndarray):
        """Run suffix.speculate from ctx_arr, append every returned
        node into the extension tree under anchor_global_idx (-1 means
        virtual root). depth-in-graft e starts at 1 directly under the
        anchor. With ``dedup=True``, a node whose (parent, token)
        already exists merges into the existing one."""
        try:
            draft = suffix_cache.speculate(
                cache_req_id, ctx_arr,
                max_spec_factor=SUFFIX_FACTOR,
                min_token_prob=SUFFIX_MIN_PROB,
                use_tree_spec=True,
            )
        except Exception:
            return
        if not draft.token_ids:
            return
        local_to_global: Dict[int, int] = {}
        for j, (tok, p) in enumerate(zip(draft.token_ids, draft.parents)):
            if p == -1:
                global_parent = anchor_global_idx
                depth_in_graft = 1
            else:
                global_parent = local_to_global.get(p)
                if global_parent is None:
                    continue
                depth_in_graft = ext_coords[global_parent][1] + 1
            if anchor_b > max_b or depth_in_graft > max_e:
                continue
            tok_int = int(tok)
            if dedup:
                existing = children.get(global_parent, {}).get(tok_int)
                if existing is not None:
                    local_to_global[j] = existing  # merge
                    continue
            new_idx = len(ext_tids)
            ext_tids.append(tok_int)
            ext_pids.append(int(global_parent))
            ext_coords.append((anchor_b, depth_in_graft))
            local_to_global[j] = new_idx
            if dedup:
                children.setdefault(int(global_parent), {})[tok_int] = new_idx

    # Root graft (anchor depth = 0, parent = -1)
    if base_context:
        _graft(-1, 0, np.asarray(base_context, dtype=np.int32))

    # Per-base-node grafts
    for i in range(n):
        if base_depth[i] > max_b:
            continue
        ctx = base_context + paths[i]
        _graft(i, base_depth[i], np.asarray(ctx, dtype=np.int32))

    return ext_tids, ext_pids, ext_coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-results", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--responses", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--reslice-steps", type=int, default=8)
    ap.add_argument("--reslice-topk", type=int, default=8)
    ap.add_argument("--capture-steps", type=int, default=8)
    ap.add_argument("--capture-topk", type=int, default=16)
    ap.add_argument("--budget", type=int, default=128,
                    help="Backbone tree truncation cap.")
    ap.add_argument("--max-b", type=int, default=8,
                    help="Backbone-axis grid extent (≥ reslice-steps).")
    ap.add_argument("--max-e", type=int, default=16,
                    help="Extension-axis grid extent.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--base-proposer", default="eagle3",
                    choices=["eagle3", "draft_model"],
                    help="Backbone source: eagle3 (default) or "
                         "draft_model. For draft_model you must also "
                         "pass --draft-model-drafts.")
    ap.add_argument("--draft-model-drafts", default=None,
                    help="JSONL of Stage-2 draft-model trees; required "
                         "when --base-proposer draft_model.")
    ap.add_argument("--proposer-name", default=None,
                    help="Output by_proposer key. Defaults to "
                         "'extension' for eagle3 backbone and "
                         "'extension_dm' for draft_model backbone.")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="Cap (request, call) sequences for quick runs.")
    args = ap.parse_args()

    if args.proposer_name is None:
        args.proposer_name = ("extension_dm"
                              if args.base_proposer == "draft_model"
                              else "extension")
    if args.base_proposer == "draft_model" and not args.draft_model_drafts:
        sys.exit("--draft-model-drafts is required when "
                 "--base-proposer draft_model")
    if args.base_proposer == "eagle3":
        eagle3_reslice = (args.capture_steps, args.capture_topk,
                          args.reslice_steps, args.reslice_topk)
    else:
        # draft_model backbone is captured at a fixed depth (chain) and
        # not resliced — disable the eagle3 reslice path so per_proposer
        # carries the draft_model entry verbatim.
        eagle3_reslice = None

    print(f"[ext-2d] base_proposer={args.base_proposer} → "
          f"by_proposer key='{args.proposer_name}'", file=sys.stderr)
    print(f"[ext-2d] loading capture: {args.agent_results}",
          file=sys.stderr)
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
    print(f"[ext-2d] {len(records)} records in {time.time() - t0:.1f}s",
          file=sys.stderr)

    rows = args.max_b + 1
    cols = args.max_e + 1
    stats = _empty_stats(rows, cols)

    by_seq: Dict[tuple, List[dict]] = defaultdict(list)
    for rec in records:
        by_seq[(rec["request_id"], rec.get("call_idx", 0))].append(rec)
    for k in by_seq:
        by_seq[k].sort(key=lambda r: r.get("step_idx", 0))

    try:
        from hybrid_spec_decoding.suffix_decoding.suffix_tree import (
            SuffixDecodingCache,
        )
    except Exception as e:
        sys.exit(f"[ext-2d] FATAL: cannot import SuffixDecodingCache: {e}")

    sx = SuffixDecodingCache(
        max_tree_depth=max(args.max_e, 64),
        max_cached_requests=100000,
    )

    n_seqs = 0
    n_steps = 0
    n_no_base = 0
    last_log = time.time()
    seq_keys = list(by_seq.keys())
    if args.max_questions:
        seq_keys = seq_keys[:args.max_questions]

    for ki, key in enumerate(seq_keys):
        seq = by_seq[key]
        if not seq:
            continue
        n_seqs += 1
        cache_key = f"{key[0]}_{key[1]}"
        prompt_ctx = seq[0].get("context_token_ids") or []
        sx.start_request(
            cache_key, np.asarray(prompt_ctx, dtype=np.int32))

        for rec in seq:
            gt = rec.get("ground_truth_future") or []
            if not gt:
                continue
            base = (rec.get("per_proposer") or {}).get(args.base_proposer)
            base_ctx = rec.get("context_token_ids") or []
            if not base or not base.get("token_ids") or not base_ctx:
                n_no_base += 1
                # Still advance the cache by one bonus token to keep
                # state aligned with the trajectory.
                sx.add_active_response(cache_key, [int(gt[0])])
                continue

            # ── Standard ext_2d (no-dedup tree, all cells)
            tids, pids, coords = _build_extension_tree(
                base["token_ids"], base["parents"], base_ctx,
                sx, cache_key, args.budget, args.max_b, args.max_e,
                dedup=False,
            )
            seq_g, ind_g, ca_g, cd_g, dg_g, _ = position_accept_rates_2d(
                tids, pids, coords, gt, args.max_b, args.max_e)
            for nm, src in [("seq_accept", seq_g),
                            ("ind_accept", ind_g),
                            ("depth_ge", dg_g),
                            ("cond_accept", ca_g),
                            ("cond_denom", cd_g)]:
                _add_grids(stats[nm], src)

            # ── Reachable: deduped tree (so suffix grafts that
            # duplicate eagle3 backbone tokens are excluded — they
            # aren't 'pure' suffix proposals), restricted to pure-ext
            # cells (b ≥ 1, e ≥ 1) whose anchor backbone is accepted
            # by eagle3 alone.
            from simulation.evaluation.tree_knapsack import greedy_tree_walk
            eg_chain = greedy_tree_walk(
                base["token_ids"], base["parents"], gt)
            tids_d, pids_d, coords_d = _build_extension_tree(
                base["token_ids"], base["parents"], base_ctx,
                sx, cache_key, args.budget, args.max_b, args.max_e,
                dedup=True,
            )
            seq_d, ind_d, ca_d, cd_d, dg_d, _ = position_accept_rates_2d(
                tids_d, pids_d, coords_d, gt,
                args.max_b, args.max_e)
            L = len(gt)
            for b in range(1, args.max_b + 1):  # pure-ext only: b ≥ 1
                if eg_chain < b:
                    continue
                for e in range(1, args.max_e + 1):  # pure-ext only: e ≥ 1
                    d = b + e
                    if d > L:
                        continue
                    stats["depth_ge_reachable"][b][e] += 1
                    if seq_d[b][e]: stats["seq_accept_reachable"][b][e] += 1
                    if ind_d[b][e]: stats["ind_accept_reachable"][b][e] += 1
                    if cd_d[b][e]:
                        stats["cond_denom_reachable"][b][e] += 1
                        if ca_d[b][e]:
                            stats["cond_accept_reachable"][b][e] += 1

            n_steps += 1

            # Bonus-only cache update — exactly one ground-truth token.
            sx.add_active_response(cache_key, [int(gt[0])])

        sx.stop_request(cache_key)

        if time.time() - last_log > 30:
            print(f"[ext-2d] {ki + 1}/{len(seq_keys)} seqs, "
                  f"{n_steps} steps, no_base={n_no_base}",
                  file=sys.stderr)
            last_log = time.time()

    elapsed = time.time() - t0
    print(f"[ext-2d] DONE — {n_seqs} seqs, {n_steps} steps in "
          f"{elapsed:.1f}s, no_base={n_no_base}", file=sys.stderr)

    out = {
        "metadata": {
            "input_source": args.agent_results,
            "base_proposer": args.base_proposer,
            "draft_model_drafts": args.draft_model_drafts,
            "max_b": args.max_b,
            "max_e": args.max_e,
            "budget": args.budget,
            "reslice": (
                None if args.base_proposer != "eagle3"
                else {"S": args.capture_steps, "K": args.capture_topk,
                      "s": args.reslice_steps, "k": args.reslice_topk}),
            "n_sequences": n_seqs,
            "n_steps": n_steps,
            "n_dropped_no_base_tree": n_no_base,
            "elapsed_sec": elapsed,
        },
        "position_accepts_2d": {
            "max_b": args.max_b,
            "max_e": args.max_e,
            "by_proposer": {args.proposer_name: stats},
            "_doc": ("Per-cell rate at (b, e) is "
                     "<numerator>[b][e] / depth_ge[b][e] "
                     "(seq, ind) or cond_accept[b][e] / cond_denom[b][e] "
                     "(cond). Tree is no-dedup, paths are all evaluated."),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ext-2d] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
