"""Node-count statistics for EAGLE3 backbone vs Extension tree.

For every step we record two scalars:
  n_eagle_nodes : len(base["token_ids"]) — the backbone tree size
                   after reslice.
  n_ext_nodes   : len(tids_d) — node count of the DEDUPED extension
                   tree (backbone + per-base-node suffix grafts).

Accumulates histograms (0..budget+suffix slack) plus mean/std/p50/p95.
Same loading pipeline as compute_method_depth_rates.py — different
output, light per-step work (no accept-rate accounting).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.pipeline.assemble_records import (  # noqa: E402
    assemble_records_from_artifacts,
)
from simulation.scripts.compute_extension_position_accepts_2d import (  # noqa: E402
    _build_extension_tree,
)


def _summarize(values: List[int]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.int32)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p50": 0.0,
                "p95": 0.0, "min": 0, "max": 0}
    return {
        "n":    int(arr.size),
        "mean": float(arr.mean()),
        "std":  float(arr.std(ddof=0)),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
        "min":  int(arr.min()),
        "max":  int(arr.max()),
    }


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
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--max-b", type=int, default=8)
    ap.add_argument("--max-e", type=int, default=16)
    ap.add_argument("--output", required=True)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--max-questions", type=int, default=None)
    args = ap.parse_args()

    eagle3_reslice = (args.capture_steps, args.capture_topk,
                      args.reslice_steps, args.reslice_topk)

    print(f"[nodes] loading capture: {args.agent_results}", file=sys.stderr)
    t0 = time.time()
    records = assemble_records_from_artifacts(
        agent_results_path=args.agent_results,
        suffix_drafts_path=None, draft_model_drafts_path=None,
        mtp_agent_results_path=None, exclude_path=args.exclude,
        model=args.model, dataset_path=args.dataset,
        responses_path=args.responses, eagle3_reslice=eagle3_reslice,
    )
    print(f"[nodes] {len(records)} records in {time.time() - t0:.1f}s",
          file=sys.stderr)

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
        sys.exit(f"[nodes] FATAL: cannot import SuffixDecodingCache: {e}")

    sx = SuffixDecodingCache(
        max_tree_depth=64, max_cached_requests=100000)

    # Histogram bins: 0..max_hist inclusive (one bin per integer).
    max_hist = max(args.budget, args.max_b * args.max_e + args.max_b) + 1
    eagle_hist = [0] * (max_hist + 1)
    ext_hist = [0] * (max_hist + 1)

    eagle_counts: List[int] = []
    ext_counts: List[int] = []

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
        sx.start_request(cache_key, np.asarray(prompt_ctx, dtype=np.int32))

        for rec in seq:
            gt = rec.get("ground_truth_future") or []
            base = (rec.get("per_proposer") or {}).get("eagle3")
            base_ctx = rec.get("context_token_ids") or []
            if not gt or not base or not base.get("token_ids") or not base_ctx:
                if gt:
                    sx.add_active_response(cache_key, [int(gt[0])])
                if not base or not base.get("token_ids"):
                    n_no_base += 1
                continue

            base_tids = list(base["token_ids"])
            base_pids = list(base["parents"])
            # Apply the same budget cap that _build_extension_tree uses on
            # the backbone (top-`budget` nodes by capture order) so the two
            # counts are comparable.
            n_eagle = min(args.budget, len(base_tids))

            tids_d, pids_d, _ = _build_extension_tree(
                base_tids, base_pids, base_ctx,
                sx, cache_key, args.budget, args.max_b, args.max_e,
                dedup=True)
            n_ext = len(tids_d)

            eagle_counts.append(n_eagle)
            ext_counts.append(n_ext)
            eb = min(n_eagle, max_hist)
            xb = min(n_ext, max_hist)
            eagle_hist[eb] += 1
            ext_hist[xb] += 1

            n_steps += 1
            sx.add_active_response(cache_key, [int(gt[0])])

        sx.stop_request(cache_key)

        if time.time() - last_log > 30:
            print(f"[nodes] {ki + 1}/{len(seq_keys)} seqs, "
                  f"{n_steps} steps, no_base={n_no_base}",
                  file=sys.stderr)
            last_log = time.time()

    elapsed = time.time() - t0
    print(f"[nodes] DONE — {n_seqs} seqs, {n_steps} steps in "
          f"{elapsed:.1f}s, no_base={n_no_base}", file=sys.stderr)

    out = {
        "metadata": {
            "input_source": args.agent_results,
            "max_b": args.max_b, "max_e": args.max_e,
            "budget": args.budget,
            "reslice": {"S": args.capture_steps, "K": args.capture_topk,
                        "s": args.reslice_steps, "k": args.reslice_topk},
            "n_sequences": n_seqs,
            "n_steps": n_steps,
            "n_dropped_no_base_tree": n_no_base,
            "elapsed_sec": elapsed,
            "_doc": (
                "Per-step node count for EAGLE3 backbone tree "
                "(len(base[token_ids])) and the DEDUPED extension tree "
                "(backbone + per-base-node suffix grafts, len(tids_d)). "
                "Output: per-method summary stats + histogram with one "
                "bin per integer node count up to max_hist."),
        },
        "max_hist": max_hist,
        "by_method": {
            "eagle3":    {"hist": eagle_hist, "summary": _summarize(eagle_counts)},
            "extension": {"hist": ext_hist,   "summary": _summarize(ext_counts)},
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[nodes] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
