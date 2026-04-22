"""Measure ArcticInference SuffixDecodingCache.speculate() latency under
realistic load.

Mirrors Stage 3a's (simulation/pipeline/collect_suffix_drafts.py) usage
pattern so the numbers are representative of what Stage 6 simulates:

  1. Populate the cache with full trajectories from an existing Stage 1
     output (agent_results_eagle3.json). Each trajectory's per-call token
     stream is fed via start_request / add_active_response / stop_request,
     just like Stage 3a's _warmup_cache.
  2. For each workload, pick one held-out trajectory and replay it
     token-by-token. At every step, call speculate() on the growing
     (prompt + decoded) context and time it.
  3. Report median / p90 / p99 across all speculate() calls per workload.

This is the right benchmark because the cost of speculate() scales with
(a) tree size — dominated by prior trajectory tokens — and (b) suffix
context length — grows through a single generation. The previous
one-shot benchmark had an empty tree and a short context, which is why
it reported microsecond-level times.

Usage:
    python3 simulation/scripts/measure_suffix_cost.py \\
        --workloads specbench,bfcl_v4,swebench \\
        --model Qwen/Qwen3-14B \\
        --agent-results-dir simulation/results/qwen3_14b \\
        --output results/latency/suffix_cost.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


# Map workload name → candidate sub-directory names under --agent-results-dir.
WORKLOAD_DIR_CANDIDATES = {
    "specbench": ["specbench_steps2", "specbench", "specbench_req0-3"],
    "bfcl_v4":  ["bfcl_v4_steps2", "bfcl_v4", "bfcl_v4_req0-3"],
    "swebench":  ["swebench_steps2", "swebench", "swebench_req0-3"],
}


def _find_trajectory(base: Path, workload: str) -> Path | None:
    for cand in WORKLOAD_DIR_CANDIDATES.get(workload, []):
        p = base / cand / "trajectory.json"
        if p.exists():
            return p
    return None


def _extract_trajectories(trajectory_path: Path) -> list[dict]:
    """Read Stage 2's trajectory.json ({id: [token_ids...]}) and emit a
    list of dicts compatible with the Stage 3a cache-population API.

    Each trajectory is a flat token stream; we wrap it into a single
    per_call_tokens entry (prompt unknown).
    """
    with open(trajectory_path) as f:
        data = json.load(f)
    trajectories: list[dict] = []
    for tid, tokens in data.items():
        if not tokens:
            continue
        trajectories.append({
            "bfcl_id": tid,
            "per_call_tokens": [list(tokens)],
            "per_call_prompt_ids": [[]],
        })
    return trajectories


def _populate_cache(cache, trajectories: list[dict]) -> int:
    """Feed completed trajectories into the cache. Returns #tokens added."""
    total = 0
    rid = 100_000
    for traj in trajectories:
        for call_idx, tokens in enumerate(traj["per_call_tokens"]):
            if not tokens:
                continue
            pids_list = traj.get("per_call_prompt_ids") or []
            prompt = (np.array(pids_list[call_idx], dtype=np.int32)
                      if call_idx < len(pids_list) and pids_list[call_idx]
                      else np.array([], dtype=np.int32))
            cache.start_request(rid, prompt)
            cache.add_active_response(rid, tokens)
            cache.stop_request(rid)
            total += len(tokens)
            rid += 1
    return total


def _measure_trajectory(cache, traj: dict,
                        max_spec_tokens: int, max_spec_factor: float,
                        min_token_prob: float,
                        max_steps: int = 500) -> list[tuple[float, int]]:
    """Replay one trajectory, timing each speculate() call.

    Returns list of (speculate_ms, draft_size) per step.
    """
    req_id = 900_000
    timings: list[tuple[float, int]] = []

    for call_idx, tokens in enumerate(traj["per_call_tokens"]):
        N = len(tokens)
        if N == 0:
            continue
        pids_list = traj.get("per_call_prompt_ids") or []
        prompt = (np.array(pids_list[call_idx], dtype=np.int32)
                  if call_idx < len(pids_list) and pids_list[call_idx]
                  else np.array([], dtype=np.int32))
        cache.start_request(req_id, prompt)
        decoded: list[int] = []

        for pos in range(N):
            if len(timings) >= max_steps:
                break
            future = tokens[pos:]
            if len(future) <= 1:
                decoded.append(tokens[pos])
                cache.add_active_response(req_id, [tokens[pos]])
                continue
            if len(prompt) > 0:
                if decoded:
                    suffix_context = np.concatenate(
                        [prompt, np.array(decoded, dtype=np.int32)])
                else:
                    suffix_context = prompt.copy()
            else:
                suffix_context = np.array(decoded, dtype=np.int32)

            t0 = time.perf_counter()
            draft = cache.speculate(
                req_id, suffix_context,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=max_spec_factor,
                min_token_prob=min_token_prob,
                use_tree_spec=True,
            )
            t1 = time.perf_counter()
            timings.append(((t1 - t0) * 1000, len(draft.token_ids)))

            decoded.append(tokens[pos])
            cache.add_active_response(req_id, [tokens[pos]])

        cache.stop_request(req_id)
        req_id += 1
        if len(timings) >= max_steps:
            break

    return timings


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads", default="specbench,bfcl_v4,swebench")
    parser.add_argument("--model", required=True,
                        help="Model name (for reference only; agent_results "
                             "already carry the right trajectories)")
    parser.add_argument("--agent-results-dir", required=True,
                        help="Directory whose sub-dirs contain "
                             "agent_results_eagle3.json per workload "
                             "(e.g. simulation/results/qwen3_14b)")
    parser.add_argument("--max-spec-tokens", type=int, default=256)
    parser.add_argument("--max-spec-factor", type=float, default=4.0)
    parser.add_argument("--min-token-prob", type=float, default=0.0)
    parser.add_argument("--max-steps-per-workload", type=int, default=500,
                        help="Cap speculate() timing samples per workload")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from arctic_inference.suffix_decoding import SuffixDecodingCache

    base = Path(args.agent_results_dir)
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]

    results = []
    for w in workloads:
        ar = _find_trajectory(base, w)
        if ar is None:
            print(f"SKIP {w}: no trajectory.json under {base}", file=sys.stderr)
            continue
        trajectories = _extract_trajectories(ar)
        if len(trajectories) < 2:
            print(f"SKIP {w}: need ≥2 trajectories, got {len(trajectories)}",
                  file=sys.stderr)
            continue

        # Populate with ALL trajectories — in Stage 3a and Stage 6 the
        # cache builds up across the full replay, so the "held-out" setup
        # underestimates real speculate latency (tree lacks matches for
        # the measure target). Measuring on the longest trajectory after
        # populating with itself + peers approximates steady-state cost.
        cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
        tokens_added = _populate_cache(cache, trajectories)
        held = max(trajectories, key=lambda t: sum(len(c) for c in t["per_call_tokens"]))

        timings = _measure_trajectory(
            cache, held,
            max_spec_tokens=args.max_spec_tokens,
            max_spec_factor=args.max_spec_factor,
            min_token_prob=args.min_token_prob,
            max_steps=args.max_steps_per_workload,
        )
        if not timings:
            print(f"SKIP {w}: no speculate samples produced", file=sys.stderr)
            continue

        ms_values = [t[0] for t in timings]
        draft_sizes = [t[1] for t in timings]
        ms_values.sort()
        med = statistics.median(ms_values)
        p90 = ms_values[int(0.90 * (len(ms_values) - 1))]
        p99 = ms_values[int(0.99 * (len(ms_values) - 1))]

        entry = {
            "workload": w,
            "cache_populate_trajectories": len(trajectories),
            "cache_populate_tokens": tokens_added,
            "held_bfcl_id": held.get("bfcl_id"),
            "n_speculate_calls": len(timings),
            "speculate_ms": round(med, 4),
            "speculate_ms_p90": round(p90, 4),
            "speculate_ms_p99": round(p99, 4),
            "speculate_ms_mean": round(statistics.mean(ms_values), 4),
            "draft_size_mean": round(statistics.mean(draft_sizes), 2),
            "draft_size_max": int(max(draft_sizes)) if draft_sizes else 0,
        }
        results.append(entry)
        print(f"  {w:10s} pop={tokens_added:>6d}tok  n={len(timings):>4d}  "
              f"med={entry['speculate_ms']:.4f}ms  "
              f"p90={entry['speculate_ms_p90']:.4f}ms  "
              f"p99={entry['speculate_ms_p99']:.4f}ms  "
              f"draft_size_mean={entry['draft_size_mean']}",
              file=sys.stderr)

    output = {
        "params": {
            "max_spec_tokens": args.max_spec_tokens,
            "max_spec_factor": args.max_spec_factor,
            "min_token_prob": args.min_token_prob,
            "use_tree_spec": True,
            "source": "realistic replay — tree populated from agent_results",
        },
        "model": args.model,
        "agent_results_dir": str(Path(args.agent_results_dir).resolve()),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
