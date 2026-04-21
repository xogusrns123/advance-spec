"""Replay Stage 1 oracle results to collect MTP drafts or verify-tries p_t.

Reads stored messages from Stage 1 agent_results and sends them directly
to the SGLang server via /v1/chat/completions. No agent or rid matching needed.

Usage:
    # Stage 3: MTP replay
    python3 simulation/scripts/replay_oracle.py \
        --agent-results results/.../agent_results_eagle3.json \
        --output simulation/results/.../agent_results_mtp.json \
        --server-url http://localhost:30000

    # Stage 5: verify-tries p_t (server started with SGLANG_ORACLE_VERIFY_TRIES)
    python3 simulation/scripts/replay_oracle.py \
        --agent-results results/.../agent_results_eagle3.json \
        --output simulation/results/.../agent_results_verify.json \
        --server-url http://localhost:30000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

from simulation.oracle.oracle_patch import (
    get_oracle_log_position,
    read_oracle_log,
)


def extract_calls(agent_results_path: str) -> list[dict]:
    """Extract all LLM calls from Stage 1 results.

    Each call has: bfcl_id, call_idx, messages, output_tokens.
    """
    with open(agent_results_path) as f:
        data = json.load(f)

    calls = []
    for q in data["questions"]:
        bfcl_id = q.get("bfcl_id", q.get("instance_id", ""))
        for call_idx, step in enumerate(q.get("agent_metrics", {}).get("steps", [])):
            messages = step.get("messages")
            if messages is None:
                print(f"WARN: {bfcl_id} step {call_idx} missing messages, "
                      f"re-run Stage 1 with updated agent", file=sys.stderr)
                continue

            entries = step.get("spec_decode", {}).get("oracle_vanilla_entries", [])

            # Filter by req_id: with workers>1, entries from concurrent
            # requests may be interleaved in the oracle log.
            # Use the most common req_id as this call's actual rid.
            if entries:
                from collections import Counter
                rid_counts = Counter(e.get("req_id", "") for e in entries)
                primary_rid = rid_counts.most_common(1)[0][0]
                entries = [e for e in entries if e.get("req_id") == primary_rid]

            output_tokens = [e["tokens"][0][0] for e in entries if e.get("tokens")]

            calls.append({
                "bfcl_id": bfcl_id,
                "call_idx": call_idx,
                "messages": messages,
                "output_tokens": output_tokens,
                "n_tokens": len(output_tokens),
            })

    return calls


def build_trajectory(calls: list[dict], output_path: str) -> None:
    """Build trajectory.json from extracted calls.

    Keys are sequential — TrajectoryState's FIFO matching assigns them
    in order when replay_oracle sends requests sequentially.
    """
    trajectory = {}
    for i, call in enumerate(calls):
        trajectory[f"replay_{i:06d}"] = call["output_tokens"]

    with open(output_path, "w") as f:
        json.dump(trajectory, f)

    print(f"Trajectory: {len(trajectory)} calls", file=sys.stderr)


def _process_one_call(args):
    """Process a single LLM call. Used by both sequential and parallel modes."""
    call, client, model = args
    bfcl_id = call["bfcl_id"]
    n_output = call["n_tokens"]

    pos = get_oracle_log_position()

    t_start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=call["messages"],
            temperature=0.0,
            max_tokens=max(n_output + 10, 4096),
        )
    except Exception as e:
        return {"bfcl_id": bfcl_id, "call_idx": call["call_idx"],
                "error": str(e), "oracle_entries": [], "tokens": 0}

    latency = time.perf_counter() - t_start
    oracle_entries = read_oracle_log(pos)

    return {
        "bfcl_id": bfcl_id,
        "call_idx": call["call_idx"],
        "latency": latency,
        "tokens": response.usage.completion_tokens,
        "content": response.choices[0].message.content or "",
        "oracle_entries": oracle_entries,
    }


def replay(
    calls: list[dict],
    server_url: str,
    model: str,
    output_path: str,
    num_workers: int = 1,
) -> None:
    """Send stored messages and collect oracle entries."""
    client = OpenAI(base_url=f"{server_url}/v1", api_key="dummy")

    results_by_bfcl = {}
    total_oracle = 0
    total_tokens = 0
    t0 = time.time()

    args_list = [(call, client, model) for call in calls]

    if num_workers <= 1:
        results_iter = (_process_one_call(a) for a in args_list)
    else:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=num_workers)
        results_iter = executor.map(_process_one_call, args_list)

    for i, result in enumerate(results_iter):
        bfcl_id = result["bfcl_id"]
        oracle_entries = result["oracle_entries"]
        total_oracle += len(oracle_entries)
        total_tokens += result["tokens"]

        step_data = {
            "type": "llm",
            "step": result["call_idx"],
            "latency_s": result.get("latency", 0),
            "completion_tokens": result["tokens"],
            "content": result.get("content", ""),
        }
        if result.get("error"):
            step_data["error"] = result["error"]
        if oracle_entries:
            step_data["spec_decode"] = {
                "oracle_vanilla_entries": oracle_entries,
            }

        if bfcl_id not in results_by_bfcl:
            results_by_bfcl[bfcl_id] = {
                "bfcl_id": bfcl_id,
                "agent_metrics": {"steps": [], "mode": "replay"},
            }
        results_by_bfcl[bfcl_id]["agent_metrics"]["steps"].append(step_data)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(calls) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(calls)}] {bfcl_id[:30]} call {result['call_idx']}: "
              f"{result['tokens']} tok, {len(oracle_entries)} oracle, "
              f"{result.get('latency',0):.1f}s (ETA {eta:.0f}s)",
              file=sys.stderr)

    # Save
    questions = list(results_by_bfcl.values())
    output = {
        "metadata": {
            "num_requests": len(questions),
            "total_oracle_entries": total_oracle,
            "total_tokens": total_tokens,
            "mode": "replay_oracle",
        },
        "questions": questions,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"\nSaved: {output_path}", file=sys.stderr)
    print(f"  Questions: {len(questions)}, Oracle: {total_oracle}, "
          f"Tokens: {total_tokens}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-results", required=True,
                        help="Stage 1 agent_results.json (with messages per step)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--trajectory-output", default="/tmp/replay_trajectory.json")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Concurrent requests (use 1 for verify-tries mode)")
    args = parser.parse_args()

    calls = extract_calls(args.agent_results)
    print(f"Extracted {len(calls)} LLM calls", file=sys.stderr)

    if not calls:
        print("ERROR: No calls with stored messages found.", file=sys.stderr)
        sys.exit(1)

    build_trajectory(calls, args.trajectory_output)
    replay(calls, args.server_url, args.model, args.output,
           num_workers=args.num_workers)


if __name__ == "__main__":
    main()
