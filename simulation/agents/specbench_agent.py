"""
SpecBench / MT-Bench agent for SGLang oracle trajectory collection.

Multi-turn Q&A without tool calls. Sends each turn's user message,
collects the assistant response, and logs oracle draft entries.

Output format is compatible with _extract_online() in run_oracle_sim.py.

Usage:
    python3 -m simulation.agents.specbench_agent \
        --url http://localhost:30000/v1 \
        --model Qwen/Qwen3-8B \
        --input-file data/specbench/dataset.jsonl \
        --output-file results/qwen3_8b/specbench/agent_results_eagle3.json \
        --num-requests 80

    # MTP replay (Round 2):
    python3 -m simulation.agents.specbench_agent \
        --url http://localhost:30000/v1 \
        --model Qwen/Qwen3-8B \
        --input-file data/specbench/dataset.jsonl \
        --output-file results/qwen3_8b/specbench/agent_results_mtp.json \
        --replay results/qwen3_8b/specbench/agent_results_eagle3.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from simulation.oracle.oracle_patch import (
    clear_oracle_log,
    get_oracle_log_position,
    read_oracle_log,
    is_oracle_enabled,
)


def load_specbench_dataset(
    path: str,
    num_requests: int | None = None,
) -> list[dict]:
    """Load SpecBench/MT-Bench JSONL dataset."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if num_requests is not None:
        records = records[:num_requests]
    return records


def run_single_request(
    client: OpenAI,
    model: str,
    item: dict,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    collect_oracle: bool = True,
) -> dict:
    """Run multi-turn Q&A for one SpecBench question.

    Each turn: send user message → collect assistant response → log oracle.
    No tool calls, no agent loop.
    """
    question_id = item.get("question_id", "unknown")
    turns = item.get("turns", [])
    messages = []
    turns_data = []

    total_oracle = 0
    total_tokens = 0

    for turn_idx, user_msg in enumerate(turns):
        if collect_oracle:
            oracle_pos = get_oracle_log_position()

        messages.append({"role": "user", "content": user_msg})

        t_start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            turns_data.append({
                "response": "",
                "error": str(e),
            })
            continue

        latency = time.perf_counter() - t_start
        choice = response.choices[0]
        content = choice.message.content or ""
        messages.append({"role": "assistant", "content": content})

        completion_tokens = response.usage.completion_tokens
        total_tokens += completion_tokens

        turn_data = {
            "response": content,
            "latency_s": latency,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        if collect_oracle:
            oracle_entries = read_oracle_log(oracle_pos)
            if oracle_entries:
                turn_data["spec_decode"] = {
                    "oracle_vanilla_entries": oracle_entries,
                }
                total_oracle += len(oracle_entries)

        turns_data.append(turn_data)

    return {
        "question_id": question_id,
        "category": item.get("category", ""),
        "turns": turns_data,
        "total_oracle_entries": total_oracle,
        "total_tokens": total_tokens,
    }


def replay_single_request(
    client: OpenAI,
    model: str,
    item: dict,
    round1_question: dict,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> dict:
    """Replay Round 1 message history to collect MTP drafts.

    Reconstructs exact conversation from Round 1 and sends each turn's
    prompt to the MTP server. The oracle_replay patch forces tokens.
    """
    question_id = item.get("question_id", "unknown")
    turns = item.get("turns", [])
    r1_turns = round1_question.get("turns", [])
    messages = []
    turns_data = []
    total_oracle = 0

    for turn_idx, user_msg in enumerate(turns):
        oracle_pos = get_oracle_log_position()

        messages.append({"role": "user", "content": user_msg})

        t_start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            turns_data.append({"response": "", "error": str(e)})
            # Use Round 1 response to keep conversation history consistent
            if turn_idx < len(r1_turns):
                messages.append({"role": "assistant",
                                 "content": r1_turns[turn_idx].get("response", "")})
            continue

        latency = time.perf_counter() - t_start
        content = response.choices[0].message.content or ""

        # Use Round 1 response for conversation history (MTP might differ)
        if turn_idx < len(r1_turns):
            messages.append({"role": "assistant",
                             "content": r1_turns[turn_idx].get("response", "")})
        else:
            messages.append({"role": "assistant", "content": content})

        turn_data = {
            "response": content,
            "latency_s": latency,
            "completion_tokens": response.usage.completion_tokens,
        }

        oracle_entries = read_oracle_log(oracle_pos)
        if oracle_entries:
            turn_data["spec_decode"] = {
                "oracle_vanilla_entries": oracle_entries,
            }
            total_oracle += len(oracle_entries)

        turns_data.append(turn_data)

    return {
        "question_id": question_id,
        "category": item.get("category", ""),
        "turns": turns_data,
        "total_oracle_entries": total_oracle,
        "mode": "replay",
    }


def run_benchmark(
    url: str,
    model: str,
    input_file: str,
    output_file: str,
    num_requests: int | None = None,
    max_iterations: int = 1,  # unused, kept for CLI compat
    temperature: float = 0.0,
    max_tokens: int = 2048,
    replay_path: str | None = None,
    num_workers: int = 1,
) -> None:
    """Run SpecBench benchmark and save results."""
    collect_oracle = is_oracle_enabled()

    if collect_oracle:
        print("Oracle collection enabled (SGLANG_ORACLE_VANILLA=1)")
    else:
        print("Oracle collection disabled (set SGLANG_ORACLE_VANILLA=1 to enable)")

    dataset = load_specbench_dataset(input_file, num_requests)

    # Load Round 1 results for replay mode
    round1_map = {}
    if replay_path:
        print(f"REPLAY mode: following trajectory from {replay_path}")
        with open(replay_path) as f:
            r1_data = json.load(f)
        for q in r1_data["questions"]:
            qid = str(q.get("question_id", ""))
            round1_map[qid] = q
        print(f"  Round 1 questions: {len(round1_map)}")

    client = OpenAI(base_url=url, api_key="dummy")
    print(f"Running {len(dataset)} SpecBench requests against {url}"
          f" (workers={num_workers})")

    questions = []
    total_oracle = 0
    total_tokens = 0

    if replay_path:
        def _process(item):
            qid = str(item.get("question_id", ""))
            r1_q = round1_map.get(qid)
            if not r1_q:
                return None
            return replay_single_request(
                client, model, item, r1_q, temperature, max_tokens)
    else:
        def _process(item):
            return run_single_request(
                client, model, item, temperature, max_tokens, collect_oracle)

    if num_workers <= 1:
        results_iter = (_process(item) for item in dataset)
    else:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=num_workers)
        results_iter = executor.map(_process, dataset)

    for result in tqdm(results_iter, total=len(dataset), desc="SpecBench"):
        if result is None:
            continue
        questions.append(result)
        total_oracle += result.get("total_oracle_entries", 0)
        total_tokens += result.get("total_tokens", 0)

    # Save results
    output = {
        "metadata": {
            "model": model,
            "url": url,
            "benchmark": "specbench",
            "num_requests": len(questions),
            "total_oracle_entries": total_oracle,
            "total_tokens": total_tokens,
            "oracle_enabled": collect_oracle,
        },
        "questions": questions,
    }

    from simulation.pipeline.save_results import save_agent_results
    save_agent_results(output, output_file)

    print(f"\nResults saved to {output_file}")
    print(f"  Requests: {len(questions)}")
    print(f"  Oracle entries: {total_oracle}")
    print(f"  Total tokens: {total_tokens}")


def main():
    parser = argparse.ArgumentParser(
        description="SpecBench agent for SGLang oracle collection")
    parser.add_argument("--url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=1,
                        help="Unused, kept for CLI compatibility")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--replay", default=None,
                        help="Path to Round 1 results for MTP replay")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Concurrent requests to SGLang server")
    args = parser.parse_args()

    run_benchmark(
        url=args.url,
        model=args.model,
        input_file=args.input_file,
        output_file=args.output_file,
        num_requests=args.num_requests,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        replay_path=args.replay,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
