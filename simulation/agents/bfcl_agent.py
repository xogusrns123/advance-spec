"""
BFCL multi-turn benchmark runner for SGLang oracle trajectory collection.

Uses official BFCL tool execution (execute_multi_turn_func_call) and
decode_execute for accurate workload simulation. Collects per-step
EAGLE3 draft tokens from oracle log.

Usage:
    python3 -m simulation.agents.bfcl_agent \
        --url http://localhost:30000/v1 \
        --model zai-org/GLM-4.7-Flash \
        --input-file data/bfcl_multi_turn/dataset.jsonl \
        --output-file results/glm4_flash/oracle_vanilla/agent_results.json \
        --num-requests 80
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

import bfcl_eval
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.utils import (
    system_prompt_pre_processing_chat_model,
    default_decode_execute_prompting,
)
from bfcl_eval.constants.executable_backend_config import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
)
from bfcl_eval.utils import load_file

MULTI_TURN_FUNC_DOC_PATH = (
    Path(bfcl_eval.__file__).parent / "data" / "multi_turn_func_doc"
)

from simulation.oracle.oracle_patch import (
    clear_oracle_log,
    get_oracle_log_position,
    read_oracle_log,
    is_oracle_enabled,
)
from simulation.agents.tools.bfcl import patch_websearch_class, patch_websearch_in_globals, cleanup_globals

# Patch WebSearchAPI class BEFORE any instances are created
patch_websearch_class()


def load_bfcl_dataset(path: str, num_requests: int | None = None) -> list[dict]:
    """Load BFCL multi-turn dataset from JSONL."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if num_requests is not None:
        items = items[:num_requests]
    return items


def preprocess_bfcl_requests(requests: list[dict]) -> list[dict]:
    """Populate function docs from involved_classes (official BFCL logic).

    BFCL multi-turn data doesn't have a 'function' field — function docs
    are loaded dynamically from involved_classes via bfcl_eval package.
    """
    for entry in requests:
        if "bfcl_id" in entry and "id" not in entry:
            entry["id"] = entry["bfcl_id"]

        # Load function docs for all involved classes
        involved_classes = entry.get("involved_classes", [])
        entry["function"] = []
        for class_name in involved_classes:
            filename = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(class_name)
            if filename:
                func_docs = load_file(MULTI_TURN_FUNC_DOC_PATH / filename)
                entry["function"].extend(func_docs)

        # Handle missed_function: convert name → doc dict, remove from function list
        if "missed_function" in entry:
            missed = entry["missed_function"]
            for turn_index, missed_func_names in list(missed.items()):
                missed_docs = []
                for missed_func_name in missed_func_names:
                    for i, func_doc in enumerate(entry["function"]):
                        if func_doc["name"] == missed_func_name:
                            missed_docs.append(func_doc)
                            entry["function"].pop(i)
                            break
                missed[turn_index] = missed_docs

    return requests


def _format_prompt(messages: list[dict], functions: list[dict]) -> str:
    """Format messages + function docs into a single prompt string.

    Uses BFCL's official system_prompt_pre_processing_chat_model to
    embed function docs into the system prompt with [func(param=val)] format.
    """
    processed = system_prompt_pre_processing_chat_model(messages, functions)
    parts = []
    for msg in processed:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
        elif role == "tool":
            parts.append(f"<|tool|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _strip_thinking_tags(text: str) -> str:
    """Remove thinking content from model output.

    Handles multiple patterns:
    - <think>...</think> (standard)
    - Bare text...</think> (GLM-4.7-Flash: no opening tag)
    - Multiple </think> closings
    """
    # First: standard <think>...</think>
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Then: if </think> still remains (no opening <think>), take text after last </think>
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return text


def run_single_request(
    client: OpenAI,
    model: str,
    item: dict,
    max_iterations: int = 20,
    temperature: float = 0.0,
    collect_oracle: bool = True,
) -> dict:
    """Run a single BFCL multi-turn request with actual tool execution.

    Args:
        client: OpenAI client pointing to SGLang server.
        item: BFCL dataset item with 'id', 'question', 'function', etc.
        max_iterations: Max agent loop iterations per turn.
        temperature: Sampling temperature.
        collect_oracle: Whether to collect oracle vanilla entries.

    Returns:
        Result dict with agent_metrics containing full responses and oracle data.
    """
    bfcl_id = item.get("id", "unknown")
    question = item.get("question", [])
    functions = item.get("function", [])
    initial_config = item.get("initial_config", {})
    involved_classes = item.get("involved_classes", [])
    long_context = "long_context" in item.get("category", "")

    messages = []
    all_steps = []
    model_name_safe = model.replace("/", "_")

    for turn_idx, turn_messages in enumerate(question):
        # Add user turn messages
        if isinstance(turn_messages, list):
            messages.extend(turn_messages)
        elif isinstance(turn_messages, dict):
            messages.append(turn_messages)
        else:
            messages.append({"role": "user", "content": str(turn_messages)})

        # Agent loop for this turn
        for step in range(max_iterations):
            step_data = {
                "type": "llm",
                "turn": turn_idx,
                "step": step,
            }

            if collect_oracle:
                oracle_pos = get_oracle_log_position()

            # Format prompt with BFCL system prompt
            formatted_messages = system_prompt_pre_processing_chat_model(
                messages, functions, bfcl_id
            )

            # API call
            t_start = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=4096,
                )
            except Exception as e:
                step_data["error"] = str(e)
                all_steps.append(step_data)
                break

            latency = time.perf_counter() - t_start

            # Parse response
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason

            step_data["latency_s"] = latency
            step_data["prompt_tokens"] = response.usage.prompt_tokens
            step_data["completion_tokens"] = response.usage.completion_tokens
            step_data["content"] = content
            step_data["finish_reason"] = finish_reason
            step_data["messages"] = copy.deepcopy(formatted_messages)

            # Collect oracle entries
            if collect_oracle:
                oracle_entries = read_oracle_log(oracle_pos)
                if oracle_entries:
                    step_data["spec_decode"] = {
                        "oracle_vanilla_entries": oracle_entries,
                    }

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": content})

            # Decode model output to BFCL function call strings
            try:
                text_to_decode = _strip_thinking_tags(content)
                decoded_calls = default_decode_execute_prompting(text_to_decode)
            except Exception as e:
                step_data["decode_error"] = str(e)
                all_steps.append(step_data)
                break

            # Check if no tool calls → end turn
            if is_empty_execute_response(decoded_calls):
                step_data["action"] = "end_of_turn"
                all_steps.append(step_data)
                break

            step_data["decoded_calls"] = decoded_calls
            step_data["has_tool_calls"] = True

            # Execute tool calls via official BFCL executor
            t_exec_start = time.perf_counter()
            try:
                exec_results, _ = execute_multi_turn_func_call(
                    decoded_calls,
                    initial_config,
                    involved_classes,
                    model_name_safe,
                    bfcl_id,
                    long_context=long_context,
                    is_evaL_run=False,
                )
                # Patch WebSearchAPI with DuckDuckGo (free, no SerpAPI key)
                patch_websearch_in_globals(bfcl_id)
            except Exception as e:
                step_data["exec_error"] = str(e)
                all_steps.append(step_data)
                break

            exec_latency = time.perf_counter() - t_exec_start
            step_data["exec_latency_s"] = exec_latency

            # Store execution results
            exec_results_str = [str(r) for r in exec_results]
            step_data["exec_results"] = exec_results_str

            all_steps.append(step_data)

            # Add tool results to message history
            for exec_result in exec_results:
                messages.append({
                    "role": "tool",
                    "content": str(exec_result),
                })

    # Clean up globals to prevent memory leak
    cleanup_globals(bfcl_id)

    return {
        "bfcl_id": bfcl_id,
        "category": item.get("category", ""),
        "agent_metrics": {
            "steps": all_steps,
            "total_turns": len(question),
            "total_steps": len(all_steps),
        },
    }


def replay_single_request(
    client: OpenAI,
    model: str,
    item: dict,
    round1_question: dict,
    temperature: float = 0.0,
) -> dict:
    """Replay a Round 1 request to collect MTP drafts on the same trajectory.

    Instead of running the agent loop, reconstructs the exact message
    history from Round 1 and sends each step's prompt to the MTP server.
    The server's oracle_replay patch forces the trajectory tokens.
    """
    bfcl_id = item.get("id", "unknown")
    functions = item.get("function", [])
    round1_steps = round1_question["agent_metrics"]["steps"]

    messages = []
    all_steps = []

    for i, r1_step in enumerate(round1_steps):
        step_data = {
            "type": "llm",
            "turn": r1_step.get("turn", 0),
            "step": r1_step.get("step", i),
        }

        # Reconstruct messages up to this step from Round 1
        # For the first step, build from dataset question
        # For subsequent steps, append Round 1's content + tool results
        if i == 0:
            # Build initial messages from dataset
            question = item.get("question", [])
            for turn_msgs in question:
                if isinstance(turn_msgs, list):
                    messages.extend(turn_msgs)
                    break
                elif isinstance(turn_msgs, dict):
                    messages.append(turn_msgs)
                    break
        else:
            # Append previous step's assistant response + tool results
            prev_step = round1_steps[i - 1]
            prev_content = prev_step.get("content", "")
            messages.append({"role": "assistant", "content": prev_content})

            # Add tool execution results if any
            exec_results = prev_step.get("exec_results", [])
            for exec_result in exec_results:
                messages.append({"role": "tool", "content": str(exec_result)})

            # If new turn started, add user message
            if r1_step.get("turn", 0) != prev_step.get("turn", 0):
                turn_idx = r1_step["turn"]
                question = item.get("question", [])
                if turn_idx < len(question):
                    turn_msgs = question[turn_idx]
                    if isinstance(turn_msgs, list):
                        messages.extend(turn_msgs)
                    elif isinstance(turn_msgs, dict):
                        messages.append(turn_msgs)

        oracle_pos = get_oracle_log_position()

        # Format and send to MTP server
        formatted_messages = system_prompt_pre_processing_chat_model(
            messages, functions, bfcl_id
        )

        t_start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=4096,
            )
        except Exception as e:
            step_data["error"] = str(e)
            all_steps.append(step_data)
            continue

        latency = time.perf_counter() - t_start

        choice = response.choices[0]
        content = choice.message.content or ""

        step_data["latency_s"] = latency
        step_data["prompt_tokens"] = response.usage.prompt_tokens
        step_data["completion_tokens"] = response.usage.completion_tokens
        step_data["content"] = content

        # Collect oracle entries (MTP drafts)
        oracle_entries = read_oracle_log(oracle_pos)
        if oracle_entries:
            step_data["spec_decode"] = {
                "oracle_vanilla_entries": oracle_entries,
            }

        all_steps.append(step_data)

    return {
        "bfcl_id": bfcl_id,
        "category": item.get("category", ""),
        "agent_metrics": {
            "steps": all_steps,
            "total_turns": round1_question["agent_metrics"].get("total_turns", 0),
            "total_steps": len(all_steps),
            "mode": "replay",
        },
    }


def run_benchmark(
    url: str,
    model: str,
    input_file: str,
    output_file: str,
    num_requests: int | None = None,
    max_iterations: int = 20,
    temperature: float = 0.0,
    replay: str | None = None,
    num_workers: int = 1,
    resume: bool = False,
) -> dict:
    """Run full BFCL benchmark and save results.

    Args:
        replay: Path to Round 1 agent_results.json for replay mode.
                If set, replays the exact message history from Round 1
                to collect MTP drafts on the same trajectory.
        resume: When True, skip requests already present in the
                output file's checkpoint partial; ``--num-requests`` then
                applies AFTER the resume filter (round-robin coordinator
                pattern).
    """
    client = OpenAI(base_url=url, api_key="dummy")
    # Load full dataset; --num-requests applied AFTER resume filter so the
    # round-robin coordinator can advance one new request per invocation.
    dataset = load_bfcl_dataset(input_file, None)
    dataset = preprocess_bfcl_requests(dataset)
    collect_oracle = is_oracle_enabled()

    # Load Round 1 results for replay mode
    round1_data = None
    if replay:
        with open(replay) as f:
            round1_data = json.load(f)
        print(f"REPLAY mode: following trajectory from {replay}")
        print(f"  Round 1 questions: {len(round1_data['questions'])}")

    if collect_oracle:
        print("Oracle collection enabled (SGLANG_ORACLE_VANILLA=1)")
    else:
        print("Oracle collection disabled (set SGLANG_ORACLE_VANILLA=1 to enable)")

    # Resume support: skip requests already in the checkpoint partial.
    from simulation.pipeline.save_results import (
        load_checkpoint, append_to_checkpoint, finalize_checkpoint,
    )
    cp = load_checkpoint(output_file) if resume else None
    done = set()
    results: list = []
    total_oracle_entries = 0
    total_tokens = 0
    total_tool_calls = 0
    if cp:
        results = list(cp.get("questions", []))
        for q in results:
            done.add(str(q.get("bfcl_id", q.get("id", ""))))
            for s in q.get("agent_metrics", {}).get("steps", []):
                total_tokens += s.get("completion_tokens", 0)
                total_oracle_entries += len(
                    s.get("spec_decode", {}).get("oracle_vanilla_entries", []))
                if s.get("has_tool_calls"):
                    total_tool_calls += 1
        print(f"RESUME: {len(done)} requests already done; skipping them")

    pending = [r for r in dataset
               if str(r.get("bfcl_id", r.get("id", ""))) not in done]
    if num_requests is not None:
        pending = pending[:num_requests]
    print(f"Running {len(pending)}/{len(dataset)} BFCL requests against {url}"
          f" (workers={num_workers})")

    if replay and round1_data:
        # Replay path doesn't get resume support (single-shot trajectory
        # follow); use the full dataset paired with Round 1 results.
        pairs = list(zip(dataset, round1_data["questions"]))

        def _process_replay(pair):
            item, round1_q = pair
            return replay_single_request(
                client=client, model=model, item=item,
                round1_question=round1_q, temperature=temperature)

        if num_workers <= 1:
            results_iter = (_process_replay(p) for p in pairs)
        else:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=num_workers)
            results_iter = executor.map(_process_replay, pairs)
        iter_total = len(pairs)
    else:
        def _process_normal(item):
            return run_single_request(
                client=client, model=model, item=item,
                max_iterations=max_iterations, temperature=temperature,
                collect_oracle=collect_oracle)

        if num_workers <= 1:
            results_iter = (_process_normal(item) for item in pending)
        else:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=num_workers)
            results_iter = executor.map(_process_normal, pending)
        iter_total = len(pending)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _meta():
        return {
            "model": model,
            "url": url,
            "num_requests": len(results),
            "total_oracle_entries": total_oracle_entries,
            "total_tokens": total_tokens,
            "total_tool_calls": total_tool_calls,
            "oracle_enabled": collect_oracle,
        }

    for result in tqdm(results_iter, total=iter_total, desc="BFCL"):
        results.append(result)

        for step in result["agent_metrics"]["steps"]:
            sd = step.get("spec_decode", {})
            entries = sd.get("oracle_vanilla_entries", [])
            total_oracle_entries += len(entries)
            total_tokens += step.get("completion_tokens", 0)
            if step.get("has_tool_calls"):
                total_tool_calls += 1

        # Per-question checkpoint append (rr coordinator can resume from here)
        append_to_checkpoint(output_path, result, _meta())

    # Finalize: rename .partial → final
    finalize_checkpoint(output_path, _meta())

    print(f"\nResults saved to {output_file}")
    print(f"  Requests: {len(results)}")
    print(f"  Oracle entries: {total_oracle_entries}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Tool calls: {total_tool_calls}")

    return {"metadata": _meta(), "questions": results}


def main():
    parser = argparse.ArgumentParser(
        description="BFCL agent for SGLang oracle collection"
    )
    parser.add_argument("--url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--replay", default=None,
                        help="Path to Round 1 agent_results.json for replay mode")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Concurrent requests to SGLang server")
    parser.add_argument("--resume", action="store_true",
                        help="Skip requests already in the checkpoint partial.")
    args = parser.parse_args()

    run_benchmark(
        url=args.url,
        model=args.model,
        input_file=args.input_file,
        output_file=args.output_file,
        num_requests=args.num_requests,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        replay=args.replay,
        num_workers=args.num_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
