"""
BFCLv4 agentic benchmark runner (web_search + memory) for SGLang oracle collection.

Ported from ~/agentic-bench/bench/bfcl_v4_agent.py.

Key differences from bfcl_agent.py (v3 multi-turn):
  - Single turn, multi-step (not multi-turn)
  - Memory prereq conversations run sequentially before test entries
  - initial_config populated at runtime via bfcl_eval helpers
  - WebSearchAPI patched with DuckDuckGo (free, no SerpAPI key)

Usage:
    python3 -m hybrid_spec_decoding.analysis.bfcl_v4_agent \
        --url http://localhost:30000/v1 \
        --model zai-org/GLM-4.7-Flash \
        --input-file data/bfcl_agent/dataset.jsonl \
        --output-file results/glm4_flash/bfclv4/agent_results_eagle3.json \
        --num-requests 5
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

# BFCL official imports
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.utils import (
    system_prompt_pre_processing_chat_model,
    default_decode_execute_prompting,
)
try:
    from bfcl_eval.model_handler.utils import add_memory_instruction_system_prompt
except ImportError:
    add_memory_instruction_system_prompt = None

try:
    from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
except ImportError:
    MAXIMUM_STEP_LIMIT = 20

from bfcl_eval.utils import (
    load_dataset_entry,
    extract_test_category_from_id,
    sort_key,
)
try:
    from bfcl_eval.utils import (
        is_memory,
        is_memory_prereq,
        is_web_search,
        populate_initial_settings_for_memory_test_cases,
        populate_initial_settings_for_web_search_test_cases,
    )
except ImportError:
    is_memory = lambda x: "memory" in str(x)
    is_memory_prereq = lambda x: "prereq" in str(x)
    is_web_search = lambda x: "web_search" in str(x)
    populate_initial_settings_for_memory_test_cases = None
    populate_initial_settings_for_web_search_test_cases = None

from bfcl_eval.constants.category_mapping import AGENTIC_CATEGORY

from ..sglang_integration.oracle_patch import (
    clear_oracle_log,
    read_oracle_log,
    is_oracle_enabled,
)
from .tools.bfcl import patch_websearch_in_globals, cleanup_globals


def _strip_thinking(text: str) -> str:
    """Strip thinking content.

    Handles:
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


def load_bfcl_v4_dataset(
    path: str, num_requests: int | None = None,
) -> list[dict]:
    """Load BFCLv4 agentic dataset (JSONL)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if num_requests is not None:
        # Sample from non-prereq entries, keep required prereqs
        from collections import defaultdict
        by_cat = defaultdict(list)
        for r in records:
            cat = r.get("category", "")
            if "prereq" not in r.get("id", ""):
                by_cat[cat].append(r)
        cats = sorted(by_cat.keys())
        per_cat = max(1, num_requests // len(cats)) if cats else num_requests
        sampled = []
        for cat in cats:
            sampled.extend(by_cat[cat][:per_cat])
        sampled = sampled[:num_requests]
        # Add required prereqs
        sampled_ids = {r.get("id", "") for r in sampled}
        needed_deps = set()
        for r in sampled:
            for dep in r.get("depends_on", []):
                if dep not in sampled_ids:
                    needed_deps.add(dep)
        id_to_req = {r.get("id", ""): r for r in records}
        for dep_id in needed_deps:
            if dep_id in id_to_req:
                sampled.append(id_to_req[dep_id])
        sampled.sort(key=sort_key)
        records = sampled
    return records


def process_request(
    client: OpenAI,
    model: str,
    request: dict,
    max_iterations: int,
    collect_oracle: bool = True,
) -> dict:
    """Process a single BFCLv4 agentic request.

    Single-turn, multi-step: the model iteratively calls tools
    until it produces a final text answer.
    """
    bfcl_id = request.get("bfcl_id", request.get("id", ""))
    entry_id = request.get("id", bfcl_id)
    category = request.get("category", "")
    test_category = extract_test_category_from_id(entry_id)

    functions = request.get("function", [])
    initial_config = request.get("initial_config", {})
    involved_classes = request.get("involved_classes", [])
    scenario = request.get("scenario", "")

    model_name_safe = f"bench_v4_{entry_id}"

    # Prepare conversation
    conversation_turns = request.get("question", [])
    if not conversation_turns:
        return {
            "bfcl_id": bfcl_id, "category": category,
            "error": "No conversation turns",
            "agent_metrics": {"steps": []},
        }

    # Initialize class instances (empty call to create instances)
    cleanup_globals(entry_id)
    _, involved_instances = execute_multi_turn_func_call(
        [], initial_config, involved_classes,
        model_name_safe, entry_id,
        long_context=False, is_evaL_run=False,
    )

    # Memory: inject system prompt
    all_turn_messages = copy.deepcopy(conversation_turns)
    if is_memory(test_category) and involved_instances and add_memory_instruction_system_prompt:
        memory_instance = list(involved_instances.values())[0]
        all_turn_messages = add_memory_instruction_system_prompt(
            all_turn_messages, test_category, scenario, memory_instance,
        )

    # Web search: patch with DuckDuckGo
    if is_web_search(test_category):
        patch_websearch_in_globals(entry_id)

    # Build messages with function docs
    all_turn_messages[0] = system_prompt_pre_processing_chat_model(
        all_turn_messages[0], functions, entry_id
    )
    # Append efficiency instruction to system prompt
    all_turn_messages[0][0]["content"] += (
        "\n\nIMPORTANT: Be efficient. Use the minimum number of function calls needed. "
        "Once you have enough information to answer, respond with the final answer immediately "
        "instead of making additional searches. Do NOT over-verify or repeat similar queries."
    )
    messages = []
    for turn_msgs in all_turn_messages:
        messages.extend(turn_msgs)

    # Agent step loop
    all_steps = []
    model_result_steps = []
    t_start = time.perf_counter()

    for step in range(max_iterations):
        if collect_oracle:
            clear_oracle_log()

        # LLM call
        t_llm = time.perf_counter()
        try:
            formatted_prompt = system_prompt_pre_processing_chat_model(
                messages, functions, entry_id
            ) if not messages else messages
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
            )
        except Exception as e:
            all_steps.append({"type": "llm", "step": step, "error": str(e)})
            break

        latency = time.perf_counter() - t_llm
        content = response.choices[0].message.content or ""

        step_data = {
            "type": "llm",
            "step": step,
            "latency_s": latency,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "content": content,
        }

        # Collect oracle entries
        if collect_oracle:
            oracle_entries = read_oracle_log()
            if oracle_entries:
                step_data["spec_decode"] = {
                    "oracle_vanilla_entries": oracle_entries,
                }

        messages.append({"role": "assistant", "content": content})

        # Decode response
        try:
            text_to_decode = _strip_thinking(content)
            decoded_calls = default_decode_execute_prompting(text_to_decode)
        except Exception:
            # Decode failure = final text answer
            all_steps.append(step_data)
            break

        if is_empty_execute_response(decoded_calls):
            all_steps.append(step_data)
            break

        step_data["decoded_calls"] = decoded_calls
        step_data["has_tool_calls"] = True
        model_result_steps.append(decoded_calls)

        # Execute tool calls
        t_exec = time.perf_counter()
        try:
            exec_results, involved_instances = execute_multi_turn_func_call(
                decoded_calls, initial_config, involved_classes,
                model_name_safe, entry_id,
                long_context=False, is_evaL_run=False,
            )
            if is_web_search(test_category):
                patch_websearch_in_globals(entry_id)
        except Exception as e:
            step_data["exec_error"] = str(e)
            all_steps.append(step_data)
            break

        step_data["exec_latency_s"] = time.perf_counter() - t_exec
        step_data["exec_results"] = [str(r) for r in exec_results]
        all_steps.append(step_data)

        # Add tool results to messages
        for exec_result in exec_results:
            messages.append({
                "role": "tool",
                "content": str(exec_result),
            })

    cleanup_globals(entry_id)

    return {
        "bfcl_id": bfcl_id,
        "category": category,
        "agent_metrics": {
            "steps": all_steps,
            "total_steps": len(all_steps),
        },
    }


def run_benchmark(
    url: str,
    model: str,
    input_file: str,
    output_file: str,
    num_requests: int | None = None,
    max_iterations: int = 10,
) -> None:
    """Run BFCLv4 agentic benchmark."""
    collect_oracle = is_oracle_enabled()
    if collect_oracle:
        print("Oracle collection enabled (SGLANG_ORACLE_VANILLA=1)")
    else:
        print("Oracle collection disabled")

    dataset = load_bfcl_v4_dataset(input_file, num_requests)

    # Populate initial_config at runtime
    print("Populating initial configs...")
    output_dir = Path(output_file).parent
    if populate_initial_settings_for_memory_test_cases:
        mem_dir = output_dir / "memory_snapshots"
        mem_dir.mkdir(parents=True, exist_ok=True)
        dataset = populate_initial_settings_for_memory_test_cases(dataset, mem_dir)
    if populate_initial_settings_for_web_search_test_cases:
        dataset = populate_initial_settings_for_web_search_test_cases(dataset)

    dataset.sort(key=sort_key)
    print(f"Running {len(dataset)} BFCLv4 requests against {url}")

    client = OpenAI(base_url=url, api_key="dummy")
    questions = []
    total_oracle = 0
    total_tokens = 0

    for request in tqdm(dataset, desc="BFCLv4"):
        result = process_request(
            client, model, request, max_iterations, collect_oracle)
        questions.append(result)
        for s in result.get("agent_metrics", {}).get("steps", []):
            total_tokens += s.get("completion_tokens", 0)
            total_oracle += len(
                s.get("spec_decode", {}).get("oracle_vanilla_entries", []))

    output = {
        "metadata": {
            "model": model,
            "url": url,
            "benchmark": "bfcl_v4",
            "num_requests": len(questions),
            "total_oracle_entries": total_oracle,
            "total_tokens": total_tokens,
            "oracle_enabled": collect_oracle,
        },
        "questions": questions,
    }

    from .save_results import save_agent_results
    save_agent_results(output, output_file)

    print(f"\nResults saved to {output_file}")
    print(f"  Requests: {len(questions)}")
    print(f"  Oracle entries: {total_oracle}")
    print(f"  Total tokens: {total_tokens}")


def main():
    parser = argparse.ArgumentParser(
        description="BFCLv4 agent for SGLang oracle collection")
    parser.add_argument("--url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=10)
    args = parser.parse_args()

    run_benchmark(
        url=args.url,
        model=args.model,
        input_file=args.input_file,
        output_file=args.output_file,
        num_requests=args.num_requests,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
