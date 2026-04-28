#!/usr/bin/env python3
"""Spider 2.0 DBT agent — mini-swe-agent style for AgenticSQL trajectories.

For each Spider2-DBT instance, the agent receives the natural-language
instruction and operates inside the instance directory (which contains
``dbt_project.yml``, ``profiles.yml``, ``models/``, and the seed
``<dbname>.duckdb`` file). The single tool is ``bash`` (mini-swe-agent
style); the agent submits by running ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``.

This is the workload used by the SuffixDecoding paper (`AgenticSQL`,
arxiv 2411.04975) where standalone suffix decoding hit 5.35× speedup
because SQL agent loops produce highly repetitive token streams.

Usage:
    python -m simulation.agents.spider2_dbt_agent \\
        --url http://localhost:30000/v1 \\
        --model Qwen/Qwen3-14B \\
        --input-file data/spider2_dbt/spider2-dbt.jsonl \\
        --output-file simulation/results/qwen3_14b/spider2_dbt_steps8_topk16_capture/agent_results_eagle3.json \\
        --num-requests 1 --resume \\
        --instances-dir data/spider2_dbt/instances \\
        --max-iterations 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow `python <file>` direct invocation in addition to ``python -m``
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import (  # noqa: E402
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_openai import ChatOpenAI  # noqa: E402

from simulation.agents.tools.swebench import (  # noqa: E402
    create_minisweagent_tools,
)
from simulation.pipeline.save_results import (  # noqa: E402
    append_to_checkpoint, done_ids, finalize_checkpoint, load_checkpoint,
)


# ---------------------------------------------------------------------------
# Prompts (mini-swe-agent style, adapted for Spider 2.0 DBT)
# ---------------------------------------------------------------------------

MINISWEAGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer shell "
    "to solve data analytics tasks using dbt and DuckDB."
)

INSTANCE_TEMPLATE = """<task_description>
Consider the following data analytics task:
{instruction}
</task_description>

You are working inside a dbt project directory. The project uses
DuckDB as the database backend. The directory contains:
- `dbt_project.yml`: dbt project configuration
- `profiles.yml`: connection profile (DuckDB path)
- `models/`: dbt models (SQL transformations)
- `<name>.duckdb`: the DuckDB database with seed data

Workflow guidance (you MUST follow):
1. Inspect the project layout (`ls`, `cat dbt_project.yml`,
   `cat profiles.yml`).
2. Look at existing models (`ls models/`, `cat models/<file>.sql`).
3. Inspect the database schema (`dbt run-operation list_tables` if
   available, or use `python3 -c "import duckdb; ..."` directly).
4. Write or modify SQL/dbt models to satisfy the task.
5. Run `dbt run` to execute the models. Iterate until success.
6. Verify outputs with SQL queries.

When you believe you have completed the task, run:
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""


# ---------------------------------------------------------------------------
# Single-request worker
# ---------------------------------------------------------------------------

def _serialize_messages(messages: list) -> list[dict]:
    """Same shape as swebench_agent — used by oracle log assembly later."""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"type": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"type": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"type": "ai", "content": m.content})
        elif isinstance(m, ToolMessage):
            out.append({"type": "tool", "content": m.content,
                        "tool_call_id": m.tool_call_id})
    return out


def process_request(
    llm,
    request: dict,
    instances_dir: str,
    max_iterations: int = 30,
    collect_oracle: bool = True,
) -> dict:
    """Run agent loop for one Spider 2.0 DBT instance, return result dict.

    Captures per-LLM-call oracle entries for downstream Stage 3 sim.
    """
    instance_id = request["instance_id"]
    instruction = request["instruction"]
    workdir = os.path.abspath(os.path.join(instances_dir, instance_id))

    if not os.path.isdir(workdir):
        return {
            "instance_id": instance_id,
            "category": request.get("type", ""),
            "error": f"workdir not found: {workdir}",
        }

    tools = create_minisweagent_tools(workdir, repo=instance_id)
    tool_map = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True)

    sys_prompt = MINISWEAGENT_SYSTEM_PROMPT
    first_user = INSTANCE_TEMPLATE.format(instruction=instruction)

    messages = [SystemMessage(sys_prompt), HumanMessage(first_user)]
    turns_with_messages: list[dict] = []
    all_steps: list[dict] = []

    t_start = time.perf_counter()

    try:
        for turn_idx in range(max_iterations):
            t0 = time.perf_counter()
            ai_msg = llm_with_tools.invoke(messages)
            latency = time.perf_counter() - t0
            messages.append(ai_msg)

            tool_calls_serialized = [
                {"name": tc["name"], "args": tc["args"], "id": tc["id"],
                 "type": "tool_call"}
                for tc in (ai_msg.tool_calls or [])
            ]
            turns_with_messages.append({
                "messages": _serialize_messages(messages),
                "tool_calls": tool_calls_serialized,
                "latency": latency,
                "response": ai_msg.content if isinstance(ai_msg.content, str)
                else "",
            })

            spec_decode = None
            if collect_oracle and hasattr(ai_msg, "response_metadata"):
                rm = ai_msg.response_metadata or {}
                spec_decode = rm.get("spec_decode")

            step = {
                "type": "llm",
                "turn": turn_idx,
                "step": turn_idx,
                "latency_s": latency,
                "content": ai_msg.content if isinstance(ai_msg.content, str)
                else "",
                "has_tool_calls": bool(ai_msg.tool_calls),
                "messages": _serialize_messages(messages[-1:]),
                "spec_decode": spec_decode,
            }
            all_steps.append(step)

            # Submit detection (same as mini-swe-agent in swebench_agent):
            #  * Empty tool_calls = soft submit (mini-swe-agent style)
            #  * COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT in command = explicit submit
            if not ai_msg.tool_calls:
                break

            submit_signal = False
            for tc in ai_msg.tool_calls:
                tool_name = tc["name"]
                try:
                    if tool_name in tool_map:
                        result = tool_map[tool_name].invoke(tc["args"])
                    else:
                        result = f"[ERROR] Unknown tool: {tool_name}"
                except Exception as e:
                    result = f"[ERROR] {e}"

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

                if (tool_name == "bash"
                        and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                        in str(tc["args"].get("command", ""))):
                    submit_signal = True

            if submit_signal:
                break

    except Exception as e:
        return {
            "instance_id": instance_id,
            "category": request.get("type", ""),
            "error": str(e),
            "turns": turns_with_messages,
            "agent_metrics": {"steps": all_steps},
        }

    total_latency = time.perf_counter() - t_start

    final_output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_output = msg.content
            break

    return {
        "instance_id": instance_id,
        "category": request.get("type", ""),
        "num_turns": len(turns_with_messages),
        "total_latency": total_latency,
        "output": final_output,
        "turns": turns_with_messages,
        "agent_metrics": {
            "steps": all_steps,
            "total_steps": len(all_steps),
        },
    }


# ---------------------------------------------------------------------------
# Dataset / runner
# ---------------------------------------------------------------------------

def load_spider2_dbt_dataset(path: str, num_requests: int | None = None
                             ) -> list[dict]:
    """Load Spider2-DBT instructions jsonl. Each line: {instance_id, instruction, type}."""
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            out.append(entry)
    if num_requests is not None:
        out = out[:num_requests]
    return out


def run_benchmark(
    url: str,
    model: str,
    input_file: str,
    output_file: str,
    instances_dir: str,
    num_requests: int | None = None,
    num_workers: int = 1,
    max_iterations: int = 30,
    resume: bool = False,
):
    requests = load_spider2_dbt_dataset(input_file, num_requests=None)

    pending: list[dict] = []
    if resume:
        cp = load_checkpoint(output_file)
        done = done_ids(cp, id_keys=("instance_id",))
        for r in requests:
            if str(r["instance_id"]) not in done:
                pending.append(r)
    else:
        pending = list(requests)

    if num_requests is not None:
        pending = pending[:num_requests]

    if not pending:
        print(f"All {len(requests)} requests already done. Nothing to run.")
        # finalize partial → final, in case still in partial state
        finalize_checkpoint(output_file)
        return

    print(f"Running {len(pending)} of {len(requests)} requests "
          f"(resume={resume}, instances_dir={instances_dir})")

    llm = ChatOpenAI(
        model=model, openai_api_base=url,
        openai_api_key="EMPTY",
        temperature=0.0,
        max_tokens=4096,
    )

    metadata = {
        "model": model,
        "url": url,
        "benchmark": "spider2_dbt",
        "num_requests": len(requests),
        "max_iterations": max_iterations,
        "oracle_enabled": True,
        "instances_dir": instances_dir,
    }

    def _one(req):
        return process_request(
            llm, req,
            instances_dir=instances_dir,
            max_iterations=max_iterations,
            collect_oracle=True,
        )

    if num_workers <= 1:
        for r in pending:
            res = _one(r)
            append_to_checkpoint(output_file, res, metadata=metadata)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = {ex.submit(_one, r): r for r in pending}
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as e:
                    r = futs[fut]
                    res = {"instance_id": r["instance_id"],
                           "error": f"worker exception: {e}"}
                append_to_checkpoint(output_file, res, metadata=metadata)

    # Defensive dedupe (in case repeat invocation captures same instance_id):
    # rewrite partial with unique-by-instance_id BEFORE finalize.
    cp = load_checkpoint(output_file)
    if cp and cp.get("questions"):
        seen = set()
        deduped = []
        for q in cp["questions"]:
            iid = q.get("instance_id")
            if iid in seen:
                continue
            seen.add(iid)
            deduped.append(q)
        if len(deduped) != len(cp["questions"]):
            print(f"NOTE: deduped {len(cp['questions'])} → {len(deduped)} "
                  f"questions before finalize")
            cp["questions"] = deduped
            from simulation.pipeline.save_results import (
                _atomic_write_json, checkpoint_path,
            )
            _atomic_write_json(cp, checkpoint_path(output_file))

    finalize_checkpoint(output_file, metadata=metadata)
    print(f"Done. Results in {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-file", required=True,
                    help="spider2-dbt.jsonl path")
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--instances-dir",
                    default="data/spider2_dbt/instances",
                    help="Directory containing per-instance dbt project dirs")
    ap.add_argument("--num-requests", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run_benchmark(
        url=args.url,
        model=args.model,
        input_file=args.input_file,
        output_file=args.output_file,
        instances_dir=args.instances_dir,
        num_requests=args.num_requests,
        num_workers=args.num_workers,
        max_iterations=args.max_iterations,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
