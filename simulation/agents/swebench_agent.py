"""
SWE-Bench LangChain agent for SGLang oracle trajectory collection.

Uses the same LangChain tool-calling agent loop as ~/agentic-bench/bench/agent.py,
ported to work with SGLang's oracle collection (oracle_patch.py).

Tools from tools/swebench.py provide real repo operations:
  bash, file_view, file_read, file_write, file_str_replace, search

Repository management:
  - Pre-cloned repos in --repos-dir/{instance_id}/
  - git reset --hard base_commit before each instance
  - git clean -fd after completion

Usage:
    python3 -m simulation.agents.swebench_agent \
        --url http://localhost:30000/v1 \
        --model Qwen/Qwen3-8B \
        --input-file data/swebench/dataset.jsonl \
        --output-file results/qwen3_8b/swebench/agent_results_eagle3.json \
        --repos-dir data/swebench/repos \
        --max-iterations 15
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from simulation.oracle.oracle_patch import (
    clear_oracle_log,
    get_oracle_log_position,
    read_oracle_log,
    is_oracle_enabled,
)
from simulation.agents.tools.swebench import (
    create_swebench_tools,
    create_sweagent_tools,
    create_minisweagent_tools,
)


# ---------------------------------------------------------------------------
# System prompts (from agentic-bench/bench/agent.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert software engineer. Fix the GitHub issue in the provided repository.\n\n"
    "IMPORTANT: The current working directory is already the root of the repository. "
    "All file paths should be relative to this root.\n\n"
    "TOOLS:\n"
    "- bash(command): Execute shell commands\n"
    "- file_view(path, view_range): View file or directory\n"
    "- file_read(path, start_line, end_line): Read file with line range\n"
    "- file_write(path, content): Write file (overwrites entire file)\n"
    "- file_str_replace(path, old_str, new_str): Replace string in file (preferred for edits)\n"
    "- search(pattern, path): Search for patterns\n\n"
    "STRATEGY:\n"
    "1. First run 'bash ls' to see the repository structure\n"
    "2. Use search or bash grep to find relevant files\n"
    "3. Read and understand the relevant code thoroughly\n"
    "4. Implement a minimal, targeted fix using file_str_replace\n"
    "5. Use 'bash git diff' to verify changes are correct and minimal\n\n"
    "RULES:\n"
    "- NEVER modify a file before reading it first\n"
    "- Use file_str_replace for targeted edits, file_write ONLY for new files\n"
    "- Make minimal, focused changes\n"
)


# mini-swe-agent style: copies the official benchmarks/swebench.yaml
# templates verbatim. The system_template is intentionally one line; all
# task-specific guidance lives in the instance_template that becomes the
# first HumanMessage. The {{task}} placeholder is replaced with the PR
# description / issue body at runtime.
MINISWEAGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer shell "
    "to solve programming tasks."
)

MINISWEAGENT_INSTANCE_TEMPLATE = """<pr_description>
Consider the following PR description:
{task}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue AT LEAST ONE command, see the result, then think and issue your next command(s).</important>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide one or more bash tool calls to execute

## Important Boundaries

- MODIFY: Regular source code files in the working directory
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call. You can make MULTIPLE tool calls in a single response when the commands are independent (e.g., searching multiple files, reading different parts of the codebase).
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

Example of a CORRECT response:
<example_response>
I need to understand the Builder-related code. Let me find relevant files and check the project structure.

[Makes multiple bash tool calls: {"command": "ls -la"}, {"command": "find src -name '*.java' | grep -i builder"}, {"command": "cat README.md | head -50"}]
</example_response>

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- You can use bash commands or invoke any tool that is available in the environment
- You can also create new tools or scripts to help you with the task
- If a tool isn't available, you can also install it

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to fix the issue.
Do not submit file creations or changes to any of the following files:

- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
- If you modify patch.txt after verifying, you SHOULD verify again before submitting.
- You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>
</instructions>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_messages(messages: list) -> list[dict]:
    """Convert LangChain messages to serializable dicts."""
    result = []
    for msg in messages:
        entry = {"type": type(msg).__name__.replace("Message", "").lower()}
        if hasattr(msg, "content"):
            entry["content"] = msg.content
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        if hasattr(msg, "tool_call_id"):
            entry["tool_call_id"] = msg.tool_call_id
        result.append(entry)
    return result


def _setup_repo(workdir: str, repo: str, base_commit: str) -> str | None:
    """Setup repository to base_commit state. Returns error string or None."""
    git_dir = os.path.join(workdir, ".git")

    if os.path.isdir(git_dir):
        try:
            subprocess.run(
                ["git", "reset", "--hard", base_commit],
                cwd=workdir, capture_output=True, timeout=30, check=True,
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=workdir, capture_output=True, timeout=30, check=True,
            )
            return None
        except subprocess.CalledProcessError:
            try:
                shutil.rmtree(workdir)
            except Exception:
                pass

    if not os.path.isdir(git_dir):
        try:
            repo_url = f"https://github.com/{repo}.git"
            subprocess.run(
                ["git", "clone", repo_url, workdir],
                capture_output=True, timeout=300, check=True,
            )
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=workdir, capture_output=True, timeout=30, check=True,
            )
            return None
        except Exception as e:
            return f"Failed to setup repo {repo}: {e}"

    return None


def _cleanup_repos(repos_dir: str, base_commits: dict[str, str] | None = None):
    """Reset all repos to their base_commit state."""
    repos_path = Path(repos_dir)
    if not repos_path.exists():
        return

    cleaned = 0
    for d in repos_path.iterdir():
        if not d.is_dir() or not (d / ".git").exists():
            continue
        try:
            target = "HEAD"
            if base_commits and d.name in base_commits:
                target = base_commits[d.name]
            subprocess.run(["git", "reset", "--hard", target],
                           cwd=d, capture_output=True, timeout=10)
            subprocess.run(["git", "clean", "-fd"],
                           cwd=d, capture_output=True, timeout=10)
            cleaned += 1
        except Exception as e:
            print(f"  WARN: cleanup failed for {d.name}: {e}")

    print(f"  Cleaned {cleaned} repositories")


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

def process_request(
    llm,
    request: dict,
    repos_dir: str,
    max_iterations: int,
    collect_oracle: bool = True,
    tool_style: str = "full",
) -> dict:
    """Run tool-calling agent loop for one SWE-bench instance.

    Ported from agentic-bench/bench/agent.py with SGLang oracle collection.
    """
    instance_id = request.get("instance_id", "")
    workdir = os.path.join(repos_dir, instance_id)
    base_commit = request.get("base_commit")
    repo = request.get("repo")

    if not (repo and base_commit):
        return {
            "instance_id": instance_id,
            "category": request.get("category", ""),
            "error": "Missing repo or base_commit",
            "turns": [],
            "agent_metrics": {"steps": []},
        }

    # Setup repo
    err = _setup_repo(workdir, repo, base_commit)
    if err:
        return {
            "instance_id": instance_id,
            "category": request.get("category", ""),
            "error": err,
            "turns": [],
            "agent_metrics": {"steps": []},
        }

    # Create tools
    if tool_style == "sweagent":
        tools = create_sweagent_tools(workdir, repo=repo)
    elif tool_style == "minisweagent":
        tools = create_minisweagent_tools(workdir, repo=repo)
    else:
        tools = create_swebench_tools(workdir, repo=repo)
    tool_map = {t.name: t for t in tools}
    # mini-swe-agent's official config sets parallel_tool_calls=True; keep
    # serial calls for the other styles since their tool sets weren't
    # designed for that.
    parallel_tool_calls = (tool_style == "minisweagent")
    llm_with_tools = llm.bind_tools(
        tools, parallel_tool_calls=parallel_tool_calls)

    # Initialize messages — pick prompt template to match the active tool
    # set. mini-swe-agent splits role (system) from task instructions
    # (instance template wrapped around the PR description).
    if tool_style == "minisweagent":
        sys_prompt = MINISWEAGENT_SYSTEM_PROMPT
        # Use replace, not str.format — the template contains JSON-like
        # examples (e.g. {"command": "ls"}) that would otherwise be parsed
        # as format placeholders.
        first_user = MINISWEAGENT_INSTANCE_TEMPLATE.replace(
            "{task}", str(request["turns"][0]))
    else:
        sys_prompt = SYSTEM_PROMPT
        first_user = request["turns"][0]
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=first_user),
    ]

    t_start = time.perf_counter()
    all_steps = []
    turns_with_messages = []

    try:
        for iteration in range(max_iterations):
            if collect_oracle:
                oracle_pos = get_oracle_log_position()

            # LLM call
            t_llm = time.perf_counter()
            ai_msg = llm_with_tools.invoke(messages)
            latency = time.perf_counter() - t_llm

            # Build step data — convert langchain messages to OpenAI format
            openai_msgs = []
            type_to_role = {"human": "user", "ai": "assistant",
                            "system": "system", "tool": "tool"}
            for m in messages:
                role = type_to_role.get(
                    type(m).__name__.replace("Message", "").lower(), "user")
                openai_msgs.append({"role": role, "content": m.content or ""})

            step_data = {
                "type": "llm",
                "turn": 0,
                "step": iteration,
                "latency_s": latency,
                "content": ai_msg.content or "",
                "has_tool_calls": bool(ai_msg.tool_calls),
                "messages": copy.deepcopy(openai_msgs),
            }

            # Collect oracle entries
            if collect_oracle:
                oracle_entries = read_oracle_log(oracle_pos)
                if oracle_entries:
                    step_data["spec_decode"] = {
                        "oracle_vanilla_entries": oracle_entries,
                    }

            all_steps.append(step_data)
            messages.append(ai_msg)

            # Store turn messages for prompt reconstruction
            turns_with_messages.append({
                "messages": _serialize_messages(messages),
                "tool_calls": ai_msg.tool_calls or [],
                "latency": latency,
                "response": ai_msg.content or "",
            })

            # Termination logic:
            #  * tool_styles with a `submit` tool (full / sweagent): no
            #    tool_calls is treated as a parser misconfiguration → nudge
            #    the model. Explicit `submit` call breaks the loop.
            #  * mini-swe-agent style: no `submit` tool exists, and the
            #    system prompt says "produce a final assistant message with
            #    no further tool calls" to submit. Empty tool_calls is the
            #    intended termination signal — break.
            has_submit_tool = "submit" in tool_map
            if not ai_msg.tool_calls:
                if not has_submit_tool:
                    break  # mini-swe-agent style: empty tool_calls = submit
                messages.append(HumanMessage(
                    content=(
                        "Your previous response did not call any tool. You "
                        "must respond with a tool call. Use the `submit` "
                        "tool when the task is complete; otherwise call one "
                        "of the available tools to continue."
                    )
                ))
                continue
            if has_submit_tool and any(
                    tc["name"] == "submit" for tc in ai_msg.tool_calls):
                break

            # Limit to single tool call (Llama 3.1 compat). mini-swe-agent's
            # official config enables parallel_tool_calls — preserve that
            # by skipping the truncation for that style.
            if tool_style != "minisweagent" and len(ai_msg.tool_calls) > 1:
                ai_msg = AIMessage(
                    content=ai_msg.content,
                    tool_calls=[ai_msg.tool_calls[0]],
                )
                messages[-1] = ai_msg

            # Execute tools. mini-swe-agent's submission protocol asks the
            # model to run ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT &&
            # cat patch.txt`` as the final command. We honour that
            # sentinel: append the tool result as usual, then break out of
            # the loop without scheduling another LLM turn.
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

                if (tool_style == "minisweagent"
                        and tool_name == "bash"
                        and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                        in str(tc["args"].get("command", ""))):
                    submit_signal = True

            if submit_signal:
                break

    except Exception as e:
        return {
            "instance_id": instance_id,
            "category": request.get("category", ""),
            "error": str(e),
            "turns": turns_with_messages,
            "agent_metrics": {"steps": all_steps},
        }

    total_latency = time.perf_counter() - t_start

    # Extract final output
    final_output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_output = msg.content
            break

    return {
        "instance_id": instance_id,
        "category": request.get("category", ""),
        "num_turns": len(turns_with_messages),
        "total_latency": total_latency,
        "output": final_output,
        "turns": turns_with_messages,
        "agent_metrics": {
            "steps": all_steps,
            "total_steps": len(all_steps),
        },
    }


def replay_request(
    llm,
    request: dict,
    round1_question: dict,
) -> dict:
    """Replay Round 1 trajectory to collect MTP drafts.

    Reconstructs the message history from Round 1 steps and sends
    each step to the MTP server. Only oracle entries are collected.
    """
    instance_id = request.get("instance_id", "")
    round1_steps = round1_question.get("agent_metrics", {}).get("steps", [])
    round1_turns = round1_question.get("turns", [])

    # Reconstruct messages from Round 1's serialized turns
    # Each turn has "messages" which is the full message list at that point
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=request["turns"][0]),
    ]

    all_steps = []

    for i, r1_step in enumerate(round1_steps):
        step_data = {
            "type": "llm",
            "turn": 0,
            "step": r1_step.get("step", i),
        }

        # For steps after the first, append Round 1's AI response + tool results
        if i > 0 and i - 1 < len(round1_turns):
            prev_turn = round1_turns[i - 1]
            prev_response = prev_turn.get("response", "")
            prev_tool_calls = prev_turn.get("tool_calls", [])

            ai_msg = AIMessage(
                content=prev_response,
                tool_calls=prev_tool_calls,
            )
            messages.append(ai_msg)

            # Re-execute tools to get the same results for message history
            # (tool results aren't serialized in turns, so we use content from steps)
            prev_step = round1_steps[i - 1]
            if prev_step.get("has_tool_calls") and prev_tool_calls:
                for tc in prev_tool_calls:
                    # Use a placeholder result — the oracle replay forces tokens anyway
                    messages.append(
                        ToolMessage(content="[replayed]", tool_call_id=tc.get("id", ""))
                    )

        oracle_pos = get_oracle_log_position()

        t_start = time.perf_counter()
        try:
            ai_msg = llm.invoke(messages)
        except Exception as e:
            step_data["error"] = str(e)
            all_steps.append(step_data)
            continue

        latency = time.perf_counter() - t_start
        step_data["latency_s"] = latency
        step_data["content"] = ai_msg.content or ""

        oracle_entries = read_oracle_log(oracle_pos)
        if oracle_entries:
            step_data["spec_decode"] = {
                "oracle_vanilla_entries": oracle_entries,
            }

        all_steps.append(step_data)

    return {
        "instance_id": instance_id,
        "category": request.get("category", ""),
        "agent_metrics": {
            "steps": all_steps,
            "total_steps": len(all_steps),
            "mode": "replay",
        },
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def load_swebench_dataset(path: str, num_requests: int | None = None) -> list[dict]:
    """Load SWE-Bench dataset (JSONL with instance_id, repo, base_commit, turns)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if num_requests is not None:
        records = records[:num_requests]
    return records


def run_benchmark(
    url: str,
    model: str,
    input_file: str,
    output_file: str,
    repos_dir: str,
    num_requests: int | None = None,
    max_iterations: int = 15,
    temperature: float = 0.0,
    tool_style: str = "full",
    num_workers: int = 1,
    replay: str | None = None,
    resume: bool = False,
) -> None:
    """Run SWE-Bench benchmark with oracle collection."""
    collect_oracle = is_oracle_enabled()

    if collect_oracle:
        print("Oracle collection enabled (SGLANG_ORACLE_VANILLA=1)")
    else:
        print("Oracle collection disabled (set SGLANG_ORACLE_VANILLA=1 to enable)")

    # Load full dataset; --num-requests applies AFTER resume filter.
    dataset = load_swebench_dataset(input_file, None)

    # Load Round 1 results for replay mode
    round1_by_id = {}
    if replay:
        with open(replay) as f:
            round1_data = json.load(f)
        for q in round1_data["questions"]:
            iid = q.get("instance_id", "")
            round1_by_id[iid] = q
        print(f"REPLAY mode: following trajectory from {replay}")
        print(f"  Round 1 questions: {len(round1_by_id)}")

    # Resume: skip instances already in checkpoint partial.
    from simulation.pipeline.save_results import (
        load_checkpoint, append_to_checkpoint, save_agent_results,
        checkpoint_path,
    )
    cp = load_checkpoint(output_file) if resume else None
    done = set()
    questions: list = []
    total_oracle = 0
    total_steps = 0
    if cp:
        questions = list(cp.get("questions", []))
        for q in questions:
            done.add(str(q.get("instance_id", "")))
            for s in q.get("agent_metrics", {}).get("steps", []):
                total_steps += 1
                total_oracle += len(
                    s.get("spec_decode", {}).get("oracle_vanilla_entries", []))
        print(f"RESUME: {len(done)} instances already done; skipping them")

    pending = [r for r in dataset
               if str(r.get("instance_id", "")) not in done]
    if num_requests is not None:
        pending = pending[:num_requests]

    mode_str = "replay" if replay else "normal"
    print(f"Running {len(pending)}/{len(dataset)} SWE-Bench instances "
          f"({mode_str}) against {url}")

    llm = ChatOpenAI(
        base_url=url,
        model=model,
        api_key="dummy",
        temperature=temperature,
        max_tokens=32768,
        timeout=14400.0,  # 4h — oracle mode (3 tok/s) needs long ceiling
    )

    # Collect base_commits for cleanup (only for what we'll touch this run)
    base_commits = {
        r["instance_id"]: r["base_commit"]
        for r in pending
        if r.get("instance_id") and r.get("base_commit")
    }

    # Initialize repos (skip in replay mode — no tool execution)
    if not replay and pending:
        print("Initializing repositories...")
        _cleanup_repos(repos_dir, base_commits)

    if replay:
        def _process(request):
            iid = request.get("instance_id", "")
            r1_q = round1_by_id.get(iid)
            if not r1_q:
                return None
            return replay_request(llm, request, r1_q)
    else:
        def _process(request):
            return process_request(
                llm, request, repos_dir, max_iterations,
                collect_oracle, tool_style)

    if num_workers <= 1:
        results_iter = (_process(r) for r in pending)
    else:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=num_workers)
        results_iter = executor.map(_process, pending)

    def _meta():
        return {
            "model": model,
            "url": url,
            "benchmark": "swebench",
            "num_requests": len(questions),
            "total_oracle_entries": total_oracle,
            "total_steps": total_steps,
            "max_iterations": max_iterations,
            "oracle_enabled": collect_oracle,
        }

    for result in tqdm(results_iter, total=len(pending), desc="SWE-Bench"):
        if result is None:
            continue
        questions.append(result)
        for s in result.get("agent_metrics", {}).get("steps", []):
            total_steps += 1
            total_oracle += len(
                s.get("spec_decode", {}).get("oracle_vanilla_entries", []))
        append_to_checkpoint(output_file, result, _meta())

    # Cleanup repos after this batch
    if not replay and pending:
        print("Cleaning up repositories...")
        _cleanup_repos(repos_dir, base_commits)

    output = {"metadata": _meta(), "questions": questions}
    save_agent_results(output, output_file)
    try:
        checkpoint_path(output_file).unlink()
    except FileNotFoundError:
        pass

    print(f"\nResults saved to {output_file}")
    print(f"  Instances: {len(questions)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Oracle entries: {total_oracle}")

    # Extract patches
    print("Extracting patches...")
    patches = {}
    for q in questions:
        iid = q.get("instance_id", "")
        if not iid:
            continue
        repo_path = Path(repos_dir) / iid
        if not repo_path.exists():
            continue
        try:
            diff = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=repo_path, capture_output=True, text=True, timeout=10,
            )
            if diff.stdout.strip():
                patches[iid] = diff.stdout
        except Exception:
            pass

    if patches:
        patches_path = Path(output_file).parent / "patches.json"
        with open(patches_path, "w") as f:
            json.dump(patches, f, indent=2)
        print(f"  Saved {len(patches)} patches to {patches_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SWE-Bench LangChain agent for SGLang oracle collection")
    parser.add_argument("--url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--input-file", required=True,
                        help="SWE-Bench dataset JSONL")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--repos-dir", required=True,
                        help="Directory with pre-cloned repos")
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tool-style", default="full",
                        choices=["full", "sweagent", "minisweagent"],
                        help="Tool style: full (6 tools) or sweagent (3 tools)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Concurrent requests (caution: SWE-bench repos may conflict)")
    parser.add_argument("--replay", default=None,
                        help="Path to Round 1 agent_results.json for replay mode")
    parser.add_argument("--resume", action="store_true",
                        help="Skip instances already saved in <output>.partial")
    args = parser.parse_args()

    run_benchmark(
        url=args.url,
        model=args.model,
        input_file=args.input_file,
        output_file=args.output_file,
        repos_dir=args.repos_dir,
        num_requests=args.num_requests,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        tool_style=args.tool_style,
        num_workers=args.num_workers,
        replay=args.replay,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
