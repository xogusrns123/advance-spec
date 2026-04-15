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
    python3 -m hybrid_spec_decoding.analysis.swebench_agent \
        --url http://localhost:30000/v1 \
        --model Qwen/Qwen3-8B \
        --input-file data/swebench/dataset.jsonl \
        --output-file results/qwen3_8b/swebench/agent_results_eagle3.json \
        --repos-dir data/swebench/repos \
        --max-iterations 15
"""

from __future__ import annotations

import argparse
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

from ..sglang_integration.oracle_patch import (
    clear_oracle_log,
    read_oracle_log,
    is_oracle_enabled,
)
from .tools.swebench import create_swebench_tools, create_sweagent_tools


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
    else:
        tools = create_swebench_tools(workdir, repo=repo)
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    # Initialize messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=request["turns"][0]),
    ]

    t_start = time.perf_counter()
    all_steps = []
    turns_with_messages = []

    try:
        for iteration in range(max_iterations):
            # Clear oracle log before LLM call
            if collect_oracle:
                clear_oracle_log()

            # LLM call
            t_llm = time.perf_counter()
            ai_msg = llm_with_tools.invoke(messages)
            latency = time.perf_counter() - t_llm

            # Build step data
            step_data = {
                "type": "llm",
                "turn": 0,
                "step": iteration,
                "latency_s": latency,
                "content": ai_msg.content or "",
                "has_tool_calls": bool(ai_msg.tool_calls),
            }

            # Collect oracle entries
            if collect_oracle:
                oracle_entries = read_oracle_log()
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

            # Check for completion
            if not ai_msg.tool_calls:
                break
            if any(tc["name"] == "submit" for tc in ai_msg.tool_calls):
                break

            # Limit to single tool call (Llama 3.1 compat)
            if len(ai_msg.tool_calls) > 1:
                ai_msg = AIMessage(
                    content=ai_msg.content,
                    tool_calls=[ai_msg.tool_calls[0]],
                )
                messages[-1] = ai_msg

            # Execute tools
            for tc in ai_msg.tool_calls:
                tool_name = tc["name"]
                t_tool = time.perf_counter()
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
) -> None:
    """Run SWE-Bench benchmark with oracle collection."""
    collect_oracle = is_oracle_enabled()

    if collect_oracle:
        print("Oracle collection enabled (SGLANG_ORACLE_VANILLA=1)")
    else:
        print("Oracle collection disabled (set SGLANG_ORACLE_VANILLA=1 to enable)")

    dataset = load_swebench_dataset(input_file, num_requests)
    print(f"Running {len(dataset)} SWE-Bench instances against {url}")

    llm = ChatOpenAI(
        base_url=url,
        model=model,
        api_key="dummy",
        temperature=temperature,
        max_tokens=4096,
    )

    # Collect base_commits for cleanup
    base_commits = {
        r["instance_id"]: r["base_commit"]
        for r in dataset
        if r.get("instance_id") and r.get("base_commit")
    }

    # Initialize repos
    print("Initializing repositories...")
    _cleanup_repos(repos_dir, base_commits)

    questions = []
    total_oracle = 0
    total_steps = 0

    for request in tqdm(dataset, desc="SWE-Bench"):
        result = process_request(
            llm, request, repos_dir, max_iterations,
            collect_oracle, tool_style,
        )
        questions.append(result)
        for s in result.get("agent_metrics", {}).get("steps", []):
            total_steps += 1
            total_oracle += len(
                s.get("spec_decode", {}).get("oracle_vanilla_entries", []))

    # Cleanup repos after all instances
    print("Cleaning up repositories...")
    _cleanup_repos(repos_dir, base_commits)

    # Save results
    output = {
        "metadata": {
            "model": model,
            "url": url,
            "benchmark": "swebench",
            "num_requests": len(questions),
            "total_oracle_entries": total_oracle,
            "total_steps": total_steps,
            "max_iterations": max_iterations,
            "oracle_enabled": collect_oracle,
        },
        "questions": questions,
    }

    from .save_results import save_agent_results
    save_agent_results(output, output_file)

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
        patches_path = out_path.parent / "patches.json"
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
                        choices=["full", "sweagent"],
                        help="Tool style: full (6 tools) or sweagent (3 tools)")
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
    )


if __name__ == "__main__":
    main()
