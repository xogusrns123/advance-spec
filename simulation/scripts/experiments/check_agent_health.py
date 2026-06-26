#!/usr/bin/env python3
"""Quick health check over chain-hybrid agent result files: tool calling
success, response patterns, error indicators. Usage:
    python3 check_agent_health.py <agent_results.json> [label] [max_iter]
"""
import json
import re
import sys
from collections import Counter


# A single step whose generated text exceeds this is a degenerate repetition
# runaway (the model loops "[call]</think>[call]</think>..." to max_tokens).
RUNAWAY_CHARS = 20000


def _tool_name(call: str) -> str:
    """'search_engine_query(keywords=...)' -> 'search_engine_query'."""
    s = str(call).strip().lstrip("[")
    return s.split("(", 1)[0].strip() or "?"


def analyze(path: str, label: str, max_iter: int) -> None:
    d = json.load(open(path))
    qs = d.get("questions", [])
    print(f"\n===== {label}: {len(qs)} tasks =====")
    agg_tools = Counter()
    issues = []
    n_no_exec = n_runaway = 0
    for q in qs:
        m = q.get("agent_metrics", {})
        steps = m.get("steps", [])
        # In this harness every step is type=="llm"; a tool call is recorded
        # INLINE via has_tool_calls / decoded_calls / exec_results, not as a
        # separate step. (The old type!="llm" count always read 0 here.)
        llm = [s for s in steps if s.get("type", "llm") == "llm"]
        tool_names = Counter()
        tool_errors = 0
        exec_steps = 0
        runaway = 0
        for s in steps:
            executed = bool(s.get("exec_results")) or bool(s.get("has_tool_calls"))
            if executed:
                exec_steps += 1
                for call in (s.get("decoded_calls") or []):
                    name = _tool_name(call)
                    tool_names[name] += 1
                    agg_tools[name] += 1
            for res in (s.get("exec_results") or []):
                if re.search(r"error|exception|traceback|failed", str(res)[:500], re.I):
                    tool_errors += 1
            if s.get("exec_error"):
                tool_errors += 1
            if len(s.get("content") or "") > RUNAWAY_CHARS:
                runaway += 1
        empty_llm = sum(1 for s in llm if not (s.get("content") or "").strip())
        hit_max = len(llm) >= max_iter
        extra = {k: m[k] for k in m if k != "steps"}
        bid = str(q.get("bfcl_id"))
        if exec_steps == 0:
            n_no_exec += 1
        if runaway:
            n_runaway += 1
        print(f"{bid:24s} llm={len(llm):2d} exec={exec_steps:2d} "
              f"tool_err={tool_errors} empty_llm={empty_llm} runaway={runaway} "
              f"hit_max_iter={hit_max} extra={extra if extra else ''}")
        if tool_names:
            print(f"{'':24s}   tools: {dict(tool_names)}")
        if tool_errors or empty_llm or hit_max or runaway or exec_steps == 0:
            issues.append(bid)
    print(f"  ALL TOOLS USED: {dict(agg_tools)}")
    print(f"  tasks that executed 0 tools: {n_no_exec}/{len(qs)} | "
          f"tasks with a runaway step (>{RUNAWAY_CHARS} chars): {n_runaway}/{len(qs)}")
    print(f"  tasks with potential issues: {issues if issues else 'none'}")


if __name__ == "__main__":
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    analyze(path, label, max_iter)
