#!/usr/bin/env python3
"""Quick health check over chain-hybrid agent result files: tool calling
success, response patterns, error indicators. Usage:
    python3 check_agent_health.py <agent_results.json> [label] [max_iter]
"""
import json
import re
import sys
from collections import Counter


def analyze(path: str, label: str, max_iter: int) -> None:
    d = json.load(open(path))
    qs = d.get("questions", [])
    print(f"\n===== {label}: {len(qs)} tasks =====")
    agg_tools = Counter()
    issues = []
    for q in qs:
        m = q.get("agent_metrics", {})
        steps = m.get("steps", [])
        llm = [s for s in steps if s.get("type") == "llm"]
        tools = [s for s in steps if s.get("type") != "llm"]
        tool_names = Counter()
        tool_errors = 0
        for s in tools:
            name = s.get("name") or s.get("tool") or s.get("type")
            tool_names[name] += 1
            agg_tools[name] += 1
            res = str(s.get("result") or s.get("content") or "")
            if re.search(r"error|exception|traceback|failed", res[:500], re.I):
                tool_errors += 1
        empty_llm = sum(1 for s in llm if not (s.get("content") or "").strip())
        hit_max = len(llm) >= max_iter
        extra = {k: m[k] for k in m if k != "steps"}
        bid = str(q.get("bfcl_id"))
        print(f"{bid:24s} llm={len(llm):2d} tool={len(tools):2d} "
              f"tool_err={tool_errors} empty_llm={empty_llm} "
              f"hit_max_iter={hit_max} extra={extra if extra else ''}")
        if tool_names:
            print(f"{'':24s}   tools: {dict(tool_names)}")
        if tool_errors or empty_llm or hit_max:
            issues.append(bid)
    print(f"  ALL TOOLS USED: {dict(agg_tools)}")
    print(f"  tasks with potential issues: {issues if issues else 'none'}")


if __name__ == "__main__":
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    analyze(path, label, max_iter)
