#!/usr/bin/env python3
"""Scan all agent_trajectory *_response.json under simulation/results for the
bfcl_v4 runaway-repetition contamination (model loops "[call]</think>..." to
max_tokens). Reports, per file: #tasks, #tasks with a runaway step (>20K chars),
#tasks that executed 0 tools, and the % of generated text in runaway steps.

Usage (inside sglang-bench):  python3 scan_runaway_contamination.py
"""
import glob
import json
import os
import sys

ROOT = "/workspace/simulation/results"
RUNAWAY_CHARS = 20000


def main() -> None:
    files = sorted(glob.glob(ROOT + "/**/*_response.json", recursive=True))
    rows = []
    for f in files:
        try:
            if os.path.getsize(f) > 120_000_000:
                continue  # skip giant per-token capture; the _response sibling has the text
            d = json.load(open(f))
        except Exception:
            continue
        qs = d.get("questions") if isinstance(d, dict) else None
        if not qs:
            continue
        nq = len(qs)
        runaway_tasks = noexec = run_chars = tot_chars = 0
        is_ws = False
        for q in qs:
            if "web_search" in str(q.get("bfcl_id", "")) + str(q.get("category", "")):
                is_ws = True
            anyexec = hasrun = False
            for s in q.get("agent_metrics", {}).get("steps", []):
                c = len(s.get("content") or "")
                tot_chars += c
                if s.get("exec_results") or s.get("has_tool_calls"):
                    anyexec = True
                if c > RUNAWAY_CHARS:
                    hasrun = True
                    run_chars += c
            runaway_tasks += hasrun
            noexec += not anyexec
        frac = round(100 * run_chars / tot_chars, 1) if tot_chars else 0.0
        rows.append((f.replace(ROOT + "/", ""), "WS" if is_ws else "other",
                     nq, runaway_tasks, noexec, frac))

    rows = [r for r in rows if r[1] == "WS" or r[3] > 0]
    rows.sort(key=lambda r: -r[5])
    print(f'{"file":72s} {"kind":5s} {"nq":>3} {"runaway":>7} {"noexec":>6} {"run%":>6}')
    for r in rows:
        print(f"{r[0]:72s} {r[1]:5s} {r[2]:>3} {r[3]:>7} {r[4]:>6} {r[5]:>6}")
    print(f"\nfiles shown (web_search or contaminated): {len(rows)}")
    print(f"contaminated (>=1 runaway task): {sum(1 for r in rows if r[3] > 0)}")


if __name__ == "__main__":
    sys.exit(main())
