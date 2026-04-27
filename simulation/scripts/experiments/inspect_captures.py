#!/usr/bin/env python3
"""Inspect rr captures — sanity check on agent execution.

Reads the FULL agent_results_eagle3.json (oracle entries present) and
prints per-question summary + sample text.
"""
import json
import sys
import collections
from pathlib import Path


CAPTURES = Path("/workspace/simulation/results/qwen3_14b")


def safe_str(x) -> str:
    return x if isinstance(x, str) else str(x)


def truncate(s: str, n: int = 200) -> str:
    s = safe_str(s)
    return s if len(s) <= n else s[:n] + "...(+%d)" % (len(s) - n)


def get_q_id(q: dict) -> str:
    qid = (q.get("bfcl_id") or q.get("instance_id")
           or q.get("question_id") or q.get("id") or "?")
    return safe_str(qid)


def summarize_question(q: dict) -> dict:
    cat = q.get("category", "?")
    output = q.get("output", "")
    am = q.get("agent_metrics") or {}
    steps = am.get("steps") or []

    # Two structures: agent (bfcl/swebench) vs single-turn (specbench/longbench)
    # For single-turn workloads, oracle entries are inside `turns[].spec_decode`.
    turns_in_q = q.get("turns") or []
    n_steps = len(steps)
    total_tokens = sum(s.get("completion_tokens") or 0 for s in steps)
    oracle_entries = 0
    for s in steps:
        sd = s.get("spec_decode") or {}
        oracle_entries += len(sd.get("oracle_vanilla_entries") or [])
    # Also check single-turn turn-level oracle entries
    for t in turns_in_q:
        if isinstance(t, dict):
            sd = t.get("spec_decode") or {}
            oracle_entries += len(sd.get("oracle_vanilla_entries") or [])

    # Output may be list (specbench has output per turn) or str
    out_text = ""
    out_len = 0
    if isinstance(output, list):
        out_text = "\n---\n".join(safe_str(x) for x in output)
        out_len = sum(len(safe_str(x)) for x in output)
    elif isinstance(output, str):
        out_text = output
        out_len = len(output)

    # Specbench: turns have "response" appended after run? Check turn-level.
    turn_responses_chars = 0
    if not out_len:
        for t in turns_in_q:
            if isinstance(t, dict):
                resp = t.get("response", "")
                turn_responses_chars += len(safe_str(resp))
        out_len = turn_responses_chars

    return {
        "id": get_q_id(q),
        "category": safe_str(cat),
        "n_input_turns": len(turns_in_q),
        "n_steps": n_steps,
        "total_tokens": total_tokens,
        "oracle_entries": oracle_entries,
        "total_latency": q.get("total_latency"),
        "max_iter_hit": am.get("max_iter_reached", False),
        "output_len": out_len,
        "_out_text": out_text,
    }


def inspect(cap_dir: Path) -> None:
    name = cap_dir.name
    full = cap_dir / "agent_results_eagle3.json"
    if not full.is_file():
        print(f"\n=== {name}: no agent_results_eagle3.json ===")
        return
    print(f"\n=== {name} ===")
    print(f"  loading {full} ({full.stat().st_size // 1024 // 1024} MB)...")
    try:
        d = json.load(open(full))
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return
    qs = d.get("questions", [])
    print(f"  questions: {len(qs)}")
    if not qs:
        return

    cats = collections.Counter(q.get("category", "?") for q in qs)
    for c, n in cats.most_common():
        print(f"    cat[{c}]: {n}")

    summaries = [summarize_question(q) for q in qs]
    print(f"  {'id':<48} {'cat':<28} {'turns':>5} {'steps':>5} {'toks':>6} {'oracle':>6} {'lat_s':>6} {'maxIt':>5} {'out_ch':>7}")
    for s in summaries[:6] + (summaries[-3:] if len(summaries) > 9 else []):
        latency = f"{s['total_latency']:.1f}" if isinstance(s['total_latency'], (int, float)) else "-"
        idstr = s["id"][:47]
        catstr = s["category"][:27]
        print(f"  {idstr:<48} {catstr:<28} {s['n_input_turns']:>5d} "
              f"{s['n_steps']:>5d} {s['total_tokens']:>6d} {s['oracle_entries']:>6d} "
              f"{latency:>6} {'Y' if s['max_iter_hit'] else '-':>5} {s['output_len']:>7d}")

    # Aggregate
    n_zero_oracle = sum(1 for s in summaries if s['oracle_entries'] == 0)
    n_zero_tokens = sum(1 for s in summaries if s['total_tokens'] == 0)
    n_max_iter = sum(1 for s in summaries if s['max_iter_hit'])
    n_zero_out = sum(1 for s in summaries if s['output_len'] == 0)
    print(f"  → 0 oracle: {n_zero_oracle}/{len(qs)}, "
          f"0 tokens: {n_zero_tokens}/{len(qs)}, "
          f"0 output: {n_zero_out}/{len(qs)}, "
          f"max_iter: {n_max_iter}/{len(qs)}")

    # Sample first non-empty output
    for q, s in zip(qs, summaries):
        if s["output_len"] > 0:
            print(f"  [SAMPLE q={s['id'][:50]} cat={s['category'][:30]}]")
            print(f"    output (first 300 chars):")
            print(f"      {truncate(s['_out_text'], 300)}")
            break

    # Probe top-level keys of one question for schema awareness
    if qs:
        q = qs[0]
        print(f"  top-level keys[0]: {sorted(q.keys())[:15]}")


def main():
    workloads = sys.argv[1:] if len(sys.argv) > 1 else None
    caps = sorted(CAPTURES.glob("*_steps8_topk16_capture"))
    for cap in caps:
        if workloads and not any(w in cap.name for w in workloads):
            continue
        inspect(cap)


if __name__ == "__main__":
    main()
