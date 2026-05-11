"""Extract per-request trajectory tokens from agent_results_eagle3.json
for SGLANG_ORACLE_REPLAY mode.

For each oracle_vanilla_entry, the verified token at that step is
``entry["tokens"][0][0]`` (the per-step base verified token committed
by the target). Replay mode forces accept_length=0 so each replay step
emits exactly one token from the trajectory.

For trajectory faithfulness, however, we also include the draft-accepted
tokens (``entry["tokens"][0][1:]``) — these are the tokens that the
ORIGINAL eagle3 capture committed at that step. The replay loader
walks one token per replay step in order.

Output schema (one JSON file per workload):
    {req_id: [committed_token_ids_in_order], ...}

Streamed via ijson (mandatory for swebench's 25GB capture).

Usage:
    python3 -m simulation.scripts.extract_replay_trajectory \\
        --src-pattern 'simulation/results/qwen3_14b/{wl}_steps8_topk16_capture/agent_results_eagle3.json' \\
        --out-pattern 'simulation/results/qwen3_14b/replay_trajectories/{wl}.json' \\
        --workloads specbench,bfcl_v4,swebench_verified
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import ijson


def _walk_step_entries(step_obj: dict):
    """Yield (req_id, [tokens]) per oracle_vanilla_entry in this step.
    Filters interleaved entries (concurrent batches) by majority rid.
    """
    entries = (step_obj.get("spec_decode") or {}).get(
        "oracle_vanilla_entries", []) or []
    if not entries:
        return
    rids = [e.get("req_id", "") for e in entries]
    if len(set(rids)) > 1:
        from collections import Counter
        primary = Counter(rids).most_common(1)[0][0]
        entries = [e for e in entries if e.get("req_id") == primary]
    for e in entries:
        rid = e.get("req_id", "")
        toks = e.get("tokens")
        if not toks or not toks[0]:
            continue
        yield rid, list(toks[0])


def extract_one(src_path: Path) -> dict[str, list[int]]:
    """Stream src_path, return {req_id: [tokens]} accumulated per call."""
    traj: dict[str, list[int]] = defaultdict(list)
    n_questions = 0
    n_entries = 0
    t0 = time.time()
    with open(src_path, "rb") as f:
        for q in ijson.items(f, "questions.item"):
            n_questions += 1
            # BFCL / SWE-bench format
            agent_metrics = q.get("agent_metrics") or {}
            steps = agent_metrics.get("steps") or []
            if steps:
                for s in steps:
                    for rid, toks in _walk_step_entries(s):
                        traj[rid].extend(int(t) for t in toks)
                        n_entries += 1
                continue
            # SpecBench / online format
            turns = q.get("turns") or []
            for turn in turns:
                if isinstance(turn, dict):
                    for rid, toks in _walk_step_entries(turn):
                        traj[rid].extend(int(t) for t in toks)
                        n_entries += 1
            if n_questions % 50 == 0:
                print(f"  ... {n_questions} questions, {len(traj)} rids, "
                      f"{n_entries} entries", file=sys.stderr)
    print(f"  done: {n_questions} questions, {len(traj)} rids, "
          f"{n_entries} entries in {time.time() - t0:.1f}s",
          file=sys.stderr)
    # Drop empty trajectories (defensive)
    return {rid: toks for rid, toks in traj.items() if toks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-pattern", required=True,
                    help="e.g. 'simulation/results/qwen3_14b/{wl}_steps8_topk16_capture/agent_results_eagle3.json'")
    ap.add_argument("--out-pattern", required=True,
                    help="e.g. 'simulation/results/qwen3_14b/replay_trajectories/{wl}.json'")
    ap.add_argument("--workloads", required=True,
                    help="comma-separated workload names")
    args = ap.parse_args()

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    for wl in workloads:
        src = Path(args.src_pattern.format(wl=wl))
        out = Path(args.out_pattern.format(wl=wl))
        if not src.exists():
            print(f"SKIP {wl}: src not found ({src})", file=sys.stderr)
            continue
        print(f"[{wl}] extracting from {src}", file=sys.stderr)
        traj = extract_one(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(traj, f)
        size_mb = out.stat().st_size / (1024 * 1024)
        total_tokens = sum(len(v) for v in traj.values())
        print(f"[{wl}] wrote {out} — {len(traj)} rids, "
              f"{total_tokens:,} tokens, {size_mb:.1f} MB",
              file=sys.stderr)


if __name__ == "__main__":
    main()
