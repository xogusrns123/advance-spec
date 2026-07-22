"""O4 — per-depth selection oracle, measured by REAL SERVING (not simulation).

O4 is the ceiling of the chain-hybrid per-depth select-1 policy: at every draft
depth the eagle/MTP top-1 and the suffix top-1 compete and the GROUND-TRUTH
matching token is committed; the chain then RE-SPECULATES from there and selects
again, per depth. Crucially this allows eagle/MTP to resume *after* a suffix
token was committed — which requires the target model's hidden state at that
suffix position. A static offline capture does not have that hidden feature
(it only holds eagle's own-chain features), so O4 is NOT offline-simulable and
must be measured online. The chain-hybrid patch's `oracle` mode does exactly
this during live serving (the real model supplies every hidden state), so O4 is
the `select1_oracle` arm's mean-accepted-tokens (MAT).

This script is intentionally SEPARATE from the offline O0-O3 hand-off oracle
(simulation/scripts/{experiments/run_chain_handoff_oracle,analyze_chain_handoff}.py),
which stays a pure offline analysis. It reuses the proven serving building
blocks from measure_chain_hybrid.py (server boot / env / agent / MAT summary).

Flow (per the chain-hybrid GT plumbing):
  1. `baseline`  — eagle/MTP-only chain (reference MAT; = serving O0_eagle).
  2. `record`    — select-1 in record mode: dumps gt_tokens.jsonl (the realized
                   greedy GT trajectory) + agent_results_record.json.
  3. `select1_oracle` — select-1 in oracle mode: consumes the GT and REPLAYS the
                   record conversation (byte-identical), doing per-depth oracle
                   selection. Its MAT == O4.

Usage (inside the sglang-bench container, cwd = repo root; GPU0/Blackwell needs
the same triton/sampling backend env as the captures — pass via --extra-args or
rely on the launcher env):
    python simulation/scripts/measure_o4_perdepth_oracle.py \
        --preset qwen35_27b_mtp --workload bfcl_v4 --steps 16 --n-tasks 10 \
        --include-category web_search --tp-size 1 --port 30021 \
        --output simulation/results/o4_perdepth/qwen35_27b_mtp/o4.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_eagle3_cost import (  # noqa: E402
    kill_server,
    read_timing_window,
    summarize_entries,
    wait_for_server,
)
from measure_chain_hybrid import (  # noqa: E402
    MODEL_PRESETS,
    WORKLOAD_REGISTRY,
    build_env,
    build_server_cmd,
    is_chain_hybrid_arm,
    run_agent,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Minimal arm set for O4. baseline = eagle/MTP-only reference; record produces
# the GT + replay conversation; select1_oracle IS O4. (select1, the DEPLOYABLE
# non-oracle per-depth policy, can be added via --arms for a deployable-vs-O4
# gap.) Order matters: record must precede select1_oracle.
DEFAULT_ARMS = "baseline,record,select1_oracle"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="qwen35_27b_mtp",
                    choices=list(MODEL_PRESETS))
    ap.add_argument("--workload", default="bfcl_v4", choices=list(WORKLOAD_REGISTRY))
    ap.add_argument("--steps", type=int, default=16,
                    help="chain length S (topk=1, num_draft_tokens=S+1)")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N workload tasks (carve a slice)")
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--include-category", default=None)
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help="comma list; must include record before select1_oracle")
    ap.add_argument("--suffix-num-draft-tokens", type=int, default=64)
    ap.add_argument("--tail-max-tokens", type=int, default=0,
                    help="suffix tail-append cap (0 = off; keep off so O4 is the "
                         "pure per-depth select-1 oracle)")
    ap.add_argument("--tail-factor", type=float, default=4.0)
    ap.add_argument("--tail-min-prob", type=float, default=0.1)
    ap.add_argument("--tail-check", action="store_true")
    # hybrid_e3 knobs are unused here but build_env reads them defensively.
    ap.add_argument("--hybrid-score-threshold", type=float, default=5.0)
    ap.add_argument("--hybrid-fb-factor", type=float, default=1.0)
    ap.add_argument("--hybrid-fb-min-prob", type=float, default=0.1)
    ap.add_argument("--port", type=int, default=30021)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--kv-cache-dtype", default="fp8_e5m2")
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--output", required=True,
                    help="summary JSON path (per-arm MAT + headline O4)")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                    help="extra sglang.launch_server flags (e.g. --attention-backend triton)")
    args = ap.parse_args()

    preset = MODEL_PRESETS[args.preset]
    workload = WORKLOAD_REGISTRY[args.workload]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if "select1_oracle" in arms and "record" not in arms[:arms.index("select1_oracle")]:
        sys.exit("--arms must list 'record' before 'select1_oracle' (oracle "
                 "needs the record arm's GT + replay conversation)")

    url = f"http://localhost:{args.port}"
    output_path = Path(args.output)
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "preset": args.preset, "model": preset["model"],
        "draft_model": preset["draft_model"], "workload": args.workload,
        "steps": args.steps, "n_tasks": args.n_tasks, "offset": args.offset,
        "headline": "O4 = select1_oracle accept_length_mean (per-depth selection,"
                    " real serving)",
        "arms": {},
    }

    def _save():
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
    _save()

    gt_path = out_dir / "gt_tokens.jsonl"
    for arm in arms:
        print("=" * 72, file=sys.stderr)
        print(f"ARM {arm}: {args.workload} x {args.n_tasks} (offset {args.offset}),"
              f" S={args.steps} topk=1", file=sys.stderr)
        print("=" * 72, file=sys.stderr)

        timing_log = Path(f"/tmp/o4_timing_{arm}_p{args.port}.jsonl")
        decision_log = Path(f"/tmp/o4_decisions_{arm}_p{args.port}.jsonl")
        for p in (timing_log, decision_log):
            p.unlink(missing_ok=True)

        # GT plumbing (mirrors measure_chain_hybrid): record dumps GT; oracle
        # consumes it AND replays the record conversation byte-for-byte.
        gt_out = gt_file = replay_file = None
        if arm == "record":
            gt_path.unlink(missing_ok=True)
            gt_out = gt_path
        if arm == "select1_oracle":
            gt_file = gt_path
            replay_file = out_dir / "agent_results_record.json"
            missing = [str(p) for p in (gt_file, replay_file) if not p.exists()]
            if missing:
                summary["arms"][arm] = {"error": f"oracle inputs missing: {missing}"}
                _save()
                print(f"ERROR: {arm}: missing {missing}", file=sys.stderr)
                continue

        env = build_env(args, arm, timing_log, decision_log, None,
                        gt_out=gt_out, gt_file=gt_file)
        cmd = build_server_cmd(args, arm, preset)
        server_log = out_dir / f"server_{arm}.log"
        agent_out = out_dir / f"agent_trajectory_{arm}.json"
        log_fh = open(server_log, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh,
                                cwd=str(REPO_ROOT))
        arm_row: dict = {"server_log": str(server_log)}
        try:
            if not wait_for_server(url):
                kill_server(proc)
                arm_row["error"] = "server_failed"
                summary["arms"][arm] = arm_row
                _save()
                print(f"ERROR: {arm} server failed. See {server_log}", file=sys.stderr)
                continue
            rc, wall = run_agent(args, workload, agent_out, env,
                                 n_tasks=args.n_tasks, offset=args.offset,
                                 replay=replay_file)
            arm_row.update({"agent_rc": rc, "wall_time_s": round(wall, 1),
                            "agent_output": str(agent_out)})
            if timing_log.exists():
                arm_row.update(summarize_entries(read_timing_window(timing_log, 0)))
                kept = out_dir / f"timing_{arm}.jsonl"
                shutil.copyfile(timing_log, kept)
                arm_row["timing_log"] = str(kept)
            if is_chain_hybrid_arm(arm) and decision_log.exists():
                kept = out_dir / f"decisions_{arm}.jsonl"
                shutil.copyfile(decision_log, kept)
                arm_row["decision_log"] = str(kept)
            summary["arms"][arm] = arm_row
            _save()
            print(f"  {arm}: MAT(accept_len_mean)="
                  f"{arm_row.get('accept_length_mean', float('nan')):.3f} "
                  f"n={arm_row.get('n_samples', 0)} wall={arm_row.get('wall_time_s')}s",
                  file=sys.stderr)
        finally:
            kill_server(proc)
            try:
                log_fh.close()
            except Exception:
                pass
            time.sleep(3)

    o4 = summary["arms"].get("select1_oracle", {}).get("accept_length_mean")
    base = summary["arms"].get("baseline", {}).get("accept_length_mean")
    summary["O4_mat"] = o4
    summary["baseline_mat"] = base
    _save()
    print("\n" + "=" * 72, file=sys.stderr)
    print(f"O4 (per-depth selection oracle, serving MAT) = "
          f"{o4 if o4 is None else round(o4, 3)}"
          + (f"   |   baseline eagle/MTP-only = {round(base, 3)}"
             if base is not None else ""), file=sys.stderr)
    print(f"Summary: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
