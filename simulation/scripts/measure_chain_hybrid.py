"""A/B driver for the chain-hybrid (EAGLE3 + suffix) serving PoC.

Per arm: boot one SGLang server (chain mode: topk=1, num_draft_tokens=S+1),
run the workload agent for N tasks against it (the server stays alive within
an arm so the global suffix trie warms across tasks), aggregate per-step
timings from the oracle timing log, and emit one JSON block per arm.

Arms:
  baseline  eagle3-only chain   (SGLANG_ORACLE_VANILLA=1 SGLANG_LATENCY_ONLY=1)
  select1   per-depth select-1  (+ SGLANG_CHAIN_HYBRID=1; at every depth
                                 eagle3 top-1 and suffix top-1 are re-drawn
                                 and compete by probability, decision JSONL)
  suffix    suffix-only CHAIN   (--speculative-algorithm SUFFIX +
                                 SGLANG_SUFFIX_CHAIN=1: top-1 path spec, no
                                 length cap beyond --suffix-num-draft-tokens;
                                 SuffixWorker has no per-step timing ->
                                 wall-clock only)

Usage (inside the sglang-bench container, cwd = repo root):
    python simulation/scripts/measure_chain_hybrid.py \\
        --preset qwen3_14b --workload bfcl_v4 --steps 8 --n-tasks 10 \\
        --arms baseline,select1,suffix \\
        --output simulation/results/chain_hybrid_poc/poc.json

Analysis: simulation/scripts/analyze_chain_hybrid.py joins the decision and
timing logs (copied next to --output) into the per-arm comparison table.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from measure_eagle3_cost import (  # noqa: E402
    kill_server,
    read_timing_window,
    summarize_entries,
    wait_for_server,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PRESETS: dict[str, dict] = {
    "qwen3_14b": {
        "model": "Qwen/Qwen3-14B",
        "draft_model": "AngelSlim/Qwen3-14B_eagle3",
        "tool_call_parser": "qwen25",
    },
    "qwen3_8b": {
        "model": "Qwen/Qwen3-8B",
        "draft_model": "AngelSlim/Qwen3-8B_eagle3",
        "tool_call_parser": "qwen25",
    },
}

WORKLOAD_REGISTRY: dict[str, dict] = {
    "bfcl_v4": {
        "agent_module": "simulation.agents.bfcl_v4_agent",
        "dataset": "data/bfcl_agent/dataset_stratified_interleaved.jsonl",
    },
    "specbench": {
        "agent_module": "simulation.agents.specbench_agent",
        "dataset": "data/specbench/dataset_interleaved.jsonl",
    },
}

ARM_NAMES = ("baseline", "select1", "suffix")


def build_server_cmd(args, arm: str, preset: dict) -> list[str]:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", preset["model"],
        "--tp-size", str(args.tp_size),
    ]
    if arm == "suffix":
        # SuffixWorker path (install_hook's on-disk SUFFIX patch). No draft
        # model. Chain mode: the draft length is uncapped up to the verify
        # tensor size (--suffix-num-draft-tokens), not tied to eagle steps.
        cmd += [
            "--speculative-algorithm", "SUFFIX",
            "--speculative-num-draft-tokens", str(args.suffix_num_draft_tokens),
        ]
    else:
        cmd += [
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", preset["draft_model"],
            "--speculative-num-steps", str(args.steps),
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", str(args.steps + 1),
        ]
    cmd += [
        "--tool-call-parser", preset["tool_call_parser"],
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--max-running-requests", "1",
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--disable-cuda-graph",
        # RTX 4090s have no P2P peer access; SGLang falls back anyway but
        # warns loudly per rank — disable explicitly.
        "--disable-custom-all-reduce",
        "--watchdog-timeout", "600",
        "--host", "0.0.0.0", "--port", str(args.port),
    ]
    if args.context_length:
        cmd += ["--context-length", str(args.context_length)]
    cmd += args.extra_args
    return cmd


def build_env(args, arm: str, timing_log: Path, decision_log: Path) -> dict:
    env = os.environ.copy()
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    # Eagle arms get the LATENCY_ONLY instrumentation via the worker-init
    # hook; the suffix arm uses SuffixWorker (not an EAGLEWorker), so the
    # oracle env vars are irrelevant there and left unset.
    if arm in ("baseline", "select1"):
        env["SGLANG_ORACLE_VANILLA"] = "1"
        env["SGLANG_LATENCY_ONLY"] = "1"
        env["SGLANG_ORACLE_TIMING_LOG"] = str(timing_log)
    if arm == "select1":
        env["SGLANG_CHAIN_HYBRID"] = "1"
        env["SGLANG_CHAIN_HYBRID_LOG"] = str(decision_log)
    if arm == "suffix":
        env["SGLANG_SUFFIX_CHAIN"] = "1"
    return env


def run_agent(args, workload: dict, out_file: Path, env: dict) -> tuple[int, float]:
    """Run the workload agent once against the live server. Returns
    (returncode, wall_time_s)."""
    preset = MODEL_PRESETS[args.preset]
    cmd = [
        sys.executable, "-m", workload["agent_module"],
        "--url", f"http://localhost:{args.port}/v1",
        "--model", preset["model"],
        "--input-file", workload["dataset"],
        "--output-file", str(out_file),
        "--num-requests", str(args.n_tasks),
        "--num-workers", "1",
    ]
    if args.workload == "bfcl_v4":
        if args.max_iterations:
            cmd += ["--max-iterations", str(args.max_iterations)]
        if args.include_category:
            cmd += ["--include-category", args.include_category]

    log_path = out_file.parent / f"{out_file.stem}_agent.log"
    t0 = time.perf_counter()
    with open(log_path, "a") as lf:
        rc = subprocess.call(cmd, env=env, stdout=lf,
                             stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    wall = time.perf_counter() - t0
    if rc != 0:
        print(f"  agent rc={rc} (see {log_path})", file=sys.stderr)
    return rc, wall


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="qwen3_14b",
                        choices=sorted(MODEL_PRESETS.keys()))
    parser.add_argument("--workload", default="bfcl_v4",
                        choices=sorted(WORKLOAD_REGISTRY.keys()))
    parser.add_argument("--steps", type=int, default=8,
                        help="Chain depth S (num_draft_tokens is forced to S+1)")
    parser.add_argument("--n-tasks", type=int, default=10,
                        help="Agent tasks per arm (--num-requests)")
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="bfcl_v4 agent --max-iterations")
    parser.add_argument("--include-category", default=None,
                        help="bfcl_v4 agent --include-category substring "
                             "filter (e.g. 'web_search')")
    parser.add_argument("--arms", default="baseline,select1,suffix",
                        help=f"Comma list from {ARM_NAMES}")
    parser.add_argument("--suffix-num-draft-tokens", type=int, default=64,
                        help="Verify tensor size for the suffix arm (the "
                             "chain draft is uncapped up to this minus 1)")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--kv-cache-dtype", default="fp8_e5m2")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--output", required=True,
                        help="Summary JSON; raw logs are copied next to it")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Passed through to sglang.launch_server")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_NAMES:
            parser.error(f"unknown arm '{a}' (choices: {ARM_NAMES})")

    preset = MODEL_PRESETS[args.preset]
    workload = WORKLOAD_REGISTRY[args.workload]
    url = f"http://localhost:{args.port}"

    output_path = Path(args.output).resolve()
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure SGLang worker sources carry the oracle init hook (idempotent).
    hook_env = os.environ.copy()
    hook_env["SGLANG_ORACLE_VANILLA"] = "1"
    subprocess.run(
        [sys.executable, "-m", "simulation.oracle.install_hook"],
        env=hook_env, check=True, cwd=str(REPO_ROOT))

    summary: dict = {
        "preset": args.preset,
        "model": preset["model"],
        "draft_model": preset["draft_model"],
        "workload": args.workload,
        "steps": args.steps,
        "topk": 1,
        "num_draft_tokens": args.steps + 1,
        "n_tasks": args.n_tasks,
        "arms": {},
    }

    def _save():
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    _save()

    for arm in arms:
        print("=" * 72, file=sys.stderr)
        print(f"ARM {arm}: {args.workload} x {args.n_tasks} tasks, "
              f"S={args.steps}, topk=1", file=sys.stderr)
        print("=" * 72, file=sys.stderr)

        timing_log = Path(f"/tmp/sglang_ch_timing_{arm}_p{args.port}.jsonl")
        decision_log = Path(f"/tmp/sglang_ch_decisions_{arm}_p{args.port}.jsonl")
        for p in (timing_log, decision_log):
            p.unlink(missing_ok=True)

        env = build_env(args, arm, timing_log, decision_log)
        cmd = build_server_cmd(args, arm, preset)

        server_log = out_dir / f"server_{arm}.log"
        agent_out = out_dir / f"agent_results_{arm}.json"
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
                print(f"ERROR: {arm} server failed. See {server_log}",
                      file=sys.stderr)
                continue

            rc, wall = run_agent(args, workload, agent_out, env)
            arm_row["agent_rc"] = rc
            arm_row["wall_time_s"] = round(wall, 1)
            arm_row["agent_output"] = str(agent_out)

            if arm in ("baseline", "select1"):
                entries = read_timing_window(timing_log, 0)
                arm_row.update(summarize_entries(entries))
                kept_timing = out_dir / f"timing_{arm}.jsonl"
                if timing_log.exists():
                    shutil.copyfile(timing_log, kept_timing)
                    arm_row["timing_log"] = str(kept_timing)
            if arm == "select1" and decision_log.exists():
                kept_dec = out_dir / "decisions_select1.jsonl"
                shutil.copyfile(decision_log, kept_dec)
                arm_row["decision_log"] = str(kept_dec)

            summary["arms"][arm] = arm_row
            _save()
            print(f"  {arm}: wall={arm_row.get('wall_time_s')}s "
                  f"n={arm_row.get('n_samples', 0)} "
                  f"accept_mean={arm_row.get('accept_length_mean', float('nan')):.3f} "
                  f"step_ms={arm_row.get('step_ms', float('nan')):.2f}"
                  if arm != "suffix" else
                  f"  {arm}: wall={arm_row.get('wall_time_s')}s (reference)",
                  file=sys.stderr)
        finally:
            kill_server(proc)
            try:
                log_fh.close()
            except Exception:
                pass
            time.sleep(3)

    _save()
    print(f"\nSummary: {output_path}", file=sys.stderr)
    print("Next: python simulation/scripts/analyze_chain_hybrid.py "
          f"--summary {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
