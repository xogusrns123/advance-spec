#!/usr/bin/env python3
"""Reslice sweep orchestrator.

Runs `simulation.evaluation.run_tree_oracle_sim` for each (s, k) reslice in
the grid, producing one JSON output per reslice. The simulator itself sweeps
all enrolled methods (including the suffix hyperparameter grid for
extension_sfx*, hybrid_e3_sfx*, the count/score filter variants, etc.).

Usage:
    python3 simulation/scripts/experiments/run_reslice_sweep.py \\
        --workload swebench_verified \\
        --reslices s1k16,s2k16,s2k8,s4k16,s6k16 \\
        --budgets 8,16,32,64 \\
        [--methods 'extension_oracle,extension_sfx_oracle:,...']
        [--max-parallel 1]

Outputs go to simulation/results/explorations/sim_<workload>_<reslice>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HOST_ROOT = Path("/home/muchwater/advance-spec")
DOCKER_ROOT = Path("/workspace")
RESULTS_DIR_REL = Path("simulation/results/qwen3_14b")
OUT_DIR_REL = Path("simulation/results/explorations")
DATASET_MAP = {  # workload → dataset path RELATIVE to project root
    "swebench_verified": Path("data/swebench_verified/dataset_interleaved.jsonl"),
    "specbench": Path("data/specbench/dataset.jsonl"),
    "bfcl_v3": Path("data/bfcl_multi_turn/dataset_stratified_interleaved.jsonl"),
    "bfcl_v4": Path("data/bfcl_agent/dataset_stratified_interleaved.jsonl"),
    "longbench_lcc": Path("data/longbench_lcc/dataset_interleaved.jsonl"),
    "longbench_repobench": Path("data/longbench_repobench/dataset_interleaved.jsonl"),
}

# Default method set (current spec — see simulation/config/sim_qwen3_14b.yaml).
DEFAULT_METHODS = ",".join([
    "single:eagle3", "single:suffix", "single:draft_model",
    "hybrid_e3:",      "hybrid_oracle:",
    "extension:",      "extension_oracle:",
    "extension_by_count:", "extension_by_score:",
    "extension_prune_pt:",
    # Draft-model-backbone extension family (parallel to eagle3-base):
    "extension_dm:",   "extension_dm_oracle:",
    "extension_dm_by_count:", "extension_dm_by_score:",
])


def parse_reslice(s: str) -> tuple[int, int]:
    """Parse 's2k16' → (2, 16)."""
    m = re.match(r"^s(\d+)k(\d+)$", s)
    if not m:
        raise ValueError(f"reslice must look like 's2k16', got {s!r}")
    return int(m.group(1)), int(m.group(2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True, choices=list(DATASET_MAP.keys()))
    p.add_argument("--reslices", required=True,
                   help="comma-separated reslice tags, e.g. 's1k16,s2k16,s4k16'")
    p.add_argument("--budgets", default="8,16,32,64",
                   help="comma-separated budgets")
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--capture-steps", type=int, default=8)
    p.add_argument("--capture-topk", type=int, default=16)
    p.add_argument("--out-dir", default=None,
                   help="default: simulation/results/explorations under host root")
    p.add_argument("--timeout-sec", type=int, default=5400)
    p.add_argument("--max-parallel", type=int, default=1,
                   help="run N reslices in parallel (CARE: each consumes ~50GB)")
    p.add_argument("--in-docker", action="store_true",
                   help="execute via `docker exec sglang-bench` (recommended)")
    p.add_argument("--docker-name", default="sglang-bench")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Compute paths relative to project root, then prefix host vs docker root.
    cap_rel = RESULTS_DIR_REL / f"{args.workload}_steps{args.capture_steps}_topk{args.capture_topk}_capture"
    agent_rel = cap_rel / "agent_results_eagle3.json"
    latency_rel = cap_rel / "latency_config.json"
    dataset_rel = DATASET_MAP[args.workload]

    # Sanity-check via host paths (always exist on host if mounted)
    if not args.dry_run:
        for rel in [agent_rel, latency_rel, dataset_rel]:
            if not (HOST_ROOT / rel).exists():
                sys.exit(f"missing on host: {HOST_ROOT / rel}")

    out_root = HOST_ROOT  # output JSONs always written to host (shared mount)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = HOST_ROOT / out_dir
        out_dir_rel = out_dir.relative_to(HOST_ROOT)
    else:
        out_dir = out_root / OUT_DIR_REL
        out_dir_rel = OUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    # When running inside docker, the simulator must write to /workspace path
    out_dir_docker = DOCKER_ROOT / out_dir_rel  # mirror of host path

    reslices = [parse_reslice(s) for s in args.reslices.split(",")]
    print(f"Sweep: workload={args.workload}, reslices={reslices}, "
          f"budgets={args.budgets}", flush=True)

    procs: list[tuple[subprocess.Popen, str, float]] = []
    completed = 0

    def launch(s: int, k: int):
        tag = f"s{s}k{k}"
        out_json_host = out_dir / f"sim_{args.workload}_{tag}_full.json"
        log_path_host = out_dir / f"sim_{args.workload}_{tag}_full.log"
        # Pick paths based on runner (host vs docker)
        if args.in_docker:
            base = DOCKER_ROOT
            out_json = out_dir_docker / out_json_host.name
        else:
            base = HOST_ROOT
            out_json = out_json_host
        cmd_inner = [
            "python3", "-m", "simulation.evaluation.run_tree_oracle_sim",
            "--agent-results", str(base / agent_rel),
            "--dataset", str(base / dataset_rel),
            "--model", "Qwen/Qwen3-14B",
            "--latency-config", str(base / latency_rel),
            "--steps", str(s), "--topk", str(k),
            "--reslice-steps", str(s), "--reslice-topk", str(k),
            "--capture-steps", str(args.capture_steps),
            "--capture-topk", str(args.capture_topk),
            "--budgets", args.budgets,
            "--methods", args.methods,
            "--output", str(out_json),
            "--print-summary",
        ]
        # Auto-attach Stage-2 draft-model drafts when present so single:draft_model
        # (and any extension variant that consults per_proposer["draft_model"])
        # has data to read. Sites that don't want draft_model can pass
        # --methods explicitly to suppress its dispatch.
        dm_rel = cap_rel / "draft_model_drafts.jsonl"
        dm_partial_rel = cap_rel / "draft_model_drafts_partial.jsonl"
        dm_pick = None
        if (HOST_ROOT / dm_rel).exists():
            dm_pick = dm_rel
        elif (HOST_ROOT / dm_partial_rel).exists():
            dm_pick = dm_partial_rel
        if dm_pick is not None:
            cmd_inner += ["--draft-model-drafts", str(base / dm_pick)]
        if args.in_docker:
            inner = " ".join(shlex.quote(a) for a in cmd_inner)
            sim_par = os.environ.get("SIM_PARALLEL", "1")
            wrapped = (f"cd {DOCKER_ROOT} && SIM_PARALLEL={sim_par} timeout {args.timeout_sec} "
                       f"{inner} > /tmp/sim_{args.workload}_{tag}_full.log 2>&1")
            cmd = ["docker", "exec", args.docker_name, "bash", "-c", wrapped]
        else:
            cmd = ["timeout", str(args.timeout_sec)] + cmd_inner
            log_path_host.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            print(f"DRY: would run {tag}: {' '.join(cmd[:8])}...", flush=True)
            return None
        print(f"  launching {tag} → {out_json_host.name}", flush=True)
        if args.in_docker:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        else:
            log_f = open(log_path_host, "w")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        return (proc, tag, time.time())

    queue = list(reslices)
    while queue or procs:
        # launch up to max_parallel
        while queue and len(procs) < args.max_parallel:
            s, k = queue.pop(0)
            entry = launch(s, k)
            if entry is not None:
                procs.append(entry)

        # poll
        still_running = []
        for proc, tag, start_t in procs:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, tag, start_t))
            else:
                elapsed = time.time() - start_t
                completed += 1
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                print(f"  [{completed}/{len(reslices)}] {tag}: {status} "
                      f"({elapsed:.0f}s)", flush=True)
        procs = still_running

        if procs:
            time.sleep(15)

    print(f"\nDone. {completed} reslices completed.", flush=True)


if __name__ == "__main__":
    main()
