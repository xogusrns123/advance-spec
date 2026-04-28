#!/usr/bin/env python3
"""Config-driven multi-workload sim sweep orchestrator.

Reads a YAML config (e.g. ``simulation/config/sim_full_sweep_qwen3_14b.yaml``)
and launches ``run_reslice_sweep.py`` for each listed workload sequentially.

Usage:
    python3 simulation/scripts/experiments/run_sim_sweeps.py \\
        simulation/config/sim_full_sweep_qwen3_14b.yaml [--dry-run]

Each per-workload sim sweep runs ``run_reslice_sweep.py`` which itself
iterates the reslice grid serially. Per-workload runs are also serial
so RAM-heavy captures (specbench/swebench) don't OOM each other.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SWEEP_SCRIPT = (REPO_ROOT / "simulation" / "scripts" / "experiments"
                / "run_reslice_sweep.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="Path to sim sweep YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan only, don't run")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name", "(unnamed)")
    workloads = cfg.get("workloads") or []
    reslices = ",".join(cfg.get("reslices") or [])
    budgets = ",".join(str(b) for b in (cfg.get("budgets") or []))
    timeout = int(cfg.get("timeout_sec", 0))
    max_parallel = int(cfg.get("max_parallel", 1))
    in_docker = bool(cfg.get("in_docker", True))
    out_dir = cfg.get("out_dir")
    capture_steps = cfg.get("capture_steps")
    capture_topk = cfg.get("capture_topk")
    global_methods = cfg.get("methods")  # global override for --methods
    overrides = cfg.get("workload_overrides", {}) or {}

    if not workloads:
        print(f"ERROR: '{args.config}' has no workloads", file=sys.stderr)
        return 2
    if not reslices or not budgets:
        print(f"ERROR: '{args.config}' missing reslices/budgets",
              file=sys.stderr)
        return 2

    print(f"Sim sweep config: {name}")
    print(f"  workloads ({len(workloads)}): {workloads}")
    print(f"  reslices: {reslices}")
    print(f"  budgets: {budgets}")
    print(f"  timeout_sec: {timeout} ({'no limit' if timeout == 0 else 'limit'})")
    print(f"  max_parallel: {max_parallel}")
    print(f"  in_docker: {in_docker}")
    print(f"  out_dir: {out_dir}")
    if global_methods:
        print(f"  methods (global): {global_methods}")
    print()

    if args.dry_run:
        for wl in workloads:
            print(f"  dry-run: would sweep {wl}")
        return 0

    rcs: list[tuple[str, int, float]] = []
    for wl in workloads:
        cmd = [
            sys.executable, str(SWEEP_SCRIPT),
            "--workload", wl,
            "--reslices", reslices,
            "--budgets", budgets,
            "--max-parallel", str(max_parallel),
            "--timeout-sec", str(timeout),
        ]
        if in_docker:
            cmd.append("--in-docker")
        if out_dir:
            cmd += ["--out-dir", out_dir]
        if capture_steps is not None:
            cmd += ["--capture-steps", str(capture_steps)]
        if capture_topk is not None:
            cmd += ["--capture-topk", str(capture_topk)]
        ovr = overrides.get(wl, {}) or {}
        # Per-workload methods override > global methods > sweep defaults
        methods_to_use = ovr.get("methods") or global_methods
        if methods_to_use:
            cmd += ["--methods", methods_to_use]
        if ovr.get("budgets"):
            cmd[cmd.index("--budgets") + 1] = ",".join(
                str(b) for b in ovr["budgets"])
        if ovr.get("reslices"):
            cmd[cmd.index("--reslices") + 1] = ",".join(ovr["reslices"])

        print(f"\n=== Workload: {wl} ===")
        print(f"  cmd: {' '.join(cmd[:8])} ...")
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
        dt = time.time() - t0
        print(f"  → rc={rc} ({dt/60:.1f} min)")
        rcs.append((wl, rc, dt))

    print()
    print("=== Summary ===")
    for wl, rc, dt in rcs:
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"  {wl:<25} {status:<12} {dt/60:>6.1f} min")
    return 0 if all(rc == 0 for _, rc, _ in rcs) else 1


if __name__ == "__main__":
    sys.exit(main())
