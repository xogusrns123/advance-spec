#!/usr/bin/env python3
"""Config-driven experiment runner.

Reads a YAML config from simulation/config/<name>.yaml, expands sweep
axes (workloads × stage1_steps × stage1_configs × model_preset),
merges workload overrides, validates EAGLE3 tree capacity, and invokes
run_pipeline.sh with the appropriate env vars per run.

Parallel mode: if infra.num_workers > 1, the runner launches that many
concurrent pipeline workers. Each worker claims
`infra.num_gpus // num_workers` GPUs and its own port range (offset by
worker index) so multiple pipelines can coexist on the same host.

Usage:
    python3 simulation/scripts/run_experiment.py <config.yaml> [--dry-run]
"""
import argparse
import itertools
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = REPO_ROOT / "simulation" / "scripts" / "run_pipeline.sh"

VALID_WORKLOADS = {"bfcl_v3", "bfcl_v4", "specbench", "swebench"}
VALID_PRESETS = {"glm4_flash", "qwen3_8b", "qwen3_14b", "qwen3_32b", "llama3_8b"}

# Per-worker port offsets. Base ports are spaced 100 apart so each
# worker's Stage 1/3b/3c sub-servers never collide.
WORKER_PORT_STRIDE = 100
STAGE1_BASE = 30000
STAGE3B_BASE = 31000
STAGE3C_BASE = 32000


def as_list(v: Any) -> list:
    if v is None:
        return [None]
    if isinstance(v, list):
        return v
    return [v]


def eagle3_capacity(topk: int, steps: int) -> int:
    return topk + (steps - 1) * topk * topk + 1


def merge_workload_overrides(defaults: dict, overrides: dict, workload: str) -> dict:
    merged = dict(defaults or {})
    merged.update((overrides or {}).get(workload, {}) or {})
    return merged


def expand_runs(cfg: dict) -> list[dict]:
    workloads = as_list(cfg.get("workloads"))
    steps_list = as_list(cfg.get("stage1_steps"))
    presets = as_list(cfg.get("model_preset"))
    stage1_configs = cfg.get("stage1_configs") or [
        {"topk": 8, "num_draft_tokens": 256,
         "sim_budgets": [1, 2, 4, 8, 16, 32, 64, 128]}
    ]

    runs = []
    for workload, steps, s1c, preset in itertools.product(
        workloads, steps_list, stage1_configs, presets
    ):
        merged = merge_workload_overrides(
            cfg.get("defaults", {}),
            cfg.get("workload_overrides", {}),
            workload,
        )
        runs.append({
            "workload": workload,
            "model_preset": preset,
            "stage1_steps": steps,
            "stage1_topk": s1c["topk"],
            "stage1_num_draft_tokens": s1c["num_draft_tokens"],
            "sim_budgets": s1c.get("sim_budgets"),
            "req_start": merged.get("req_start"),
            "req_end": merged.get("req_end"),
            "max_tokens_override": merged.get("max_tokens_override"),
            "max_iterations": merged.get("max_iterations"),
            "input_file": merged.get("input_file"),
        })
    return runs


def render_suffix(tmpl: str | None, run: dict) -> str:
    fields = {
        "steps": run["stage1_steps"],
        "topk": run["stage1_topk"],
        "num_draft_tokens": run["stage1_num_draft_tokens"],
        "workload": run["workload"],
        "preset": run["model_preset"],
        "req_start": run["req_start"],
        "req_end": run["req_end"],
    }
    if not tmpl:
        return f"steps{fields['steps']}_topk{fields['topk']}"
    return tmpl.format(**fields)


def compute_output_dir(root: str, run: dict, suffix: str) -> Path:
    base = Path(root) / run["model_preset"].lower() / f"{run['workload']}_{suffix}"
    rs, re_ = run.get("req_start"), run.get("req_end")
    if rs is not None and re_ is not None:
        base = Path(f"{base}_req{rs}-{re_}")
    return base


def validate_run(run: dict) -> str | None:
    if run["workload"] not in VALID_WORKLOADS:
        return f"unknown workload: {run['workload']}"
    if run["model_preset"] not in VALID_PRESETS:
        return f"unknown model_preset: {run['model_preset']}"
    cap = eagle3_capacity(run["stage1_topk"], run["stage1_steps"])
    if run["stage1_num_draft_tokens"] > cap:
        return (f"num_draft_tokens={run['stage1_num_draft_tokens']} exceeds "
                f"EAGLE3 tree capacity={cap} "
                f"(topk={run['stage1_topk']}, steps={run['stage1_steps']})")
    if run["workload"] == "bfcl_v4" and run.get("input_file"):
        p = REPO_ROOT / run["input_file"]
        if not p.is_file():
            return f"bfcl_v4 input_file not found: {run['input_file']}"
    return None


def build_env(run: dict, cfg: dict, suffix: str,
              gpu_ids: list[int], worker_idx: int) -> dict:
    env = os.environ.copy()

    env["STAGE1_TOPK"] = str(run["stage1_topk"])
    env["STAGE1_STEPS"] = str(run["stage1_steps"])
    env["STAGE1_NUM_DRAFT_TOKENS"] = str(run["stage1_num_draft_tokens"])
    env["OUTPUT_DIR_SUFFIX"] = suffix

    if run.get("sim_budgets"):
        env["SIM_BUDGETS"] = ",".join(str(b) for b in run["sim_budgets"])

    if run.get("req_start") is not None:
        env["REQ_START"] = str(run["req_start"])
    if run.get("req_end") is not None:
        env["REQ_END"] = str(run["req_end"])

    stages = cfg.get("stages", {}) or {}
    env["UNION_TRIE"] = "1" if stages.get("union_trie") else "0"
    env["EU_ORACLE"] = "1" if stages.get("eu_oracle") else "0"

    wl = run["workload"]
    if wl in ("bfcl_v3", "bfcl_v4") and run.get("max_iterations") is not None:
        env["BFCL_MAX_ITER"] = str(run["max_iterations"])
    if wl == "bfcl_v4" and run.get("input_file"):
        env["BFCL_V4_INPUT"] = run["input_file"]
    if wl == "swebench" and run.get("max_iterations") is not None:
        env["SWE_MAX_ITER"] = str(run["max_iterations"])
    if wl == "specbench" and run.get("max_tokens_override") is not None:
        env["MAX_TOKENS_OVERRIDE"] = str(run["max_tokens_override"])

    s3b = cfg.get("stage3b", {}) or {}
    if s3b.get("max_draft_tokens") is not None:
        env["STAGE3B_MAX_TOKENS"] = str(s3b["max_draft_tokens"])

    # Optional per-model overrides. Each field overrides the preset default
    # in run_pipeline.sh via env (MODEL / DRAFT_MODEL / DRAFT_LM).
    models = cfg.get("models", {}) or {}
    if models.get("target_model"):
        env["MODEL"] = str(models["target_model"])
    if models.get("draft_model"):
        env["DRAFT_MODEL"] = str(models["draft_model"])
    if models.get("draft_lm") is not None:
        env["DRAFT_LM"] = str(models["draft_lm"])

    # Per-worker GPU & port assignment. NUM_GPUS is the count this worker
    # owns; GPU_IDS lists the actual indices. Port bases are offset so
    # parallel workers don't collide on localhost sockets.
    env["NUM_GPUS"] = str(len(gpu_ids))
    env["GPU_IDS"] = ",".join(str(g) for g in gpu_ids)
    env["STAGE1_BASE_PORT"] = str(STAGE1_BASE + worker_idx * WORKER_PORT_STRIDE)
    env["STAGE3B_BASE_PORT"] = str(STAGE3B_BASE + worker_idx * WORKER_PORT_STRIDE)
    env["PORT"] = str(STAGE3C_BASE + worker_idx * WORKER_PORT_STRIDE)

    return env


def assign_gpus(num_gpus: int, num_workers: int,
                gpu_ids: list[int] | None) -> list[list[int]]:
    """Split the GPU pool into `num_workers` disjoint groups."""
    pool = list(gpu_ids) if gpu_ids else list(range(num_gpus))
    if len(pool) < num_workers:
        raise ValueError(
            f"num_workers={num_workers} exceeds available GPUs={len(pool)}"
        )
    per = len(pool) // num_workers
    return [pool[i * per:(i + 1) * per] for i in range(num_workers)]


_print_lock = threading.Lock()


def tprint(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def execute_run(p: dict, cfg: dict, gpu_ids: list[int], worker_idx: int,
                counters: dict) -> None:
    r = p["run"]
    env = build_env(r, cfg, p["suffix"], gpu_ids, worker_idx)
    out_dir: Path = p["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_pipeline.log"

    tag = (f"w{worker_idx} gpus={gpu_ids} "
           f"{r['workload']} preset={r['model_preset']} "
           f"steps={r['stage1_steps']} topk={r['stage1_topk']} "
           f"budget={r['stage1_num_draft_tokens']}")
    tprint(f"[START]   {tag} → {out_dir}")

    t0 = time.time()
    with open(log_path, "w") as lf:
        rc = subprocess.call(
            ["bash", str(PIPELINE_SCRIPT),
             r["workload"], r["model_preset"]],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
        )
    dt = time.time() - t0

    if rc == 0:
        tprint(f"[OK]      {tag}  ({dt:.0f}s)  log: {log_path}")
        with _print_lock:
            counters["ok"] += 1
    else:
        tprint(f"[FAIL]    {tag}  rc={rc} ({dt:.0f}s)  log: {log_path}")
        with _print_lock:
            counters["fail"] += 1


def worker_loop(worker_idx: int, gpu_ids: list[int], q: "Queue[dict]",
                cfg: dict, counters: dict) -> None:
    while True:
        try:
            p = q.get_nowait()
        except Empty:
            return
        try:
            execute_run(p, cfg, gpu_ids, worker_idx, counters)
        finally:
            q.task_done()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="Path to experiment YAML")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan only; do not execute")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    stages = cfg.get("stages", {}) or {}
    if stages.get("eu_oracle") and not stages.get("union_trie"):
        print("ERROR: stages.eu_oracle=true requires stages.union_trie=true",
              file=sys.stderr)
        return 2

    infra = cfg.get("infra", {}) or {}
    num_gpus = int(infra.get("num_gpus") or 1)
    num_workers = int(infra.get("num_workers") or 1)
    gpu_ids_cfg = infra.get("gpu_ids")

    if num_workers < 1:
        print("ERROR: infra.num_workers must be >= 1", file=sys.stderr)
        return 2
    try:
        worker_gpus = assign_gpus(num_gpus, num_workers, gpu_ids_cfg)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    runs = expand_runs(cfg)
    output_root = (cfg.get("output", {}) or {}).get("root", "simulation/results")
    suffix_tmpl = (cfg.get("output", {}) or {}).get("suffix_template")
    skip_if_exists = (cfg.get("output", {}) or {}).get("skip_if_exists", True)

    plan = []
    for run in runs:
        err = validate_run(run)
        suffix = render_suffix(suffix_tmpl, run)
        out_dir = compute_output_dir(output_root, run, suffix)
        plan.append({"run": run, "suffix": suffix,
                     "out_dir": out_dir, "error": err})

    print(f"Experiment: {cfg.get('name', '(unnamed)')}")
    print(f"Config: {args.config}")
    print(f"Parallelism: num_workers={num_workers} "
          f"(each owns {len(worker_gpus[0])} GPU(s): "
          f"{', '.join(str(g) for g in worker_gpus)})")
    print(f"Total runs: {len(plan)}")
    print()

    for i, p in enumerate(plan):
        r = p["run"]
        marker = ""
        if p["error"]:
            marker = f" INVALID: {p['error']}"
        elif skip_if_exists and (p["out_dir"] / "tree_oracle_sim.json").exists():
            marker = " SKIP"
        rs_re = ""
        if r.get("req_start") is not None:
            rs_re = f" req={r['req_start']}-{r['req_end']}"
        print(f"  [{i:3d}] {r['workload']:<10} "
              f"preset={r['model_preset']:<10} "
              f"steps={r['stage1_steps']:<2} topk={r['stage1_topk']:<2} "
              f"budget={r['stage1_num_draft_tokens']:<4}{rs_re}"
              f"  {p['out_dir']}{marker}")

    if args.dry_run:
        print("\nDry run — not executing.")
        return 0

    # Enqueue runs that are neither invalid nor already done.
    q: "Queue[dict]" = Queue()
    n_skip = n_invalid = 0
    for p in plan:
        if p["error"]:
            tprint(f"[INVALID] {p['run']['workload']} "
                   f"steps={p['run']['stage1_steps']} "
                   f"topk={p['run']['stage1_topk']}: {p['error']}")
            n_invalid += 1
            continue
        if skip_if_exists and (p["out_dir"] / "tree_oracle_sim.json").exists():
            tprint(f"[SKIP]    {p['out_dir']}")
            n_skip += 1
            continue
        q.put(p)

    counters = {"ok": 0, "fail": 0}
    threads = []
    for widx in range(num_workers):
        t = threading.Thread(
            target=worker_loop,
            args=(widx, worker_gpus[widx], q, cfg, counters),
            daemon=False,
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print()
    print(f"Summary: ok={counters['ok']} skip={n_skip} "
          f"fail={counters['fail']} invalid={n_invalid} "
          f"/ total={len(plan)}")
    return 1 if counters["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
