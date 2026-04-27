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

VALID_WORKLOADS = {
    "bfcl_v3", "bfcl_v4", "specbench", "swebench",
    "swebench_verified", "longbench_lcc", "longbench_repobench",
}
VALID_PRESETS = {"glm4_flash", "qwen3_8b", "qwen3_14b", "qwen3_32b", "llama3_8b"}

# Model presets — lifted from run_pipeline.sh for round-robin mode (which
# bypasses the shell script). Sweep mode still uses run_pipeline.sh.
MODEL_PRESETS: dict[str, dict] = {
    "glm4_flash": {
        "model": "zai-org/GLM-4.7-Flash",
        "draft_model": "thoughtworks/GLM-4.7-Flash-Eagle3",
        "draft_lm": None,
        "tool_call_parser": "qwen25",
    },
    "qwen3_8b": {
        "model": "Qwen/Qwen3-8B",
        "draft_model": "AngelSlim/Qwen3-8B_eagle3",
        "draft_lm": "Qwen/Qwen3-0.6B",
        "tool_call_parser": "qwen25",
    },
    "qwen3_14b": {
        "model": "Qwen/Qwen3-14B",
        "draft_model": "AngelSlim/Qwen3-14B_eagle3",
        "draft_lm": "Qwen/Qwen3-0.6B",
        "tool_call_parser": "qwen25",
    },
    "qwen3_32b": {
        "model": "Qwen/Qwen3-32B",
        "draft_model": "Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3",
        "draft_lm": "Qwen/Qwen3-0.6B",
        "tool_call_parser": "qwen25",
    },
    "llama3_8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "draft_model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "draft_lm": "meta-llama/Llama-3.2-1B-Instruct",
        "tool_call_parser": "llama3",
    },
}

# Per-workload config for round-robin mode: which agent module, which
# dataset (relative to repo root), and any agent-specific extra flags
# that come from `workload_overrides`.
WORKLOAD_REGISTRY: dict[str, dict] = {
    "specbench": {
        "agent_module": "simulation.agents.specbench_agent",
        "dataset": "data/specbench/dataset_interleaved.jsonl",
    },
    "bfcl_v3": {
        "agent_module": "simulation.agents.bfcl_agent",
        "dataset": "data/bfcl_multi_turn/dataset_stratified_interleaved.jsonl",
    },
    "bfcl_v4": {
        "agent_module": "simulation.agents.bfcl_v4_agent",
        "dataset": "data/bfcl_agent/dataset_stratified_interleaved.jsonl",
    },
    "swebench": {
        "agent_module": "simulation.agents.swebench_agent",
        "dataset": "data/swebench/dataset.jsonl",
    },
    "swebench_verified": {
        "agent_module": "simulation.agents.swebench_agent",
        "dataset": "data/swebench_verified/dataset_interleaved.jsonl",
    },
    "longbench_lcc": {
        "agent_module": "simulation.agents.specbench_agent",
        "dataset": "data/longbench_lcc/dataset_interleaved.jsonl",
    },
    "longbench_repobench": {
        "agent_module": "simulation.agents.specbench_agent",
        "dataset": "data/longbench_repobench/dataset_interleaved.jsonl",
    },
}

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

    wl = run["workload"]
    if wl in ("bfcl_v3", "bfcl_v4") and run.get("max_iterations") is not None:
        env["BFCL_MAX_ITER"] = str(run["max_iterations"])
    if wl == "bfcl_v4" and run.get("input_file"):
        env["BFCL_V4_INPUT"] = run["input_file"]
    if wl == "swebench" and run.get("max_iterations") is not None:
        env["SWE_MAX_ITER"] = str(run["max_iterations"])
    if wl == "specbench" and run.get("max_tokens_override") is not None:
        env["MAX_TOKENS_OVERRIDE"] = str(run["max_tokens_override"])

    s2 = cfg.get("stage2", cfg.get("stage3b", {})) or {}
    if s2.get("max_draft_tokens") is not None:
        env["STAGE2_MAX_TOKENS"] = str(s2["max_draft_tokens"])

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


def _wait_for_server(url: str, timeout: int = 600) -> bool:
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _kill_proc(proc: subprocess.Popen) -> None:
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def execute_round_robin(cfg: dict, dry_run: bool = False) -> int:
    """Run round-robin Stage 1 collection: one sglang server, all workloads
    rotate `batch` requests per iteration with --resume.

    Replaces the legacy `_stage1_round_robin.sh`. Each workload's agent is
    launched per-iteration; the sglang server stays alive across all.
    """
    rr = cfg.get("round_robin", {}) or {}
    preset_name = cfg.get("model_preset")
    if preset_name not in MODEL_PRESETS:
        print(f"ERROR: model_preset '{preset_name}' missing or unknown for "
              f"round_robin mode", file=sys.stderr)
        return 2
    preset = MODEL_PRESETS[preset_name]
    # Allow per-config model overrides (same as sweep mode).
    models = cfg.get("models", {}) or {}
    target_model = models.get("target_model") or preset["model"]
    draft_model = models.get("draft_model") or preset["draft_model"]
    tool_call_parser = preset["tool_call_parser"]

    # RR uses a single (steps, topk, num_draft_tokens). If stage1_configs
    # has multiple entries (sweep mode), reject — RR is single-config.
    stage1_configs = cfg.get("stage1_configs") or []
    steps_list = as_list(cfg.get("stage1_steps"))
    if len(stage1_configs) != 1 or len(steps_list) != 1:
        print("ERROR: round_robin mode requires a single stage1_steps and a "
              "single stage1_configs entry (no sweep)", file=sys.stderr)
        return 2
    s1c = stage1_configs[0]
    steps = int(steps_list[0])
    topk = int(s1c["topk"])
    ndt = int(rr.get("num_draft_tokens", s1c.get("num_draft_tokens", 2)))

    workloads = as_list(cfg.get("workloads"))
    for wl in workloads:
        if wl not in WORKLOAD_REGISTRY:
            print(f"ERROR: unknown workload '{wl}' in round_robin mode",
                  file=sys.stderr)
            return 2

    overrides = cfg.get("workload_overrides", {}) or {}
    defaults = cfg.get("defaults", {}) or {}
    output_root = (cfg.get("output", {}) or {}).get("root", "simulation/results")
    capture_full_pool = bool(rr.get("capture_full_pool", True))
    batch = int(rr.get("batch", 1))
    resume = bool(rr.get("resume", True))

    # cfg_tag goes into the output dir suffix.
    cfg_tag = (rr.get("cfg_tag")
               or f"steps{steps}_topk{topk}"
               + ("_capture" if capture_full_pool else f"_b{ndt}"))
    out_base = Path(output_root) / preset_name.lower()

    infra = cfg.get("infra", {}) or {}
    default_port = int(infra.get("port", 30000))
    default_gpu_ids = infra.get("gpu_ids") or [0]

    # Build per-workload invocation specs.
    plan: list[dict] = []
    for wl in workloads:
        reg = WORKLOAD_REGISTRY[wl]
        merged = merge_workload_overrides(defaults, overrides, wl)
        out_dir = out_base / f"{wl}_{cfg_tag}"
        spec = {
            "workload": wl,
            "agent_module": reg["agent_module"],
            "dataset": str(REPO_ROOT / reg["dataset"]),
            "out_dir": out_dir,
            "out_file": str(out_dir / "agent_results_eagle3.json"),
            "extra_flags": _build_agent_extra_flags(wl, merged),
        }
        plan.append(spec)

    # Shard layout: each shard owns a disjoint workload subset and runs its
    # own SGLang server on its own GPU(s)/port. If no `shards:` block is set,
    # fall back to a single shard with all workloads (legacy behavior).
    shards_cfg = cfg.get("shards") or []
    if shards_cfg:
        seen_wls: set[str] = set()
        for sh in shards_cfg:
            sh_wls = list(sh.get("workloads") or [])
            for w in sh_wls:
                if w not in workloads:
                    print(f"ERROR: shard workload '{w}' not in top-level "
                          f"workloads {workloads}", file=sys.stderr)
                    return 2
                if w in seen_wls:
                    print(f"ERROR: workload '{w}' appears in multiple shards",
                          file=sys.stderr)
                    return 2
                seen_wls.add(w)
        missing = set(workloads) - seen_wls
        if missing:
            print(f"ERROR: workloads {missing} not assigned to any shard",
                  file=sys.stderr)
            return 2
        shard_specs = []
        for i, sh in enumerate(shards_cfg):
            sh_wls = set(sh.get("workloads") or [])
            shard_specs.append({
                "id": str(sh.get("id", i)),
                "gpu_ids": list(sh.get("gpu_ids") or [i]),
                "port": int(sh.get("port", default_port + i)),
                "plan": [s for s in plan if s["workload"] in sh_wls],
            })
    else:
        shard_specs = [{
            "id": "0",
            "gpu_ids": default_gpu_ids,
            "port": default_port,
            "plan": plan,
        }]

    print(f"Experiment: {cfg.get('name', '(unnamed)')} (round-robin)")
    print(f"Model: {target_model}  draft: {draft_model}")
    print(f"Config: steps={steps}  topk={topk}  num_draft_tokens={ndt}  "
          f"capture_full_pool={capture_full_pool}  batch={batch}")
    print(f"Workloads ({len(workloads)}): {', '.join(workloads)}")
    print(f"Output base: {out_base}")
    print(f"Shards ({len(shard_specs)}):")
    for ss in shard_specs:
        ss_wls = [s["workload"] for s in ss["plan"]]
        print(f"  [shard {ss['id']}] gpu_ids={ss['gpu_ids']}  "
              f"port={ss['port']}  workloads={ss_wls}")
    for s in plan:
        print(f"  [{s['workload']:<22}] dataset={s['dataset']}")
        print(f"  {' ' * 24}  out={s['out_file']}")
        print(f"  {' ' * 24}  extra={' '.join(s['extra_flags']) or '(none)'}")
    if dry_run:
        print("\nDry run — not executing.")
        return 0

    out_base.mkdir(parents=True, exist_ok=True)
    for s in plan:
        s["out_dir"].mkdir(parents=True, exist_ok=True)

    # Install hook once (idempotent, host-wide).
    base_env = os.environ.copy()
    print("Installing oracle hook…")
    subprocess.run([sys.executable, "-m", "simulation.oracle.install_hook"],
                   env=base_env, check=True, cwd=str(REPO_ROOT))

    shard_kwargs = dict(
        target_model=target_model, draft_model=draft_model,
        tool_call_parser=tool_call_parser,
        steps=steps, topk=topk, ndt=ndt,
        capture_full_pool=capture_full_pool,
        batch=batch, resume=resume,
        out_base=out_base, base_env=base_env,
    )

    if len(shard_specs) == 1:
        ss = shard_specs[0]
        rc = _run_rr_shard(
            shard_id=ss["id"], gpu_ids=ss["gpu_ids"],
            port=ss["port"], plan=ss["plan"], **shard_kwargs)
        print("Round-robin done.")
        return rc

    # Multi-shard: run each shard in its own thread.
    import concurrent.futures
    rcs: list[int] = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(shard_specs)) as ex:
        futs = {
            ex.submit(_run_rr_shard,
                      shard_id=ss["id"], gpu_ids=ss["gpu_ids"],
                      port=ss["port"], plan=ss["plan"],
                      **shard_kwargs): ss
            for ss in shard_specs
        }
        for fut in concurrent.futures.as_completed(futs):
            ss = futs[fut]
            try:
                rc = fut.result()
                rcs.append(rc)
                print(f"[shard {ss['id']}] DONE rc={rc}")
            except Exception as e:
                print(f"[shard {ss['id']}] EXCEPTION: {e}", file=sys.stderr)
                rcs.append(1)

    print("Round-robin done.")
    return 0 if all(r == 0 for r in rcs) else 1


def _run_rr_shard(
    *, shard_id: str, gpu_ids: list, port: int, plan: list,
    target_model: str, draft_model: str, tool_call_parser: str,
    steps: int, topk: int, ndt: int, capture_full_pool: bool,
    batch: int, resume: bool, out_base, base_env: dict,
) -> int:
    """Boot one SGLang server on (gpu_ids, port) and run RR over `plan`."""
    rr_env = base_env.copy()
    rr_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    rr_env["SGLANG_ORACLE_VANILLA"] = "1"
    if capture_full_pool:
        rr_env["SGLANG_CAPTURE_FULL_POOL"] = "1"
    rr_env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    rr_env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    for k in ("SGLANG_ORACLE_REPLAY", "SGLANG_ORACLE_VERIFY_TRIES"):
        rr_env.pop(k, None)

    srv_log = out_base / f"_rr_sglang_server_shard{shard_id}.log"
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", target_model,
        "--tp-size", str(len(gpu_ids)),
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", draft_model,
        "--speculative-num-steps", str(steps),
        "--speculative-eagle-topk", str(topk),
        "--speculative-num-draft-tokens", str(ndt),
        "--tool-call-parser", tool_call_parser,
        "--mem-fraction-static", "0.85",
        "--disable-cuda-graph",
        "--watchdog-timeout", "600",
        "--host", "0.0.0.0", "--port", str(port),
    ]
    print(f"[shard {shard_id}] Launching SGLang on port {port} "
          f"GPU={gpu_ids} (log: {srv_log})…")
    log_fh = open(srv_log, "w")
    proc = subprocess.Popen(cmd, env=rr_env, stdout=log_fh,
                            stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    url = f"http://localhost:{port}"
    try:
        if not _wait_for_server(url):
            print(f"[shard {shard_id}] ERROR: SGLang failed to start. "
                  f"See {srv_log}", file=sys.stderr)
            return 1
        log_fh.close()
        print(f"[shard {shard_id}] Server ready. Starting RR loop.")

        iter_idx = 0
        while True:
            iter_idx += 1
            progress = 0
            for s in plan:
                done_n, total_n = _count_progress(s["out_file"], s["dataset"])
                if total_n is not None and done_n >= total_n:
                    print(f"[shard {shard_id}][iter {iter_idx}] "
                          f"{s['workload']}: exhausted ({done_n}/{total_n}) "
                          f"— skipping")
                    continue
                print(f"[shard {shard_id}][iter {iter_idx}] "
                      f"{s['workload']}: running batch={batch} "
                      f"(resume={resume}, current={done_n}/{total_n or '?'})")
                rc = _run_agent_once(
                    s, url=url, model=target_model, batch=batch,
                    resume=resume, env=rr_env)
                if rc == 0:
                    progress += 1
            if progress == 0:
                print(f"[shard {shard_id}] All workloads exhausted; "
                      f"exiting after iter={iter_idx}.")
                break
    finally:
        print(f"[shard {shard_id}] Stopping server (PID={proc.pid})…")
        _kill_proc(proc)
        try:
            log_fh.close()
        except Exception:
            pass
    return 0


def _build_agent_extra_flags(workload: str, merged: dict) -> list[str]:
    """Translate workload_overrides into agent CLI flags."""
    flags: list[str] = []
    if workload == "specbench":
        if merged.get("max_tokens_override") is not None:
            flags += ["--max-tokens", str(merged["max_tokens_override"])]
    elif workload in ("longbench_lcc", "longbench_repobench"):
        # specbench_agent serves long-bench too
        flags += ["--max-tokens", str(merged.get("max_tokens", 256))]
    elif workload == "bfcl_v3":
        if merged.get("max_iterations") is not None:
            flags += ["--max-iterations", str(merged["max_iterations"])]
    elif workload == "bfcl_v4":
        if merged.get("max_iterations") is not None:
            flags += ["--max-iterations", str(merged["max_iterations"])]
        if merged.get("include_category"):
            flags += ["--include-category", str(merged["include_category"])]
    elif workload in ("swebench", "swebench_verified"):
        if merged.get("max_iterations") is not None:
            flags += ["--max-iterations", str(merged["max_iterations"])]
        if merged.get("tool_style"):
            flags += ["--tool-style", str(merged["tool_style"])]
        if merged.get("repos_dir"):
            flags += ["--repos-dir", str(merged["repos_dir"])]
        elif workload == "swebench_verified":
            # default repo cache path used by rr
            flags += ["--repos-dir", "data/swebench_verified/repos"]
    return flags


def _count_progress(out_file: str, dataset: str) -> tuple[int, int | None]:
    """Return (done_count, total_count). Reads partial first if present."""
    import json as _json
    candidates = [out_file + ".partial", out_file]
    done = 0
    for p in candidates:
        if Path(p).is_file():
            try:
                d = _json.load(open(p))
                done = max(done, len(d.get("questions", [])))
            except Exception:
                pass
    total = None
    if Path(dataset).is_file():
        try:
            with open(dataset) as f:
                total = sum(1 for _ in f)
        except Exception:
            pass
    return done, total


def _run_agent_once(spec: dict, *, url: str, model: str, batch: int,
                    resume: bool, env: dict) -> int:
    """Invoke the agent module once with --num-requests batch [--resume]."""
    cmd = [sys.executable, "-m", spec["agent_module"],
           "--url", f"{url}/v1", "--model", model,
           "--input-file", spec["dataset"],
           "--output-file", spec["out_file"],
           "--num-requests", str(batch),
           "--num-workers", "1"]
    if resume:
        cmd.append("--resume")
    cmd.extend(spec["extra_flags"])

    log_path = spec["out_dir"] / "_rr_agent.log"
    with open(log_path, "a") as lf:
        rc = subprocess.call(cmd, env=env, stdout=lf,
                             stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    if rc != 0:
        print(f"  → rc={rc} (see {log_path})")
    else:
        done, total = _count_progress(spec["out_file"], spec["dataset"])
        print(f"  → ok  done={done}/{total or '?'}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="Path to experiment YAML")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan only; do not execute")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rr_cfg = (cfg.get("round_robin") or {})
    if rr_cfg.get("enabled"):
        return execute_round_robin(cfg, dry_run=args.dry_run)

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
