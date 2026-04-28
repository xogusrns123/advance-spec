#!/usr/bin/env python3
"""Config-driven round-robin experiment runner.

Reads a YAML config from simulation/config/<name>.yaml and runs Stage 1
collection (EAGLE3 oracle capture) in round-robin mode: one or more
SGLang servers boot once, all workloads share each server, and per
iteration each workload runs `batch` more requests via --resume. With
a `shards:` block the runner launches N parallel servers (one per
shard) on different GPUs/ports; each shard owns a disjoint workload
subset and runs an independent RR loop.

Usage:
    python3 simulation/scripts/run_experiment.py <config.yaml> [--dry-run]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

# Model presets used by round-robin mode to launch SGLang.
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
    "spider2_dbt": {
        "agent_module": "simulation.agents.spider2_dbt_agent",
        "dataset": "data/spider2_dbt/spider2-dbt.jsonl",
    },
}

def as_list(v: Any) -> list:
    if v is None:
        return [None]
    if isinstance(v, list):
        return v
    return [v]


def merge_workload_overrides(defaults: dict, overrides: dict, workload: str) -> dict:
    merged = dict(defaults or {})
    merged.update((overrides or {}).get(workload, {}) or {})
    return merged


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
    context_length = rr.get("context_length")
    if context_length is not None:
        context_length = int(context_length)

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
        context_length=context_length,
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
    context_length: int | None = None,
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
        "--max-running-requests", "1",
        "--max-prefill-tokens", "8192",
        "--kv-cache-dtype", "fp8_e5m2",
        "--disable-cuda-graph",
        "--watchdog-timeout", "600",
        "--host", "0.0.0.0", "--port", str(port),
    ]
    if context_length is not None:
        cmd += ["--context-length", str(context_length)]
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
    elif workload == "spider2_dbt":
        if merged.get("max_iterations") is not None:
            flags += ["--max-iterations", str(merged["max_iterations"])]
        if merged.get("instances_dir"):
            flags += ["--instances-dir", str(merged["instances_dir"])]
        else:
            flags += ["--instances-dir", "data/spider2_dbt/instances"]
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
    if not rr_cfg.get("enabled"):
        print("ERROR: config must have `round_robin.enabled: true` — "
              "this runner only supports round-robin Stage 1 collection.",
              file=sys.stderr)
        return 2
    return execute_round_robin(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
