"""Benchmark EAGLE3 across tree/chain configs on real workload (SpecBench).

Measures: accept rate, MAT, TPS, latency, GPU util, CPU util, etc.
for each (topk, steps, budget) configuration. Runs in NORMAL speculative
decoding mode (not oracle vanilla) to get real acceptance metrics.

Usage:
    # Default configs (~45 min):
    python3 scripts/bench_eagle3_configs.py \
        --model Qwen/Qwen3-8B \
        --draft-model Tengyunw/qwen3_8b_eagle3 \
        --dataset data/specbench/dataset.jsonl \
        --output simulation/results/qwen3_8b/eagle3_bench.json

    # Custom configs:
    python3 scripts/bench_eagle3_configs.py \
        --model Qwen/Qwen3-8B \
        --draft-model Tengyunw/qwen3_8b_eagle3 \
        --dataset data/specbench/dataset.jsonl \
        --configs "chain:1,4,16,64;tree:4-3:4,16,64;tree:8-3:8,32,64" \
        --output simulation/results/qwen3_8b/eagle3_bench.json

Config format:
    chain:B1,B2,...           → topk=1, steps=B, budget=B (chain of length B)
    tree:TOPK-STEPS:B1,B2,.. → topk=TOPK, steps=STEPS, budget=B1,B2,...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from openai import OpenAI


def wait_for_server(url: str, timeout: int = 300) -> bool:
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def kill_server(proc: subprocess.Popen):
    import psutil
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    proc.wait(timeout=30)


def load_specbench(path: str, n_requests: int = 10) -> list[dict]:
    """Load SpecBench prompts with balanced category sampling.

    Picks evenly across categories (round-robin) to ensure diversity.
    """
    from collections import defaultdict
    by_cat: dict[str, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            by_cat[d["category"]].append({
                "question_id": d["question_id"],
                "category": d["category"],
                "prompt": d["turns"][0],
            })

    # Round-robin across categories
    categories = sorted(by_cat.keys())
    prompts = []
    idx = 0
    while len(prompts) < n_requests:
        cat = categories[idx % len(categories)]
        cat_list = by_cat[cat]
        pick_idx = idx // len(categories)
        if pick_idx < len(cat_list):
            prompts.append(cat_list[pick_idx])
        idx += 1
        if idx >= n_requests * 10:  # safety
            break
    return prompts[:n_requests]


def monitor_gpu(interval: float, stop_event: threading.Event) -> list[dict]:
    """Sample GPU metrics at interval."""
    samples = []
    while not stop_event.is_set():
        try:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ], text=True, timeout=5)
            for line in out.strip().split("\n"):
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 6:
                    samples.append({
                        "gpu_idx": int(parts[0]),
                        "gpu_util": float(parts[1]),
                        "mem_util": float(parts[2]),
                        "mem_used_mb": float(parts[3]),
                        "mem_total_mb": float(parts[4]),
                        "power_w": float(parts[5]) if parts[5] != "[N/A]" else 0,
                        "time": time.time(),
                    })
        except Exception:
            pass
        stop_event.wait(interval)
    return samples


def monitor_cpu(interval: float, stop_event: threading.Event) -> list[float]:
    """Sample CPU utilization at interval."""
    import psutil
    samples = []
    while not stop_event.is_set():
        samples.append(psutil.cpu_percent(interval=0.1))
        stop_event.wait(interval)
    return samples


def parse_server_log(log_path: str) -> dict:
    """Parse SGLang server log for accept metrics."""
    accept_lens = []
    throughputs = []

    try:
        with open(log_path) as f:
            for line in f:
                # Decode batch, ... accept len: 1.00, accept rate: 0.02, ... gen throughput (token/s): 9.91
                m_accept = re.search(r"accept len:\s*([\d.]+)", line)
                m_throughput = re.search(r"gen throughput \(token/s\):\s*([\d.]+)", line)
                if m_accept:
                    accept_lens.append(float(m_accept.group(1)))
                if m_throughput:
                    throughputs.append(float(m_throughput.group(1)))
    except FileNotFoundError:
        pass

    return {
        "accept_lens": accept_lens,
        "throughputs": throughputs,
    }


def run_workload(
    url: str,
    model: str,
    prompts: list[dict],
    max_tokens: int = 256,
    n_warmup: int = 2,
) -> dict:
    """Run workload and collect per-request metrics."""
    client = OpenAI(base_url=f"{url}/v1", api_key="dummy")

    # Warmup
    for i in range(min(n_warmup, len(prompts))):
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompts[i]["prompt"]}],
                max_tokens=max_tokens, temperature=0.0,
            )
        except Exception:
            pass

    # Measure
    request_metrics = []
    total_tokens = 0
    t_start = time.perf_counter()

    for p in prompts:
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": p["prompt"]}],
                max_tokens=max_tokens, temperature=0.0,
            )
            elapsed = time.perf_counter() - t0
            n_gen = resp.usage.completion_tokens
            n_prompt = resp.usage.prompt_tokens
            total_tokens += n_gen
            request_metrics.append({
                "question_id": p["question_id"],
                "category": p["category"],
                "prompt_tokens": n_prompt,
                "completion_tokens": n_gen,
                "latency_s": elapsed,
                "tpot_ms": (elapsed / n_gen * 1000) if n_gen > 0 else 0,
            })
        except Exception as e:
            request_metrics.append({
                "question_id": p["question_id"],
                "error": str(e),
            })

    total_time = time.perf_counter() - t_start

    return {
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "overall_tps": total_tokens / total_time if total_time > 0 else 0,
        "requests": request_metrics,
    }


def parse_configs(config_str: str) -> list[dict]:
    """Parse config string into list of (topk, steps, budget) dicts.

    Format: "chain:1,4,16,64;tree:4-3:4,16,64;tree:8-3:8,32,64"
    """
    configs = []
    for part in config_str.split(";"):
        part = part.strip()
        if part.startswith("chain:"):
            budgets = [int(b) for b in part[6:].split(",")]
            for b in budgets:
                configs.append({"topk": 1, "steps": b, "budget": b,
                                "mode": "chain"})
        elif part.startswith("tree:"):
            rest = part[5:]
            params, budget_str = rest.split(":")
            topk, steps = map(int, params.split("-"))
            budgets = [int(b) for b in budget_str.split(",")]
            for b in budgets:
                configs.append({"topk": topk, "steps": steps, "budget": b,
                                "mode": "tree"})
    # Deduplicate by (topk, steps, budget)
    seen = set()
    deduped = []
    for c in configs:
        key = (c["topk"], c["steps"], c["budget"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


# Default configs (~48 min, 16 unique configs × 3 min each)
# Systematic sweep of 3 variables:
#   1. Chain baseline: topk=1, steps=budget
#   2. Topk sweep:  steps=3 fixed, budget=32 fixed, topk=2,4,8
#   3. Steps sweep: topk=8 fixed, budget=32 fixed, steps=2,3,4,5
#   4. Budget sweep: topk=8 fixed, steps=3 fixed, budget=4,8,16,32,64
DEFAULT_CONFIGS = (
    # Chain baseline
    "chain:1,2,4,8,16,32;"
    # Topk sweep (steps=3, budget=32)
    "tree:2-3:32;tree:4-3:32;tree:8-3:32;"
    # Steps sweep (topk=8, budget=32) — 8-3:32 is deduped
    "tree:8-2:32;tree:8-4:32;tree:8-5:32;"
    # Budget sweep (topk=8, steps=3) — 8-3:32 is deduped
    "tree:8-3:4,8,16,64"
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dataset", required=True,
                        help="SpecBench dataset.jsonl path")
    parser.add_argument("--n-requests", type=int, default=8,
                        help="Number of requests per config (default: 8 = 1 per category)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens per request")
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS,
                        help="Config string (see --help for format)")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-args", nargs="*", default=[])
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    configs = parse_configs(args.configs)
    prompts = load_specbench(args.dataset, args.n_requests)
    server_log = Path("/tmp/sglang_bench_server.log")

    print(f"SpecBench prompts: {len(prompts)}", file=sys.stderr)
    print(f"Configs: {len(configs)}", file=sys.stderr)
    print(f"Estimated time: ~{len(configs) * 3} min\n", file=sys.stderr)

    # --- Vanilla baseline ---
    print("=" * 80, file=sys.stderr)
    print("Measuring vanilla baseline (no speculation)...", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    env = os.environ.copy()
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--tp-size", str(args.tp_size),
        "--mem-fraction-static", "0.8",
        "--disable-cuda-graph",
        "--host", "0.0.0.0", "--port", str(args.port),
    ] + args.extra_args

    proc = subprocess.Popen(cmd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid)
    if not wait_for_server(url):
        kill_server(proc)
        print("ERROR: Vanilla server failed", file=sys.stderr)
        sys.exit(1)

    vanilla_result = run_workload(url, args.model, prompts,
                                  args.max_tokens, args.n_warmup)
    vanilla_tpot = statistics.median(
        [r["tpot_ms"] for r in vanilla_result["requests"] if "tpot_ms" in r])
    print(f"  Vanilla TPOT: {vanilla_tpot:.2f} ms/tok, "
          f"TPS: {vanilla_result['overall_tps']:.1f}\n", file=sys.stderr)
    kill_server(proc)
    time.sleep(5)

    # --- Sweep configs ---
    all_results = []

    for ci, cfg in enumerate(configs):
        topk, steps, budget = cfg["topk"], cfg["steps"], cfg["budget"]
        mode = cfg["mode"]

        print("=" * 80, file=sys.stderr)
        print(f"[{ci+1}/{len(configs)}] {mode}: topk={topk}, steps={steps}, "
              f"budget={budget}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        server_log.unlink(missing_ok=True)

        env = os.environ.copy()
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", args.model,
            "--tp-size", str(args.tp_size),
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", args.draft_model,
            "--speculative-num-steps", str(steps),
            "--speculative-eagle-topk", str(topk),
            "--speculative-num-draft-tokens", str(budget),
            "--mem-fraction-static", "0.8",
            "--disable-cuda-graph",
            "--host", "0.0.0.0", "--port", str(args.port),
        ] + args.extra_args

        log_fh = open(server_log, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh)

        if not wait_for_server(url):
            kill_server(proc)
            log_fh.close()
            print(f"  ERROR: Server failed\n", file=sys.stderr)
            all_results.append({**cfg, "error": "server_failed"})
            time.sleep(3)
            continue

        # Start monitors
        gpu_stop = threading.Event()
        cpu_stop = threading.Event()
        gpu_samples = []
        cpu_samples = []

        gpu_thread = threading.Thread(
            target=lambda: gpu_samples.extend(monitor_gpu(1.0, gpu_stop)))
        cpu_thread = threading.Thread(
            target=lambda: cpu_samples.extend(monitor_cpu(1.0, cpu_stop)))
        gpu_thread.start()
        cpu_thread.start()

        # Run workload
        workload = run_workload(url, args.model, prompts,
                                args.max_tokens, args.n_warmup)

        # Stop monitors
        gpu_stop.set()
        cpu_stop.set()
        gpu_thread.join()
        cpu_thread.join()

        # Kill server and parse logs
        kill_server(proc)
        log_fh.close()
        time.sleep(3)

        server_metrics = parse_server_log(str(server_log))

        # Compute aggregated metrics
        tpots = [r["tpot_ms"] for r in workload["requests"] if "tpot_ms" in r]
        med_tpot = statistics.median(tpots) if tpots else 0

        accept_lens = server_metrics["accept_lens"]
        # Skip warmup entries
        skip = args.n_warmup * args.max_tokens
        accept_lens = accept_lens[skip:] if len(accept_lens) > skip else accept_lens

        avg_accept = statistics.mean(accept_lens) if accept_lens else 0
        mat = avg_accept  # MAT = mean accepted tokens per step

        throughputs = server_metrics["throughputs"]
        throughputs = throughputs[skip:] if len(throughputs) > skip else throughputs
        avg_throughput = statistics.mean(throughputs) if throughputs else 0

        # GPU metrics (filter to GPU 0 for TP=1)
        gpu0 = [s for s in gpu_samples if s.get("gpu_idx", 0) == 0]
        avg_gpu_util = statistics.mean([s["gpu_util"] for s in gpu0]) if gpu0 else 0
        avg_mem_used = statistics.mean([s["mem_used_mb"] for s in gpu0]) if gpu0 else 0
        avg_power = statistics.mean([s["power_w"] for s in gpu0]) if gpu0 else 0

        avg_cpu_util = statistics.mean(cpu_samples) if cpu_samples else 0

        speedup = vanilla_tpot / med_tpot if med_tpot > 0 else 0

        entry = {
            **cfg,
            "tpot_ms": round(med_tpot, 2),
            "speedup": round(speedup, 2),
            "mat": round(mat, 2),
            "accept_rate": round(avg_accept / budget if budget > 0 else 0, 4),
            "overall_tps": round(workload["overall_tps"], 1),
            "server_throughput": round(avg_throughput, 1),
            "total_tokens": workload["total_tokens"],
            "total_time_s": round(workload["total_time_s"], 1),
            "gpu_util_pct": round(avg_gpu_util, 1),
            "gpu_mem_mb": round(avg_mem_used, 0),
            "gpu_power_w": round(avg_power, 1),
            "cpu_util_pct": round(avg_cpu_util, 1),
            "n_requests": len(workload["requests"]),
            "n_accept_samples": len(accept_lens),
        }
        all_results.append(entry)

        print(f"  TPOT={med_tpot:.1f}ms, speedup={speedup:.2f}x, "
              f"MAT={mat:.2f}, accept_rate={entry['accept_rate']:.3f}, "
              f"TPS={workload['overall_tps']:.1f}, "
              f"GPU={avg_gpu_util:.0f}%\n", file=sys.stderr)

    # --- Output ---
    output = {
        "model": args.model,
        "draft_model": args.draft_model,
        "dataset": args.dataset,
        "n_requests": args.n_requests,
        "max_tokens": args.max_tokens,
        "vanilla_tpot_ms": round(vanilla_tpot, 2),
        "vanilla_tps": round(vanilla_result["overall_tps"], 1),
        "results": all_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # --- Summary table ---
    print("\n" + "=" * 120, file=sys.stderr)
    print("EAGLE3 BENCHMARK RESULTS", file=sys.stderr)
    print("=" * 120, file=sys.stderr)
    print(f"Model: {args.model} | Vanilla TPOT: {vanilla_tpot:.2f} ms/tok | "
          f"Vanilla TPS: {vanilla_result['overall_tps']:.1f}", file=sys.stderr)
    print(file=sys.stderr)

    hdr = (f"{'mode':>5} | {'topk':>4} | {'steps':>5} | {'budget':>6} | "
           f"{'TPOT':>8} | {'speedup':>7} | {'MAT':>5} | {'acc_rate':>8} | "
           f"{'TPS':>6} | {'GPU%':>5} | {'MEM':>7} | {'CPU%':>5}")
    print(hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)

    for r in all_results:
        if "error" in r:
            print(f"{r['mode']:>5} | {r['topk']:>4} | {r['steps']:>5} | "
                  f"{r['budget']:>6} | {'ERROR':>8}", file=sys.stderr)
        else:
            print(f"{r['mode']:>5} | {r['topk']:>4} | {r['steps']:>5} | "
                  f"{r['budget']:>6} | "
                  f"{r['tpot_ms']:>7.1f}ms | {r['speedup']:>6.2f}x | "
                  f"{r['mat']:>5.2f} | {r['accept_rate']:>7.3f} | "
                  f"{r['overall_tps']:>5.1f} | {r['gpu_util_pct']:>4.0f}% | "
                  f"{r['gpu_mem_mb']:>5.0f}MB | {r['cpu_util_pct']:>4.1f}%",
                  file=sys.stderr)

    print("=" * len(hdr), file=sys.stderr)
    print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
