#!/usr/bin/env python3
"""Pop-patch reslice sweep — runs the simulator across (workload, s, k, cond)
inside docker `sglang-bench`, with a memory watchdog and parallel sharding.

Output: one JSON per (workload, sk, cond) into pop_patch_3wl_resliced/.
Skips jobs whose output already exists.
"""
import argparse
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

HOST_ROOT = Path("/home/muchwater/advance-spec")
DOCKER_ROOT = Path("/workspace")
OUT_REL = Path("simulation/results/explorations/pop_patch_3wl_resliced")
DOCKER_NAME = "sglang-bench"

DATASET_MAP = {
    "bfcl_v4": "data/bfcl_agent/dataset_stratified_interleaved.jsonl",
    "specbench": "data/specbench/dataset.jsonl",
    "swebench_verified": "data/swebench_verified/dataset_interleaved.jsonl",
}
RESLICES_FULL = [(s, k) for s in (2, 4, 6, 8) for k in (4, 8, 16)]
CONDITIONS = ("before", "after")


def parse_reslices(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        if tok == "full":
            out.extend(RESLICES_FULL); continue
        import re
        m = re.match(r"^s(\d+)k(\d+)$", tok)
        if not m: raise ValueError(f"bad reslice: {tok!r}")
        out.append((int(m.group(1)), int(m.group(2))))
    # dedupe preserving order
    seen, dedup = set(), []
    for sk in out:
        if sk not in seen:
            seen.add(sk); dedup.append(sk)
    return dedup
DEFAULT_PARALLEL = {
    "bfcl_v4": 6,            # 6.7GB capture
    "specbench": 4,          # 17GB capture
    "swebench_verified": 2,  # 25GB capture
}


def read_meminfo_mb(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) // 1024
    return 0


class MemoryWatchdog:
    """Kills all tracked processes if free memory ratio drops below threshold."""

    def __init__(self, threshold_pct, check_interval_s=5):
        self.threshold_pct = threshold_pct
        self.check_interval_s = check_interval_s
        self.killed = threading.Event()
        self.procs = []  # (popen, label)
        self.lock = threading.Lock()
        self.thread = None
        self.total_mb = read_meminfo_mb("MemTotal:")

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def add(self, proc, label):
        with self.lock:
            self.procs.append((proc, label))

    def remove(self, proc):
        with self.lock:
            self.procs = [(p, l) for p, l in self.procs if p is not proc]

    def _kill_all(self):
        # The simulator runs as root inside the docker container — host user
        # can't kill it directly. Use `docker exec ... pkill` to nuke all sim
        # processes inside the container in one shot.
        try:
            subprocess.run(
                ["docker", "exec", DOCKER_NAME,
                 "pkill", "-TERM", "-f", "run_tree_oracle_sim"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=15)
        except Exception as e:
            print(f"   docker pkill failed: {e}", flush=True)
        # Also terminate the host-side `docker exec` wrappers so their
        # `proc.poll()` returns and the orchestrator loop exits.
        with self.lock:
            for p, label in list(self.procs):
                if p.poll() is None:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    print(f"   terminated wrapper {label}", flush=True)

    def _loop(self):
        while not self.killed.is_set():
            avail = read_meminfo_mb("MemAvailable:")
            pct = avail / self.total_mb * 100 if self.total_mb else 100
            if pct < self.threshold_pct:
                print(f"\n[!! WATCHDOG] free memory {pct:.1f}% < {self.threshold_pct}% — killing sims",
                      flush=True)
                self._kill_all()
                self.killed.set()
                return
            time.sleep(self.check_interval_s)

    def stop(self):
        self.killed.set()


def build_inner_cmd(workload, s, k):
    cap = DOCKER_ROOT / "simulation" / "results" / "qwen3_14b" / f"{workload}_steps8_topk16_capture"
    cmd = [
        "python3", "-m", "simulation.evaluation.run_tree_oracle_sim",
        "--agent-results", str(cap / "agent_results_eagle3.json"),
        "--dataset", str(DOCKER_ROOT / DATASET_MAP[workload]),
        "--model", "Qwen/Qwen3-14B",
        "--latency-config", str(cap / "latency_config.json"),
        "--steps", str(s), "--topk", str(k),
        "--reslice-steps", str(s), "--reslice-topk", str(k),
        "--capture-steps", "8", "--capture-topk", "16",
        "--budgets", "1,2,4,8,16,32,64,128",
        # Pop-patch ablation only needs `extension` and `extension_oracle`
        # (the `temporary_extension` callsite). Skipping all single/hybrid/
        # extension variants cuts ~98% of method dispatches and makes each
        # sim run dramatically faster.
        "--methods", "extension:,extension_oracle:",
    ]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workloads", default="bfcl_v4,specbench,swebench_verified")
    p.add_argument("--reslices", default="full",
                   help="comma-separated, e.g. 'full' or 's6k16,s8k16'")
    p.add_argument("--watchdog-pct", type=int, default=20,
                   help="kill all sims if MemAvailable / MemTotal < this %")
    p.add_argument("--max-parallel-bfcl_v4", type=int, default=DEFAULT_PARALLEL["bfcl_v4"])
    p.add_argument("--max-parallel-specbench", type=int, default=DEFAULT_PARALLEL["specbench"])
    p.add_argument("--max-parallel-swebench_verified", type=int,
                   default=DEFAULT_PARALLEL["swebench_verified"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    parallel_map = {
        "bfcl_v4": args.max_parallel_bfcl_v4,
        "specbench": args.max_parallel_specbench,
        "swebench_verified": args.max_parallel_swebench_verified,
    }

    workloads = args.workloads.split(",")
    reslices = parse_reslices(args.reslices)
    out_host = HOST_ROOT / OUT_REL
    out_host.mkdir(parents=True, exist_ok=True)
    log_dir = out_host / "_logs"
    log_dir.mkdir(exist_ok=True)

    print(f"Reslices: {reslices}  ({len(reslices)} configs × {len(CONDITIONS)} conditions = {len(reslices)*len(CONDITIONS)} jobs/workload)",
          flush=True)

    # Build job lists.
    pending_per_wl = {}
    for wl in workloads:
        pending = []
        for s, k in reslices:
            for cond in CONDITIONS:
                out_name = f"sim_{wl}_s{s}k{k}_{cond}.json"
                out_path = out_host / out_name
                if out_path.exists() and out_path.stat().st_size > 1000:
                    continue
                pending.append((s, k, cond))
        pending_per_wl[wl] = pending
        total = len(reslices) * len(CONDITIONS)
        print(f"  {wl}: {len(pending)}/{total} pending  (parallel={parallel_map[wl]})",
              flush=True)

    if args.dry_run:
        for wl, jobs in pending_per_wl.items():
            for s, k, cond in jobs:
                print(f"DRY: docker exec {DOCKER_NAME} ... {wl} s{s}k{k} {cond}", flush=True)
        return

    wd = MemoryWatchdog(threshold_pct=args.watchdog_pct, check_interval_s=5)
    wd.start()

    def launch(wl, s, k, cond):
        out_docker = DOCKER_ROOT / OUT_REL / f"sim_{wl}_s{s}k{k}_{cond}.json"
        log_path = log_dir / f"sim_{wl}_s{s}k{k}_{cond}.log"
        inner_cmd = build_inner_cmd(wl, s, k) + ["--output", str(out_docker)]
        env_prefix = "BENCH_NO_TEMP_EXT=1 " if cond == "before" else ""
        # SIM_PARALLEL=4 lets the simulator fork 4 workers per (s,k,cond)
        # job and process the 3 extension + 1 extension_oracle dispatches in
        # parallel per budget. Combined with our outer parallelism this
        # gives a wider sweep — careful with memory but well within budget
        # given the watchdog at 20%.
        env_prefix += "SIM_PARALLEL=4 PYTHONUNBUFFERED=1 "
        wrapped = f"cd {DOCKER_ROOT} && {env_prefix}" + " ".join(
            f"'{a}'" if " " in a else a for a in inner_cmd
        )
        cmd = ["docker", "exec", DOCKER_NAME, "bash", "-c", wrapped]
        log_f = open(log_path, "w")
        label = f"{wl}_s{s}k{k}_{cond}"
        print(f"  → launch {label}", flush=True)
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                preexec_fn=os.setsid)
        wd.add(proc, label)
        return (proc, label, time.time(), log_f)

    overall_start = time.time()
    for wl in workloads:
        jobs = pending_per_wl[wl]
        if not jobs:
            continue
        max_par = parallel_map[wl]
        print(f"\n=== {wl} — {len(jobs)} jobs, parallel={max_par} ===", flush=True)
        queue = list(jobs)
        running = []  # (proc, label, t0, log_f)

        while (queue or running) and not wd.killed.is_set():
            while queue and len(running) < max_par:
                s, k, cond = queue.pop(0)
                running.append(launch(wl, s, k, cond))
            still = []
            for proc, label, t0, log_f in running:
                rc = proc.poll()
                if rc is None:
                    still.append((proc, label, t0, log_f))
                else:
                    elapsed = time.time() - t0
                    log_f.close()
                    wd.remove(proc)
                    avail = read_meminfo_mb("MemAvailable:")
                    pct = avail / wd.total_mb * 100 if wd.total_mb else 0
                    status = "OK" if rc == 0 else f"FAIL rc={rc}"
                    print(f"  {status:8s} {label}  {elapsed:.0f}s  free={pct:.1f}%",
                          flush=True)
            running = still
            time.sleep(2)

        if wd.killed.is_set():
            print("\nWATCHDOG TRIGGERED — aborting sweep", flush=True)
            # wait for any still-running children to be reaped
            for proc, label, t0, log_f in running:
                try: proc.wait(timeout=10)
                except Exception: pass
                try: log_f.close()
                except Exception: pass
            break

    wd.stop()
    overall = time.time() - overall_start
    print(f"\n=== SWEEP DONE in {overall/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
