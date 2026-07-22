#!/usr/bin/env python3
r"""Golden-section (binary-search-style) coordinate descent for the best genbeta
(a,b) tail smoothing. Reparametrized as (prior mean m = a/b, strength b); a = m*b.
Alternates a 1D golden search on strength b (at fixed m) and on mean m (at fixed b).
Each evaluation runs all 5 workloads (parallel) and takes the MEAN test-half MAT.
Appends every eval to genbeta_search_progress.jsonl so progress can be reported
incrementally. Waits for the coarse grid (GENBETA_DONE) before starting.

  docker exec -d sglang-bench bash -lc 'cd /workspace/simulation/Dr.Lee\ Solution && \
    PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace python3 scripts/analysis/golden_search_ab.py'
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import time
from pathlib import Path

RLOG = Path("readable_outputs/figures/replay_logs")
ST = Path("results/headweight_tailsmooth")
DSS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
REC = {
    "specbench": "results/perpos_specbench_alleval/specbench_4way.jsonl",
    "bfcl": "results/perpos_bfcl_alleval/bfcl_4way.jsonl",
    "swebench": "results/perpos_swebench_alleval/swebench_4way.jsonl",
    "spider": "results/perpos_spider_alleval/spider_4way.jsonl",
    "tau2": "results/perpos_tau2_alleval/tau2_4way.jsonl",
}
PROG = ST / "genbeta_search_progress.jsonl"
K_RE = re.compile(r"^\s*calib:\s*K=([0-9.]+)")
MAX_EVALS = 14
cache = {}
n_evals = 0


def read_k(p):
    try:
        for ln in open(p):
            m = K_RE.match(ln)
            if m:
                return float(m.group(1))
    except FileNotFoundError:
        pass
    return None


def eval_ab(a, b):
    global n_evals
    a = max(1e-3, float(a)); b = max(1e-3, float(b))
    key = (round(a, 4), round(b, 4))
    if key in cache:
        return cache[key]
    if n_evals >= MAX_EVALS:
        return -1.0
    n_evals += 1
    procs = []
    for ds in DSS:
        out = RLOG / f"mat_{ds}_hwts_gbsearch_a{a:.4f}_b{b:.4f}_split.replay.txt"
        p = subprocess.Popen(
            ["python3", "scripts/replay_extension.py", "--record", REC[ds],
             "--max-rounds", "4096", "--props", "calib", "--three-way",
             "--group-mode", "convlabel", "--head-cal", "raw", "--tail-cal", "genbeta",
             "--gb-a", f"{a:.4f}", "--gb-b", f"{b:.4f}"],
            stdout=open(out, "w"), stderr=subprocess.DEVNULL)
        procs.append((p, out))
    for p, _ in procs:
        p.wait()
    ks = [read_k(o) for _, o in procs]
    mat = sum(ks) / len(ks) if all(k is not None for k in ks) else None
    cache[key] = mat if mat is not None else -1.0
    rec = {"eval": n_evals, "a": a, "b": b, "prior_mean": a / b, "strength": b,
           "mat": mat, "per": dict(zip(DSS, ks))}
    with open(PROG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return cache[key]


def golden_max(f, lo, hi, tol, maxit):
    """Maximize f on [lo,hi] by golden section. Returns (x*, f*)."""
    gr = (5 ** 0.5 - 1) / 2
    x1 = hi - gr * (hi - lo); x2 = lo + gr * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(maxit):
        if abs(hi - lo) < tol or n_evals >= MAX_EVALS:
            break
        if f1 < f2:
            lo = x1; x1 = x2; f1 = f2
            x2 = lo + gr * (hi - lo); f2 = f(x2)
        else:
            hi = x2; x2 = x1; f2 = f1
            x1 = hi - gr * (hi - lo); f1 = f(x1)
    return (x1, f1) if f1 >= f2 else (x2, f2)


def main():
    while not (ST / "GENBETA_DONE").exists():
        time.sleep(30)
    PROG.write_text("")   # fresh
    m, b = 0.25, 2.0      # seed from coarse-grid best (a=0.5,b=2)
    # Round 1: strength b at fixed mean m, then mean m at fixed b
    b, _ = golden_max(lambda bb: eval_ab(m * bb, bb), 0.5, 6.0, 0.4, 5)
    m, _ = golden_max(lambda mm: eval_ab(mm * b, b), 0.06, 0.55, 0.04, 5)
    # Round 2: refine b once more around the new mean
    if n_evals < MAX_EVALS:
        b, _ = golden_max(lambda bb: eval_ab(m * bb, bb), max(0.3, b - 1.5), b + 2.0, 0.3, 4)
    best = max((v for v in cache.values() if v is not None), default=-1)
    bestkey = max(cache, key=lambda k: cache[k] if cache[k] is not None else -1)
    json.dump({"best_a": bestkey[0], "best_b": bestkey[1], "best_mat": best,
               "n_evals": n_evals}, open(ST / "genbeta_search_best.json", "w"), indent=1)
    (ST / "GBSEARCH_DONE").write_text("")


if __name__ == "__main__":
    main()
