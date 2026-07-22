#!/usr/bin/env python3
"""Per-position 3-factor decomposition of MAT loss (exact):
   L = Σ_k π(k)·m(k)·sev(k)
   π(k)   = P(routed to k)             (calibration's routing lever)
   m(k)   = P(miss | routed to k)      (routing-error rate at k)
   sev(k) = E[gap | routed to k, miss] (severity when wrong at k)
and C(k)=π·m·sev = per-position loss contribution (sums to E[Loss]).
Pooled + per-workload (motivation/ and motivation/{ds}/). One figure per arm:
4 stacked panels (π, m, sev, C).  Uses ckv5_{ds}.json (needs miss counts).
"""
from pathlib import Path
import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/workspace/simulation/Dr.Lee Solution/readable_outputs/figures/motivation")
WL = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench Verified",
      "spider": "Spider2-DBT", "tau2": "tau2-bench", "POOLED": "All workloads (pooled)"}
COL = {"raw": "#F58518", "cal": "#4C78A8"}


def draw(od, tag, arm, cnt, gl, miss, N):
    K = len(cnt); ks = np.arange(K)
    pi = cnt / N
    m = np.where(cnt > 0, miss / np.where(cnt == 0, 1, cnt), 0.0)
    sev = np.where(miss > 0, gl / np.where(miss == 0, 1, miss), 0.0)
    C = gl / N                                   # = pi*m*sev exactly
    lab = {"raw": "RAW", "cal": "CALIBRATED"}[arm]
    c = COL[arm]
    fig, ax = plt.subplots(4, 1, figsize=(9, 10.5), sharex=True)
    ax[0].bar(ks, pi, 0.7, color=c, edgecolor="k", lw=0.3)
    ax[0].set_ylabel("π(k)\nrouting prob", fontsize=10)
    ax[1].bar(ks, m, 0.7, color=c, edgecolor="k", lw=0.3, alpha=0.85)
    ax[1].set_ylabel("m(k)\nP(miss | k)", fontsize=10); ax[1].set_ylim(0, 1.02)
    ax[2].bar(ks, sev, 0.7, color=c, edgecolor="k", lw=0.3, alpha=0.7)
    ax[2].set_ylabel("sev(k)\nE[gap | k, miss]", fontsize=10)
    ax[3].bar(ks, C, 0.7, color=c, edgecolor="k", lw=0.4)
    ax[3].set_ylabel("C(k) = π·m·sev\nloss contribution", fontsize=10)
    ax[3].set_xlabel("handoff position k  (k=0 = pure suffix)", fontsize=11)
    ax[3].set_xticks(ks)
    for a in ax:
        a.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{WL[tag]} — {lab}: per-position 3-factor loss decomposition\n"
                 f"L = Σ_k π(k)·m(k)·sev(k) = {C.sum():.3f} tokens/step  (exact)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(od / f"loss_3factor_{arm}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(tag, d, od):
    od.mkdir(parents=True, exist_ok=True)
    N = d["n"]
    for arm in ("raw", "cal"):
        draw(od, tag, arm, np.array(d[f"cnt_{arm}"]), np.array(d[f"gl_{arm}"]),
             np.array(d[f"miss_{arm}"]), N)
    print(f"[{tag}] loss_3factor_{{raw,cal}} saved")


# per-workload + pooled accumulate
pooled = None
for f in sorted(glob.glob("/workspace/tmp/ckv5_*.json")):
    ds = Path(f).stem.replace("ckv5_", "")
    d = json.load(open(f))
    run(ds, d, BASE / ds)
    if pooled is None:
        pooled = {k: (np.array(v) if isinstance(v, list) else v) for k, v in d.items()}
    else:
        for k, v in d.items():
            pooled[k] = pooled[k] + (np.array(v) if isinstance(v, list) else v)
run("POOLED", {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pooled.items()}, BASE)
print("ALL 3-factor figures done")
