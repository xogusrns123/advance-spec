#!/usr/bin/env python3
"""Per-position factor graphs of the 3-factor loss decomposition L=Σ π·m·sev:
  factor_miss.png : m(k)=P(miss|k)      raw vs cal
  factor_sev.png  : sev(k)=E[gap|k,miss] raw vs cal
  factor_all_{raw,cal}.png : π, m, sev overlaid on one axes (product = C(k))
pooled (motivation/) + per-workload (motivation/{ds}/). ckv5_{ds}.json.
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
C_RAW, C_CAL = "#F58518", "#4C78A8"


def factors(d, arm):
    N = d["n"]; cnt = np.array(d[f"cnt_{arm}"], float)
    gl = np.array(d[f"gl_{arm}"], float); mc = np.array(d[f"miss_{arm}"], float)
    pi = cnt / N
    m = np.where(cnt > 0, mc / np.where(cnt == 0, 1, cnt), 0.0)
    sev = np.where(mc > 0, gl / np.where(mc == 0, 1, mc), 0.0)
    return pi, m, sev, gl / N


def compare(od, tag, ks, yr, yc, ylab, fname, title, ymax=None):
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.bar(ks - w / 2, yr, w, color=C_RAW, edgecolor="k", lw=0.4, label="raw")
    ax.bar(ks + w / 2, yc, w, color=C_CAL, edgecolor="k", lw=0.4, label="calib")
    ax.set_xlabel("handoff position k  (k=0 = pure suffix)", fontsize=11)
    ax.set_ylabel(ylab, fontsize=11); ax.set_xticks(ks)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.set_title(f"{WL[tag]} — {title}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / fname, dpi=150); plt.close(fig)


def allthree(od, tag, ks, pi, m, sev, C, arm):
    lab = {"raw": "RAW", "cal": "CALIBRATED"}[arm]
    col = C_RAW if arm == "raw" else C_CAL
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.bar(ks, pi, 0.6, color=col, alpha=0.85, edgecolor="k", lw=0.3, zorder=2,
           label="π(k)  routing prob")
    ax.plot(ks, m, color="#2ca02c", lw=2.2, marker="o", ms=4, zorder=4,
            label="m(k)  P(miss | k)")
    ax.set_xlabel("handoff position k  (k=0 = pure suffix)", fontsize=11)
    ax.set_ylabel("π(k),  m(k)   (probability)", fontsize=11)
    ax.set_ylim(0, 1.02); ax.set_xticks(ks)
    axr = ax.twinx()
    axr.plot(ks, sev, color="#9467bd", lw=2.2, marker="s", ms=4, ls="--", zorder=4,
             label="sev(k)  E[gap | k, miss]")
    axr.set_ylabel("sev(k)   (tokens)", fontsize=11, color="#6a3d9a")
    axr.set_ylim(0, max(sev.max() * 1.15, 0.1))
    ax.set_title(f"{WL[tag]} — {lab}: 3 factors in one view   "
                 f"C(k)=π·m·sev,  Σ C(k)={C.sum():.2f}", fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(od / f"factor_all_{arm}.png", dpi=150); plt.close(fig)


def run(tag, d, od):
    od.mkdir(parents=True, exist_ok=True)
    K = len(d["cnt_raw"]); ks = np.arange(K)
    pr, mr, sr, Cr = factors(d, "raw"); pc, mc, sc, Cc = factors(d, "cal")
    compare(od, tag, ks, mr, mc, "m(k) = P(miss | routed to k)", "factor_miss.png",
            "routing-error rate per position  m(k)", ymax=1.02)
    compare(od, tag, ks, sr, sc, "sev(k) = E[gap | routed to k, miss]  (tokens)",
            "factor_sev.png", "severity per position  sev(k)")
    allthree(od, tag, ks, pr, mr, sr, Cr, "raw")
    allthree(od, tag, ks, pc, mc, sc, Cc, "cal")
    print(f"[{tag}] factor_miss / factor_sev / factor_all_{{raw,cal}} saved")


pooled = None
for f in sorted(glob.glob("/workspace/tmp/ckv5_*.json")):
    ds = Path(f).stem.replace("ckv5_", ""); d = json.load(open(f))
    run(ds, d, BASE / ds)
    if pooled is None:
        pooled = {k: (np.array(v) if isinstance(v, list) else v) for k, v in d.items()}
    else:
        for k, v in d.items():
            pooled[k] = pooled[k] + (np.array(v) if isinstance(v, list) else v)
run("POOLED", {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pooled.items()}, BASE)
print("ALL factor figures done")
