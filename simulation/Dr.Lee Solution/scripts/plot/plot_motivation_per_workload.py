#!/usr/bin/env python3
"""Regenerate ALL motivation figures PER WORKLOAD into motivation/{ds}/ subfolders.
Same formats as the pooled versions, driven by the per-workload JSONs
perpos2_{ds}.json (per-position gains) and ckv5_{ds}.json (routing decomposition).
English labels only.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/workspace/simulation/Dr.Lee Solution/readable_outputs/figures/motivation")
DS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
WL = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench Verified",
      "spider": "Spider2-DBT", "tau2": "tau2-bench"}
C_HEAD, C_TAIL, L_HEAD, L_TAIL = "#9ecae1", "#fdd0a2", "#08519c", "#d94801"
C_RAW, C_CAL, C_GAP = "#F58518", "#4C78A8", "#888888"


def perpos_draw(od, ds, rh, rt, gh, gt, title, fname, loc):
    ks = list(range(len(rh)))
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.bar(ks, rh, color=C_HEAD, edgecolor="k", linewidth=0.4, zorder=2,
           label="realized head accept  E[min(k, a_head)]")
    ax.bar(ks, rt, bottom=rh, color=C_TAIL, edgecolor="k", linewidth=0.4, zorder=2,
           label="realized tail accept  (grafted at k)")
    ax.plot(ks, gh, color=L_HEAD, lw=2.4, marker="o", ms=4, zorder=4,
            label="expected head gain  G_k")
    ax.plot(ks, gh + gt, color=L_TAIL, lw=2.4, marker="s", ms=4, zorder=4,
            label="expected total gain  G_k + S_k·T_k  (gap = expected tail)")
    ax.set_xlabel("handoff position k  (head length; k=0 = pure suffix)", fontsize=11)
    ax.set_ylabel("accepted tokens", fontsize=11)
    ax.set_xticks(ks)
    ax.set_ylim(0, max(3.6, float((rh + rt).max()) * 1.15, float((gh + gt).max()) * 1.05))
    ax.set_title(f"{WL[ds]} — {title}", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc=loc, frameon=True)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout(); fig.savefig(od / fname, dpi=150); plt.close(fig)


def routing_arm(od, ds, ks, pi, gcond, color, name, loss, ymax, gcmax):
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.bar(ks, pi, 0.7, color=color, edgecolor="k", lw=0.4, zorder=2,
           label=f"π_{name}(k) (routing)")
    ax.set_xlabel("handoff position k  (k=0 = pure suffix)", fontsize=11)
    ax.set_ylabel(f"routing probability  π_{name}(k)", fontsize=11)
    ax.set_ylim(0, ymax * 1.1)
    axg = ax.twinx()
    axg.plot(ks, gcond, color=C_GAP, lw=2.4, marker="o", ms=4, ls="--", zorder=3,
             label="g_cond(k) = E[gap | routed to k]")
    axg.set_ylabel("conditional gap  g_cond(k)  (tokens)", fontsize=11, color="#555")
    axg.set_ylim(0, gcmax * 1.1)
    lab = {"raw": "RAW", "cal": "CALIBRATED"}[name]
    ax.set_title(f"{WL[ds]} — {lab} routing π vs conditional gap g_cond\n"
                 f"Σ π(k)·g_cond(k) = loss = {loss:.2f} tokens/step  (exact)",
                 fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axg.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(od / f"routing_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def loss_contrib_arm(od, ds, ks, lbk, gcond, color, name, ymax, gcmax):
    lab = {"raw": "RAW", "cal": "CALIBRATED"}[name]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.bar(ks, lbk, 0.7, color=color, edgecolor="k", lw=0.4, zorder=2,
           label=f"loss contribution C(k) = π·g_cond ({name})")
    ax.set_xlabel("chosen handoff position k_hat  (k=0 = pure suffix)", fontsize=11)
    ax.set_ylabel("loss contribution C(k)  (tokens/step)", fontsize=11)
    ax.set_ylim(0, ymax * 1.12)
    axg = ax.twinx()
    axg.plot(ks, gcond, color=C_GAP, lw=2.4, marker="o", ms=4, ls="--", zorder=3,
             label="g_cond(k) = E[gap | routed to k]  (conditional)")
    axg.set_ylabel("conditional gap  g_cond(k)  (tokens)", fontsize=11, color="#555")
    axg.set_ylim(0, gcmax * 1.1)
    ax.set_title(f"{WL[ds]} — {lab} loss contribution by position\n"
                 f"C(k) = π(k)·g_cond(k)  (exact);  Σ C(k) = {lbk.sum():.2f} tokens/step",
                 fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axg.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(od / f"loss_contrib_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def routing_decomp(od, ds, ks, pi_raw, pi_cal, gapbar, lbk_raw, lbk_cal, N):
    lr, lc = lbk_raw.sum(), lbk_cal.sum()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2))
    w = 0.4
    ax1.bar(ks - w / 2, pi_raw, w, color=C_RAW, edgecolor="k", lw=0.4, label="π_raw")
    ax1.bar(ks + w / 2, pi_cal, w, color=C_CAL, edgecolor="k", lw=0.4, label="π_cal")
    ax1.set_xlabel("handoff position k", fontsize=11)
    ax1.set_ylabel("routing probability  π(k)", fontsize=11)
    axg = ax1.twinx()
    axg.plot(ks, gapbar, color=C_GAP, lw=2.4, marker="o", ms=4, ls="--",
             label="marginal gap_bar(k)  [reference only]")
    axg.set_ylabel("marginal gap_bar(k)  (tokens)", fontsize=11, color="#555"); axg.set_ylim(bottom=0)
    ax1.set_title("Routing pi (calibration's lever) vs marginal gap (reference)",
                  fontsize=11.5, fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = axg.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right"); ax1.grid(axis="y", alpha=0.2)
    ax2.bar(ks - w / 2, lbk_raw, w, color=C_RAW, edgecolor="k", lw=0.4, label=f"raw (Σ C(k)={lr:.2f})")
    ax2.bar(ks + w / 2, lbk_cal, w, color=C_CAL, edgecolor="k", lw=0.4, label=f"calib (Σ C(k)={lc:.2f})")
    ax2.set_xlabel("chosen handoff position k_hat", fontsize=11)
    ax2.set_ylabel("loss contribution  C(k) = π(k)·g_cond(k)", fontsize=11)
    ax2.set_title(f"Loss contribution by position  C(k) = π(k)·g_cond(k)  (exact)\n"
                  f"dMAT = {lr - lc:.2f}", fontsize=11.5, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper right"); ax2.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{WL[ds]} — MAT loss = Σ_k C(k),  C(k)=π(k)·g_cond(k)   (g_cond = conditional gap; exact)"
                 f"   n={int(N)}", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(od / "routing_decomp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


for ds in DS:
    od = BASE / ds; od.mkdir(parents=True, exist_ok=True)
    # ---- per-position gains ----
    d = json.load(open(f"/workspace/tmp/perpos2_{ds}.json"))
    n = np.array(d["rn"]); n = np.where(n == 0, 1, n)
    rh = np.array(d["rh"]) / n; rt = np.array(d["rt"]) / n
    gr = np.array(d["raw"]) / n
    perpos_draw(od, ds, rh, rt, gr[0], gr[1],
                "RAW per-position gains (no calibration)", "perpos_gain_raw.png", "upper right")
    perpos_draw(od, ds, rh, rt, rh, rt,
                "PERFECT calibration (illustrative) — expected ≡ realized",
                "perpos_gain_perfect.png", "lower right")
    for c, v in d["combos"].items():
        h, t = c.split("+"); g = np.array(v) / n
        perpos_draw(od, ds, rh, rt, g[0], g[1],
                    f"Calibrated per-position gains — {h} head + {t} tail",
                    f"perpos_gain_calib_{h}_{t}.png", "lower right")
    # ---- routing decomposition ----
    r = json.load(open(f"/workspace/tmp/ckv5_{ds}.json"))
    N = r["n"]; K = len(r["cnt_cal"]); ks = np.arange(K)
    pi_raw = np.array(r["cnt_raw"]) / N; pi_cal = np.array(r["cnt_cal"]) / N
    nA = np.array(r["nA"]); gapbar = (r["sumbest"] / N) - np.array(r["sumA"]) / np.where(nA == 0, 1, nA)
    lbk_raw = np.array(r["gl_raw"]) / N; lbk_cal = np.array(r["gl_cal"]) / N
    ymax = max(pi_raw.max(), pi_cal.max())
    cnt_raw = np.array(r["cnt_raw"]); cnt_cal = np.array(r["cnt_cal"])
    gc_raw = np.array(r["gl_raw"]) / np.where(cnt_raw == 0, 1, cnt_raw)
    gc_cal = np.array(r["gl_cal"]) / np.where(cnt_cal == 0, 1, cnt_cal)
    gcmax = max(gc_raw.max(), gc_cal.max())
    routing_arm(od, ds, ks, pi_raw, gc_raw, C_RAW, "raw", lbk_raw.sum(), ymax, gcmax)
    routing_arm(od, ds, ks, pi_cal, gc_cal, C_CAL, "cal", lbk_cal.sum(), ymax, gcmax)
    lymax = max(lbk_raw.max(), lbk_cal.max())
    loss_contrib_arm(od, ds, ks, lbk_raw, gc_raw, C_RAW, "raw", lymax, gcmax)
    loss_contrib_arm(od, ds, ks, lbk_cal, gc_cal, C_CAL, "cal", lymax, gcmax)
    routing_decomp(od, ds, ks, pi_raw, pi_cal, gapbar, lbk_raw, lbk_cal, N)
    print(f"[{ds}] -> {od}  (loss raw={lbk_raw.sum():.2f} cal={lbk_cal.sum():.2f})", flush=True)
print("ALL per-workload motivation figures done")
