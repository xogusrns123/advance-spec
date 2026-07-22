#!/usr/bin/env python3
"""Standardized per-position decomposition figure set into
readable_outputs/figures/loss_decomp/{ds}/ (+ pooled at loss_decomp/).
Unified accept notation:  â(k) expected accept (controller estimate),
a(k) realized accept value (a(k*)=oracle best),  ā(k) unconditional mean accept,
a(k|select) conditional mean accept,  pi(k)=argmax_k â(k) selection dist,
gap(k)=a(k*)-a(k) unconditional selection loss,  l(k|select) conditional
selection loss (= g_cond),  L(k)=pi(k)*l(k|select) loss contribution.

1 expected_gain_{raw,calib}  : stacked bar head(blue)+tail(orange)  â(k)
2 selection_dist_{raw,calib} : green bars  pi(k)
3 selection_loss_uncond      : red bars  gap(k)  (arm-independent)
4 cond_selection_loss_{raw,calib}: light-red bars l(k|select) + green line pi(k)
5 loss_contribution_{raw,calib}  : dark-red bars L(k) + green line pi + light-red line l
Driven by ckv6_{ds}.json.
"""
from pathlib import Path
import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/workspace/simulation/Dr.Lee Solution/readable_outputs/figures/loss_decomp")
WL = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench Verified",
      "spider": "Spider2-DBT", "tau2": "tau2-bench", "POOLED": "All workloads (pooled)"}
C_HEAD, C_TAIL = "#4C78A8", "#F58518"       # head blue, tail orange
C_PI = "#2ca02c"                            # selection dist green (loss-based folder)
C_PIB = "#1f77b4"                           # selection dist blue (contribution folder)
C_RED = "#d62728"                           # unconditional loss red
C_LRED = "#f19999"                          # conditional loss light red
C_DRED = "#8b1a1a"                          # loss contribution dark red
C_SIG = "#9467bd"                           # severity sigma purple
C_CORR = "#2a9d8f"                          # correct rate teal
C_MATC = "#3d8c40"                          # MAT contribution green
C_ACC = "#74a662"                           # conditional accept mid-green
C_ABAR = "#a8d08d"                          # accept landscape light green
C_EAC = "#8c564b"                           # E[a|correct] brown (token line)
ARM = {"raw": "raw", "cal": "calib"}

# unified legend definitions (shown verbatim in every figure)
D_HEAD = "head accept  G_k"
D_TAIL = "tail accept  S_k·T_k"
D_V = "â(k) = expected accept (head + tail)"
D_PI = "π(k) = argmax_k â(k)   (selection dist)"
D_GAP = "gap(k) = a(k*) − a(k)   (unconditional)"
D_LCOND = "l(k|select) = E[a(k*) − a(k) | select]"
D_MISS = "miss(k) = P(k ≠ k* | select)"
D_SIG = "σ(k) = E[a(k*) − a(k) | select, miss]"
D_L2 = "L(k) = π(k)·l(k|select)"
D_L3 = "L(k) = π(k)·miss(k)·σ(k)"
D_CORR = "correct(k) = P(k = k* | select)"
D_MATC = "C_MAT(k) = π(k)·E[accept | select k]"
D_ACOND = "a(k|select) = E[accept | select]"
D_ABAR = "ā(k) = mean accept at k (all rounds)"
# unified factor labels for the MAT-contribution overlay
D_PK = "P(k)"
D_PCORR = "P(correct|k)"
D_EACORR = "E[a(k)|correct]"


LIM = {}   # global y-limits per figure type (filled in pass 1)


def derive(d):
    N = d["n"]; nA = np.array(d["nA"])
    gapbar = d["sumbest"] / N - np.array(d["sumA"]) / np.where(nA == 0, 1, nA)
    abar = np.where(nA > 0, np.array(d["sumA"]) / np.where(nA == 0, 1, nA), 0.0)
    out = {"gap": gapbar, "abar": abar}
    for arm in ("raw", "cal"):
        cnt = np.array(d[f"cnt_{arm}"], float); gl = np.array(d[f"gl_{arm}"])
        mс = np.array(d[f"miss_{arm}"], float)
        acc = np.array(d.get(f"acc_{arm}", np.zeros_like(cnt)), float)
        acorr = np.array(d.get(f"acccorr_{arm}", np.zeros_like(cnt)), float)
        ncorr = cnt - mс                      # correct-round count at k
        m = np.where(cnt > 0, mс / np.where(cnt == 0, 1, cnt), 0.0)
        out[arm] = dict(V=(np.array(d[f"eG_{arm}"]) + np.array(d[f"eST_{arm}"])) / N,
                        G=np.array(d[f"eG_{arm}"]) / N, ST=np.array(d[f"eST_{arm}"]) / N,
                        pi=cnt / N, gl=gl,
                        lcond=np.where(cnt > 0, gl / np.where(cnt == 0, 1, cnt), 0.0),
                        m=m, correct=np.where(cnt > 0, 1.0 - m, np.nan),
                        sev=np.where(mс > 0, gl / np.where(mс == 0, 1, mс), 0.0),
                        L=gl / N, matc=acc / N,
                        acond=np.where(cnt > 0, acc / np.where(cnt == 0, 1, cnt), 0.0),
                        eacorr=np.where(ncorr > 0, acorr / np.where(ncorr <= 0, 1, ncorr), 0.0))
    return out


def g1_expected_gain(od, tag, ks, G, ST, arm):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, G, 0.7, color=C_HEAD, edgecolor="k", lw=0.4, label=D_HEAD)
    ax.bar(ks, ST, 0.7, bottom=G, color=C_TAIL, edgecolor="k", lw=0.4, label=D_TAIL)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("expected accept  â(k)  (tokens)", fontsize=11); ax.set_xticks(ks)
    ax.set_ylim(0, LIM["V"])
    ax.set_title(f"{WL[tag]} — expected accept â(k)  [{ARM[arm]} prob]\n"
                 "head + tail (stacked); calibration changes these expected values",
                 fontsize=12, fontweight="bold")
    ax.legend(title=D_V, fontsize=10, title_fontsize=9.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"expected_gain_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def g2_selection_dist(od, tag, ks, pi, V, arm, pi_color=C_PI):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, pi, 0.7, color=pi_color, edgecolor="k", lw=0.4, label=D_PI)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("selection distribution  π(k)  (portion)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["pi"])
    axr = ax.twinx()
    axr.plot(ks, V, color="black", lw=2.2, marker="o", ms=4, label=D_V)
    axr.set_ylabel("expected accept  â(k)  (tokens)", fontsize=11)
    axr.set_ylim(0, LIM["V"])
    ax.set_title(f"{WL[tag]} — selection distribution π(k)  [{ARM[arm]}]\n"
                 "controller picks argmax â(k); calibration reshapes â -> π shifts",
                 fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"selection_dist_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def g3_uncond_loss(od, tag, ks, gapbar):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, gapbar, 0.7, color=C_RED, edgecolor="k", lw=0.4, label=D_GAP)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("unconditional selection loss  gap(k)  (tokens)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["gap"])
    ax.set_title(f"{WL[tag]} — unconditional selection loss  gap(k)\n"
                 "loss at each position before selection (routing-independent)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / "selection_loss_uncond.png", dpi=150)
    plt.close(fig)


def g4_cond_loss(od, tag, ks, lcond, pi, arm):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, lcond, 0.7, color=C_LRED, edgecolor="k", lw=0.4, label=D_LCOND)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("conditional selection loss  l(k | select)  (tokens)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["lcond"])
    axr = ax.twinx()
    axr.plot(ks, pi, color=C_PI, lw=2.2, marker="o", ms=4, label=D_PI)
    axr.set_ylabel("selection distribution  π(k)", fontsize=11, color=C_PI)
    axr.set_ylim(0, LIM["pi"])
    ax.set_title(f"{WL[tag]} — conditional selection loss l(k|select)  [{ARM[arm]}]\n"
                 "selection changes which rounds land at k -> l(k|select) shifts",
                 fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"cond_selection_loss_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def g5_loss_contrib(od, tag, ks, L, pi, lcond, arm):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, L, 0.7, color=C_DRED, edgecolor="k", lw=0.4, label=D_L2)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("loss contribution  L(k)  (tokens/step)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["L"])
    axr = ax.twinx()
    axr.plot(ks, pi, color=C_PI, lw=2.0, marker="o", ms=3.5, label=D_PI)
    axr.plot(ks, lcond / LIM["lcond"] * LIM["pi"], color=C_LRED, lw=2.0,
             marker="s", ms=3.5, ls="--", label=D_LCOND + "  [scaled]")
    axr.set_ylabel("π(k)  (l scaled to π axis)", fontsize=11, color="#555")
    axr.set_ylim(0, LIM["pi"])
    ax.set_title(f"{WL[tag]} — loss contribution L(k)  [{ARM[arm]}]\n"
                 f"L(k)=π(k)·l(k|select);  Σ L(k) = {L.sum():.2f} tokens/step (exact)",
                 fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"loss_contribution_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def g6_miss_rate(od, tag, ks, m, pi, arm):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, m, 0.7, color=C_RED, edgecolor="k", lw=0.4, label=D_MISS)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("miss rate  miss(k)  (probability)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["miss"])
    axr = ax.twinx()
    axr.plot(ks, pi, color=C_PI, lw=2.2, marker="o", ms=4, label=D_PI)
    axr.set_ylabel("selection distribution  π(k)", fontsize=11, color=C_PI)
    axr.set_ylim(0, LIM["pi"])
    ax.set_title(f"{WL[tag]} — miss rate miss(k)  [{ARM[arm]}]\n"
                 "how often the chosen handoff is suboptimal",
                 fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"miss_rate_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def g_severity(od, tag, ks, sev, arm):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, sev, 0.7, color=C_SIG, edgecolor="k", lw=0.4, label=D_SIG)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("severity  σ(k)  (tokens)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["sev"])
    ax.set_title(f"{WL[tag]} — severity σ(k)  [{ARM[arm]}]\n"
                 "how large the loss is when the chosen handoff misses",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"severity_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def g6b_miss_rate_basic(od, tag, ks, m, arm):
    """miss rate — bars only, no overlay line."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, m, 0.7, color=C_RED, edgecolor="k", lw=0.4, label=D_MISS)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("miss rate  miss(k)  (probability)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["miss"])
    ax.set_title(f"{WL[tag]} — miss rate miss(k)  [{ARM[arm]}]\n"
                 "how often the chosen handoff is suboptimal",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"miss_rate_basic_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def g5b_loss_contrib_basic(od, tag, ks, L, arm):
    """loss contribution — bars only, no overlay lines."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, L, 0.7, color=C_DRED, edgecolor="k", lw=0.4, label=D_L3)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("loss contribution  L(k)  (tokens/step)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["L"])
    ax.set_title(f"{WL[tag]} — loss contribution L(k)  [{ARM[arm]}]\n"
                 f"L(k)=π(k)·miss(k)·σ(k);  Σ L(k) = {L.sum():.2f} tokens/step (exact)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"loss_contribution_basic_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


# ===== contribution-based (accept-side) mirror of the loss family =====

def g_accept_landscape(od, tag, ks, abar):
    """mirror of g3 gap: arm-independent unconditional mean accept at each k."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, abar, 0.7, color=C_ABAR, edgecolor="k", lw=0.4, label=D_ABAR)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("mean accept  ā(k)  (tokens)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["abar"])
    ax.set_title(f"{WL[tag]} — accept landscape ā(k)\n"
                 "expected accept at each position before selection (routing-independent)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / "accept_landscape.png", dpi=150)
    plt.close(fig)


def g_cond_accept(od, tag, ks, acond, pi, arm):
    """mirror of g4 l(k|select): conditional realized accept a(k|select) + π line."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, acond, 0.7, color=C_ACC, edgecolor="k", lw=0.4, label=D_ACOND)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("conditional accept  a(k | select)  (tokens)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["acond"])
    axr = ax.twinx()
    axr.plot(ks, pi, color=C_PIB, lw=2.2, marker="o", ms=4, label=D_PI)
    axr.set_ylabel("selection distribution  π(k)", fontsize=11, color=C_PIB)
    axr.set_ylim(0, LIM["pi"])
    ax.set_title(f"{WL[tag]} — conditional accept a(k|select)  [{ARM[arm]}]\n"
                 "realized accept among rounds routed to k",
                 fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"cond_accept_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def g_correct_rate_basic(od, tag, ks, correct, arm):
    """mirror of g6b miss_rate_basic: correct(k) = 1 - miss(k), bars only."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, correct, 0.7, color=C_CORR, edgecolor="k", lw=0.4, label=D_CORR)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("correct rate  correct(k)  (probability)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["correct"])
    ax.set_title(f"{WL[tag]} — correct rate correct(k)  [{ARM[arm]}]\n"
                 "how often the chosen handoff is optimal",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"correct_rate_basic_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def g_correct_rate(od, tag, ks, correct, pi, arm):
    """mirror of g6 miss_rate: correct(k) bars + π selection line."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, correct, 0.7, color=C_CORR, edgecolor="k", lw=0.4, label=D_CORR)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("correct rate  correct(k)  (probability)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["correct"])
    axr = ax.twinx()
    axr.plot(ks, pi, color=C_PIB, lw=2.2, marker="o", ms=4, label=D_PI)
    axr.set_ylabel("selection distribution  π(k)", fontsize=11, color=C_PIB)
    axr.set_ylim(0, LIM["pi"])
    ax.set_title(f"{WL[tag]} — correct rate correct(k)  [{ARM[arm]}]\n"
                 "how often the chosen handoff is optimal",
                 fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="lower right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"correct_rate_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def g_mat_contribution_basic(od, tag, ks, matc, arm):
    """mirror of g5b loss_contribution_basic: realized MAT mass per position, bars only."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, matc, 0.7, color=C_MATC, edgecolor="k", lw=0.4, label=D_MATC)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("MAT contribution  C_MAT(k)  (tokens/step)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["matc"])
    ax.set_title(f"{WL[tag]} — MAT contribution C_MAT(k)  [{ARM[arm]}]\n"
                 f"C_MAT(k)=π(k)·E[accept|select k];  Σ C_MAT(k) = {matc.sum():.2f} tokens/step",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"mat_contribution_basic_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def g_mat_contribution(od, tag, ks, matc, pi, correct, eacorr, arm):
    """3 independent scales (no log): LEFT = C_MAT bars (tokens/step),
    inner RIGHT = probability P(k)/P(correct|k), outer RIGHT = E[a(k)|correct] (tokens)."""
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    b = ax.bar(ks, matc, 0.7, color=C_MATC, edgecolor="k", lw=0.4,
               label="C_MAT(k) = P(k)·E[a(k)|select]")
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("MAT contribution  C_MAT(k)  (tokens/step)", fontsize=11, color=C_MATC)
    ax.tick_params(axis="y", colors=C_MATC)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["matc"])
    # inner right axis: probability
    axp = ax.twinx()
    axp.plot(ks, pi, color=C_PIB, lw=2.0, marker="o", ms=3.5, label=D_PK)
    axp.plot(ks, correct, color=C_CORR, lw=2.0, marker="^", ms=3.5, label=D_PCORR)
    axp.set_ylabel("probability", fontsize=11); axp.set_ylim(0, 1.05)
    # outer right axis: E[a|correct] in tokens (own scale)
    axe = ax.twinx()
    axe.spines["right"].set_position(("outward", 52))
    axe.plot(ks, eacorr, color=C_EAC, lw=2.0, marker="s", ms=3.5, ls="--", label=D_EACORR)
    axe.set_ylabel("E[a(k)|correct]  (tokens)", fontsize=11, color=C_EAC)
    axe.tick_params(axis="y", colors=C_EAC); axe.set_ylim(0, LIM["eaL"])
    axe.spines["right"].set_color(C_EAC)
    ax.set_title(f"{WL[tag]} — MAT contribution C_MAT(k)  [{ARM[arm]}]\n"
                 f"C_MAT(k)=P(k)·E[a(k)|select];  Σ C_MAT(k) = {matc.sum():.2f} tokens/step",
                 fontsize=12, fontweight="bold")
    h0, l0 = ax.get_legend_handles_labels(); h1, l1 = axp.get_legend_handles_labels()
    h2, l2 = axe.get_legend_handles_labels()
    leg = axe.legend(h0 + h1 + h2, l0 + l1 + l2, fontsize=8.5, loc="upper center",
                     framealpha=0.95)
    leg.set_zorder(100); ax.grid(axis="y", alpha=0.25)
    fig.savefig(od / f"mat_contribution_{ARM[arm]}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def g7_loss_3factor(od, tag, ks, L, pi, m, sev, arm):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(ks, L, 0.7, color=C_DRED, edgecolor="k", lw=0.4, label=D_L3)
    ax.set_xlabel("handoff position k", fontsize=11)
    ax.set_ylabel("loss contribution  L(k)  (tokens/step)", fontsize=11)
    ax.set_xticks(ks); ax.set_ylim(0, LIM["L"])
    axr = ax.twinx()
    axr.plot(ks, pi / LIM["pi"], color=C_PI, lw=2.0, marker="o", ms=3.5, label=D_PI)
    axr.plot(ks, m / LIM["miss"], color=C_RED, lw=2.0, marker="^", ms=3.5, label=D_MISS)
    axr.plot(ks, sev / LIM["sev"], color=C_SIG, lw=2.0, marker="s", ms=3.5, ls="--",
             label=D_SIG)
    axr.set_ylabel("factors (each scaled to global max)", fontsize=11, color="#555")
    axr.set_ylim(0, 1.05)
    ax.set_title(f"{WL[tag]} — loss contribution L(k)  [{ARM[arm]}]\n"
                 f"L(k)=π(k)·miss(k)·σ(k);  Σ L(k) = {L.sum():.2f} tokens/step (exact)",
                 fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    leg = axr.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", framealpha=0.95)
    leg.set_zorder(100); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"loss_contribution_3factor_{ARM[arm]}.png", dpi=150)
    plt.close(fig)


def run(tag, dv, od):
    lo = od / "loss_based"; co = od / "contribution_based"
    lo.mkdir(parents=True, exist_ok=True); co.mkdir(parents=True, exist_ok=True)
    K = len(dv["gap"]); ks = np.arange(K)
    # arm-independent landscapes (one per methodology)
    g3_uncond_loss(lo, tag, ks, dv["gap"])          # loss: gap(k)
    g_accept_landscape(co, tag, ks, dv["abar"])      # contribution: ā(k)
    for arm in ("raw", "cal"):
        a = dv[arm]
        # shared setup graphs (controller score V, selection π) -> BOTH folders
        # π green in loss_based, blue in contribution_based (bars there are green)
        for base, pic in ((lo, C_PI), (co, C_PIB)):
            g1_expected_gain(base, tag, ks, a["G"], a["ST"], arm)
            g2_selection_dist(base, tag, ks, a["pi"], a["V"], arm, pic)
        # ---- loss-based methodology ----
        g4_cond_loss(lo, tag, ks, a["lcond"], a["pi"], arm)
        g5_loss_contrib(lo, tag, ks, a["L"], a["pi"], a["lcond"], arm)
        g5b_loss_contrib_basic(lo, tag, ks, a["L"], arm)
        g6_miss_rate(lo, tag, ks, a["m"], a["pi"], arm)
        g6b_miss_rate_basic(lo, tag, ks, a["m"], arm)
        g_severity(lo, tag, ks, a["sev"], arm)
        g7_loss_3factor(lo, tag, ks, a["L"], a["pi"], a["m"], a["sev"], arm)
        # ---- contribution-based approach ----
        g_cond_accept(co, tag, ks, a["acond"], a["pi"], arm)
        g_mat_contribution(co, tag, ks, a["matc"], a["pi"], a["correct"], a["eacorr"], arm)
        g_mat_contribution_basic(co, tag, ks, a["matc"], arm)
        g_correct_rate(co, tag, ks, a["correct"], a["pi"], arm)
        g_correct_rate_basic(co, tag, ks, a["correct"], arm)
    print(f"[{tag}] loss_based/ + contribution_based/ figures saved -> {od}")


# ---- load all tags, derive, compute GLOBAL per-type y-limits ----
DV = {}; pooled = None
for f in sorted(glob.glob("/workspace/tmp/ckv6_*.json")):
    ds = Path(f).stem.replace("ckv6_", ""); d = json.load(open(f))
    DV[ds] = derive(d)
    if pooled is None:
        pooled = {k: (np.array(v) if isinstance(v, list) else v) for k, v in d.items()}
    else:
        for k, v in d.items():
            pooled[k] = pooled[k] + (np.array(v) if isinstance(v, list) else v)
DV["POOLED"] = derive({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pooled.items()})

mx = lambda key: max(dv[a][key].max() for dv in DV.values() for a in ("raw", "cal"))
LIM.update(V=mx("V") * 1.08, pi=mx("pi") * 1.12, lcond=mx("lcond") * 1.12,
           L=mx("L") * 1.15, gap=max(dv["gap"].max() for dv in DV.values()) * 1.12,
           miss=min(1.0, mx("m") * 1.12), sev=mx("sev") * 1.12,
           correct=1.02, matc=mx("matc") * 1.15, acond=mx("acond") * 1.12,
           abar=max(dv["abar"].max() for dv in DV.values()) * 1.12,
           eaL=max(dv["cal"]["eacorr"].max() for dv in DV.values()) * 1.12)
print("global y-limits:", {k: round(v, 3) for k, v in LIM.items()})

for tag, dv in DV.items():
    run(tag, dv, BASE / (tag if tag != "POOLED" else ""))
print("ALL standardized decomposition figures done")
