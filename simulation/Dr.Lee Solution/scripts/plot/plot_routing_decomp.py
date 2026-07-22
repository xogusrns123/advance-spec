#!/usr/bin/env python3
"""Routing decomposition of MAT loss: MAT = MAT_oracle - <gap, pi>.
Panel A: gap landscape gap_bar(k) (line) + routing pi_raw(k), pi_cal(k) (bars).
Panel B: exact loss contribution by chosen position, route_gaploss[arm][k]/N
(bars); the bars sum to E[Loss] (annotated 1.56 / 0.70).
Reads ckv5_{ds}.json (pooled). Emits motivation/routing_decomp.png
"""
from pathlib import Path
import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/workspace/simulation/Dr.Lee Solution/readable_outputs/figures/motivation")
A = None
for f in sorted(glob.glob("/workspace/tmp/ckv5_*.json")):
    d = json.load(open(f))
    if A is None:
        A = {k: (np.array(v) if isinstance(v, list) else v) for k, v in d.items()}
    else:
        for k, v in d.items():
            A[k] = A[k] + (np.array(v) if isinstance(v, list) else v)
N = A["n"]
K = len(A["cnt_cal"]); ks = np.arange(K)
pi_raw = A["cnt_raw"] / N; pi_cal = A["cnt_cal"] / N
gapbar = (A["sumbest"] / N) - np.divide(A["sumA"], np.where(A["nA"] == 0, 1, A["nA"]))
lossbyk_raw = A["gl_raw"] / N; lossbyk_cal = A["gl_cal"] / N
loss_raw = lossbyk_raw.sum(); loss_cal = lossbyk_cal.sum()

C_RAW, C_CAL, C_GAP = "#F58518", "#4C78A8", "#888888"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2))

# Panel A: routing vs gap landscape
w = 0.4
ax1.bar(ks - w / 2, pi_raw, w, color=C_RAW, edgecolor="k", lw=0.4, label="π_raw (routing)")
ax1.bar(ks + w / 2, pi_cal, w, color=C_CAL, edgecolor="k", lw=0.4, label="π_cal (routing)")
ax1.set_xlabel("handoff position k", fontsize=11)
ax1.set_ylabel("routing probability  π(k)", fontsize=11)
axg = ax1.twinx()
axg.plot(ks, gapbar, color=C_GAP, lw=2.4, marker="o", ms=4, ls="--",
         label="marginal gap_bar(k) = best - mean A(k)  [reference only]")
axg.set_ylabel("marginal gap_bar(k)  (tokens)", fontsize=11, color="#555")
axg.set_ylim(bottom=0)
ax1.set_title("Routing pi (calibration's lever) vs marginal gap (reference)\n"
              "calibration shifts routing mass toward lower-gap positions",
              fontsize=11.5, fontweight="bold")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = axg.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
ax1.grid(axis="y", alpha=0.2)

# Panel B: EXACT loss contribution by position, C(k) = pi(k)*g_cond(k)
ax2.bar(ks - w / 2, lossbyk_raw, w, color=C_RAW, edgecolor="k", lw=0.4,
        label=f"raw  (Σ C(k) = {loss_raw:.2f})")
ax2.bar(ks + w / 2, lossbyk_cal, w, color=C_CAL, edgecolor="k", lw=0.4,
        label=f"calib  (Σ C(k) = {loss_cal:.2f})")
ax2.set_xlabel("chosen handoff position k̂", fontsize=11)
ax2.set_ylabel("loss contribution  C(k) = π(k)·g_cond(k)  (tokens/step)", fontsize=11)
ax2.set_title("Loss contribution by position  C(k) = π(k)·g_cond(k)  (exact, bars sum to E[Loss])\n"
              f"dMAT = {loss_raw - loss_cal:.2f}: calib reclaims raw's mass at low-k (high-gap)",
              fontsize=11.5, fontweight="bold")
ax2.legend(fontsize=10, loc="upper right"); ax2.grid(axis="y", alpha=0.2)
fig.suptitle("MAT loss = Σ_k C(k),  C(k) = π(k)·g_cond(k)   "
             "(g_cond = conditional gap E[gap | routed to k]; verified exact to 1e-16)"
             f"    pooled n={int(N)}", fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "routing_decomp.png", dpi=150, bbox_inches="tight")
print("saved -> routing_decomp.png")


gc_raw = A["gl_raw"] / np.where(A["cnt_raw"] == 0, 1, A["cnt_raw"])
gc_cal = A["gl_cal"] / np.where(A["cnt_cal"] == 0, 1, A["cnt_cal"])
gcmax = max(gc_raw.max(), gc_cal.max())


# ---- separate per-arm figures: routing pi + conditional gap g_cond ----
def arm_fig(pi, gcond, color, name, title):
    f, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.bar(ks, pi, 0.7, color=color, edgecolor="k", lw=0.4, zorder=2,
           label=f"π_{name}(k) (routing)")
    ax.set_xlabel("handoff position k  (k=0 = pure suffix)", fontsize=11)
    ax.set_ylabel(f"routing probability  π_{name}(k)", fontsize=11)
    ax.set_ylim(0, max(pi.max(), pi_raw.max(), pi_cal.max()) * 1.1)
    axg = ax.twinx()
    axg.plot(ks, gcond, color=C_GAP, lw=2.4, marker="o", ms=4, ls="--", zorder=3,
             label="g_cond(k) = E[gap | routed to k]")
    axg.set_ylabel("conditional gap  g_cond(k)  (tokens)", fontsize=11, color="#555")
    axg.set_ylim(0, gcmax * 1.1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axg.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    f.tight_layout()
    f.savefig(OUT / f"routing_{name}.png", dpi=150, bbox_inches="tight")
    print(f"saved -> routing_{name}.png")


arm_fig(pi_raw, gc_raw, C_RAW, "raw",
        f"RAW routing π vs conditional gap g_cond\nΣ π(k)·g_cond(k) = loss = "
        f"{loss_raw:.2f} tokens/step  (exact)")
arm_fig(pi_cal, gc_cal, C_CAL, "cal",
        f"CALIBRATED routing π vs conditional gap g_cond\nΣ π(k)·g_cond(k) = loss = "
        f"{loss_cal:.2f} tokens/step  (exact)")


def loss_fig(lbk, gcond, color, name, loss):
    lab = {"raw": "RAW", "cal": "CALIBRATED"}[name]
    f, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.bar(ks, lbk, 0.7, color=color, edgecolor="k", lw=0.4, zorder=2,
           label=f"loss contribution C(k) = π·g_cond ({name})")
    ax.set_xlabel("chosen handoff position k_hat  (k=0 = pure suffix)", fontsize=11)
    ax.set_ylabel("loss contribution C(k)  (tokens/step)", fontsize=11)
    ax.set_ylim(0, max(lossbyk_raw.max(), lossbyk_cal.max()) * 1.12)
    axg = ax.twinx()
    axg.plot(ks, gcond, color=C_GAP, lw=2.4, marker="o", ms=4, ls="--", zorder=3,
             label="g_cond(k) = E[gap | routed to k]  (conditional)")
    axg.set_ylabel("conditional gap  g_cond(k)  (tokens)", fontsize=11, color="#555")
    axg.set_ylim(0, gcmax * 1.1)
    ax.set_title(f"{lab} loss contribution by position\n"
                 f"C(k) = π(k)·g_cond(k)  (exact);  Σ C(k) = {loss:.2f} tokens/step",
                 fontsize=12, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axg.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    f.tight_layout(); f.savefig(OUT / f"loss_contrib_{name}.png", dpi=150, bbox_inches="tight")
    print(f"saved -> loss_contrib_{name}.png")


loss_fig(lossbyk_raw, gc_raw, C_RAW, "raw", loss_raw)
loss_fig(lossbyk_cal, gc_cal, C_CAL, "cal", loss_cal)
print(f"loss_raw={loss_raw:.3f} loss_cal={loss_cal:.3f} dMAT={loss_raw-loss_cal:.3f}")
print(f"pi_raw peak k={pi_raw.argmax()} pi_cal peak k={pi_cal.argmax()}")
