"""27B MTP served selection-accuracy + MAT ladder (parallel to the 14B EAGLE3 study),
plus the regime-contrast panel: WHEN does calibration matter?

Both ladders are REAL-SERVING, held-out, decisive + alive-conditioned.
14B numbers are the recorded served result (project_selacc_calib_ceiling);
27B numbers come from analyze_served_selacc_27b.py on this run's logs."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "simulation/results/calib_why_analysis/figures"

# (label, selacc, MAT)
ARMS = ["raw", "calib\nhist", "calib\nisot", "calib\nlogit", "calib\nbeta", "disc\n(Bayes)", "oracle"]

E14 = {  # Qwen3-14B EAGLE3, served, held-out
    "sel": [0.759, 0.774, 0.776, 0.774, 0.777, 0.798, 1.000],
    "mat": [1.328, 1.349, 1.347, 1.349, 1.351, 1.385, 1.783],
    "vanilla": 1.05,
}
M27 = {  # Qwen3.5-27B MTP, served, held-out
    "sel": [0.8179, 0.8660, 0.8695, 0.8691, 0.8633, 0.9018, 1.0000],
    "mat": [6.9004, 7.3951, 7.4228, 7.4669, 7.3837, 7.8731, 8.6757],
    "vanilla": 7.754,
}

COL = ["#888888", "#6fa8dc", "#6fa8dc", "#6fa8dc", "#6fa8dc", "#e69138", "#6aa84f"]


def ladder(ax, d, key, title, ylab, ref_lab):
    vals = d[key]
    bars = ax.bar(range(len(ARMS)), vals, color=COL, edgecolor="black", lw=0.6)
    ax.axhline(d["vanilla"] if key == "mat" else 1.0, ls=":", color="black", lw=1.2,
               label=ref_lab)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}" if key == "sel" else f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS, fontsize=8)
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    lo = min(vals + [d["vanilla"]]) if key == "mat" else min(vals)
    ax.set_ylim(lo - (max(vals) - lo) * 0.12, max(vals) * 1.06)


# ---- Figure 1: 27B MTP ladder (the requested reproduction) ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
ladder(axes[0], M27, "sel", "Qwen3.5-27B MTP — served selection accuracy",
       "decisive sel. accuracy", "oracle = 1.0")
ladder(axes[1], M27, "mat", "Qwen3.5-27B MTP — served MAT",
       "mean accepted tokens", "vanilla MTP = 7.75")
fig.suptitle("27B MTP chain-hybrid (MTP-token vs suffix select-1), real-serving held-out",
             fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/served_ladder_27b_mtp.png", dpi=130)
print(f"wrote {OUT_DIR}/served_ladder_27b_mtp.png")

# ---- Figure 2: regime contrast — fraction of raw->oracle gap recovered -----
def recov(d, key):
    base, orc = d[key][0], d[key][-1]
    gap = orc - base
    return [(d[key][i] - base) / gap * 100 for i in range(len(ARMS))]

fig2, ax = plt.subplots(figsize=(10.5, 5.0))
x = np.arange(len(ARMS)); w = 0.38
r14 = recov(E14, "mat"); r27 = recov(M27, "mat")
ax.bar(x - w/2, r14, w, label="14B EAGLE3 (weak drafter, MAT 1.05)", color="#9fc5e8", edgecolor="black", lw=0.6)
ax.bar(x + w/2, r27, w, label="27B MTP (strong drafter, MAT 7.75)", color="#e69138", edgecolor="black", lw=0.6)
for i in range(len(ARMS)):
    ax.text(i - w/2, r14[i], f"{r14[i]:.0f}%", ha="center", va="bottom", fontsize=7.5)
    ax.text(i + w/2, r27[i], f"{r27[i]:.0f}%", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(ARMS, fontsize=8.5)
ax.set_ylabel("% of raw→oracle MAT gap recovered")
ax.set_title("WHEN does calibration matter? — fraction of the MAT gap each arm recovers\n"
             "14B: scores carry little signal (info-capped) → calib≈raw (5%).  "
             "27B MTP: raw rule mis-scaled but signal-rich → calib recovers 32%.", fontsize=9.5)
ax.legend(fontsize=9, loc="upper left")
ax.axhline(0, color="black", lw=0.8)
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/calib_regime_contrast_mtp_vs_eagle.png", dpi=130)
print(f"wrote {OUT_DIR}/calib_regime_contrast_mtp_vs_eagle.png")

# ---- console summary -------------------------------------------------------
print("\n=== 27B MTP vs 14B EAGLE3 (served, held-out) ===")
print(f"{'arm':16s} {'27B sel':>9s} {'27B MAT':>9s} {'27B %gap':>9s} | {'14B sel':>9s} {'14B MAT':>9s} {'14B %gap':>9s}")
for i, a in enumerate([x.replace(chr(10), ' ') for x in ARMS]):
    print(f"{a:16s} {M27['sel'][i]:9.3f} {M27['mat'][i]:9.3f} {r27[i]:8.0f}% | "
          f"{E14['sel'][i]:9.3f} {E14['mat'][i]:9.3f} {r14[i]:8.0f}%")
print(f"\nvanilla: 27B MTP MAT={M27['vanilla']:.3f}  |  14B eagle MAT={E14['vanilla']:.3f}")
print("note: 27B raw select-1 (6.90) is BELOW vanilla MTP (7.75) — naive sp>ep over-picks suffix;")
print("      only disc (7.87) / oracle (8.68) beat plain MTP. calib (7.47) recovers but stays sub-vanilla.")
