# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import os
#!/usr/bin/env python3
"""Two independent images:
  1. boundary_density_switchforcing.png -- per-workload bar of the CORRECTED count
     (suffix-only <-> dflash-only handoffs /1K = verify steps the hybrid must spend).
  2. boundary_count_fix.png -- scatter: team count (fails, rho<0) vs switch-forcing
     count (works, rho>0) against the Kim gain (21 units).
"""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IDIR = Path(os.environ.get("INTERP_DIR", "results/interp_validation"))
FIG = IDIR / "figures"
WLC = {"spider": "#4C78A8", "swebench": "#F58518", "bfcl": "#54A24B", "specbench": "#B279A2"}
rows = json.load(open(IDIR / "switch_forcing_count.json"))


def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        od = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j < n and v[od[j]] == v[od[i]]:
                j += 1
            for k in range(i, j):
                r[od[k]] = (i + j - 1) / 2.0
            i = j
        return r
    rx, ry = rank(xs), rank(ys); mx = sum(rx) / n; my = sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


# ---- 1. per-workload bar of switch-forcing count (token-weighted mean) ----
agg = defaultdict(lambda: [0.0, 0.0])
for r in rows:
    a = agg[r["wl"]]; a[0] += r["sw_excl"]; a[1] += 1
order = ["spider", "swebench", "bfcl", "specbench"]
vals = [agg[w][0] / agg[w][1] for w in order]
fig, ax = plt.subplots(figsize=(8.2, 5.4))
bars = ax.bar(order, vals, color=[WLC[w] for w in order], edgecolor="k", linewidth=0.5,
              width=0.62, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.015, f"{v:.1f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("switch-forcing boundaries per 1K tokens", fontsize=11.5)
ax.set_xlabel("workload", fontsize=11)
ax.set_title("Corrected unified metric: switch-forcing boundary density\n"
             "(suffix-only <-> dflash-only handoffs = verify steps the SD hybrid must spend)",
             fontsize=11.5, fontweight="bold")
ax.grid(axis="y", alpha=0.2, zorder=0)
ax.set_ylim(0, max(vals) * 1.15)
fig.tight_layout()
fig.savefig(FIG / "boundary_density_switchforcing.png", dpi=150, bbox_inches="tight")
print("saved ->", FIG / "boundary_density_switchforcing.png")

# ---- 2. scatter fix: team count vs switch-forcing, against Kim gain ----
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
for ax, m, lab in ((axes[0], "bnd_pri", "TEAM count: all region s<->d boundaries /1K"),
                   (axes[1], "sw_excl", "FIXED count: switch-forcing handoffs /1K")):
    for r in rows:
        ax.scatter(r[m], r["g_kim"], s=55, c=WLC[r["wl"]], edgecolor="k",
                   linewidth=0.4, alpha=0.9, zorder=3)
    rho = spearman([r[m] for r in rows], [r["g_kim"] for r in rows])
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xlabel(lab, fontsize=10)
    ax.set_title(f"rho vs Kim gain = {rho:+.2f}", fontsize=13, fontweight="bold",
                 color=("#1a7d1a" if rho >= 0.5 else "#b00"))
    ax.grid(True, alpha=0.2)
axes[0].set_ylabel("Kim gain: compose - suffix hybrid (MAT)", fontsize=10)
handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=8, label=w)
           for w, c in WLC.items()]
axes[1].legend(handles=handles, fontsize=9, loc="lower right")
fig.suptitle("Kim's COUNT is right -- but count SWITCH-FORCING boundaries, not all "
             "region adjacencies", fontsize=12.5, y=1.02, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(FIG / "boundary_count_fix.png", dpi=150, bbox_inches="tight")
print("saved ->", FIG / "boundary_count_fix.png")
