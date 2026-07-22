#!/usr/bin/env python3
"""Figures for the Dr.Lee-vs-Kim interpretation validity study.

  1. ladder_6arm.png      4 workloads x 6 arms (singles / binary-switch pair /
                          compose / handoff oracle; ceilings hatched)
  2. gain_vs_structure.png  2x3 scatter: the two gains (vs best-single, vs
                          switch_real) against warm share / slot density /
                          boundary density, Spearman annotated per panel
  3. multislot_sweep.png  K vs slot count k on the controlled multislot bench

  PYTHONPATH=/workspace python3 scripts/plot_interp_validation.py \
      --dir results/interp_validation --units results/interp_validation/units.json
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# house roles (plot_mat_4way.py) + two new switch roles
COLORS = {"dflash": "#4C78A8", "suffix": "#F58518", "compose": "#54A24B",
          "handoff_oracle": "#E45756", "switch_real": "#B279A2",
          "switch_oracle": "#B279A2"}
LABELS = {"dflash": "DFlash (single)", "suffix": "Suffix (single)",
          "switch_real": "Switch (binary per-step,\ncalibrated signals)",
          "switch_oracle": "Switch ORACLE\n(perfect binary pick)",
          "compose": "Compose (calibrated\nhead+tail chain)",  # overridden in main() for raw dirs
          "handoff_oracle": "Handoff ORACLE\n(best k per round)"}
ARMS = ["dflash", "suffix", "switch_real", "switch_oracle", "compose",
        "handoff_oracle"]
WL_MARK = {"spider": "o", "swebench": "s", "bfcl": "^", "specbench": "D"}
WL_COLOR = {"spider": "#4C78A8", "swebench": "#F58518", "bfcl": "#54A24B",
            "specbench": "#E45756"}


def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and v[order[j]] == v[order[i]]:
                j += 1
            for k2 in range(i, j):
                r[order[k2]] = (i + j - 1) / 2.0
            i = j
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def fig_ladder(units, out):
    wls = ["spider", "swebench", "bfcl", "specbench"]
    rows = {u["wl"]: u for u in units if u["task"] == "__all__" and u["wl"] in wls}
    wls = [w for w in wls if w in rows]
    fig, ax = plt.subplots(figsize=(13.5, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    nw, na = len(wls), len(ARMS)
    bw = 0.8 / na
    ymax = 0.0
    for j, arm in enumerate(ARMS):
        xs = [i + (j - (na - 1) / 2) * bw for i in range(nw)]
        vs = [rows[w]["K"].get(arm, 0.0) for w in wls]
        ymax = max(ymax, max(vs))
        hatch = "//" if "oracle" in arm else None
        face = COLORS[arm] if "oracle" not in arm else "white"
        ax.bar(xs, vs, width=bw * 0.92, color=face, edgecolor=COLORS[arm],
               linewidth=1.4, hatch=hatch, label=LABELS[arm])
        for x, v in zip(xs, vs):
            ax.text(x, v + 0.06, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(nw))
    ax.set_xticklabels([w for w in wls], fontsize=11)
    ax.set_ylabel("K (mean accepted draft tokens / round)", fontsize=11)
    ax.set_title("Composition vs its two baselines — singles (Dr.Lee frame) and "
                 "per-step binary switch (Kim frame); ceilings hatched", fontsize=12)
    ax.set_ylim(0, ymax * 1.10)
    ax.legend(fontsize=8.5, ncol=6, loc="lower center",
              bbox_to_anchor=(0.5, 1.045), frameon=False)
    ax.set_title(ax.get_title(), fontsize=12, pad=48)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_scatter(units, out):
    tu = [u for u in units if u["task"] != "__all__"
          and u["wl"] in WL_MARK]
    gains = [("g_drlee", "compose − best single  (Dr.Lee frame)"),
             ("g_kim", "compose − binary switch  (Kim frame)")]
    mets = [("warm", "warm share (suffix-copyable, θ=4)"),
            ("slot100", "slots (gaps ≤ W) / 100 tok  [Dr.Lee metric]"),
            ("bound100", "warm↔cold boundaries / 100 tok  [Kim metric]")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey="row")
    for r, (g, glab) in enumerate(gains):
        for c, (m, mlab) in enumerate(mets):
            ax = axes[r][c]
            ax.set_axisbelow(True)
            ax.grid(alpha=0.3)
            ax.axhline(0, color="#999", lw=0.8)
            for wl in WL_MARK:
                pts = [(u[m], u[g]) for u in tu if u["wl"] == wl]
                if not pts:
                    continue
                ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                           marker=WL_MARK[wl], s=52, color=WL_COLOR[wl],
                           label=wl, alpha=0.85, edgecolor="white", linewidth=0.6)
            rho = spearman([u[m] for u in tu], [u[g] for u in tu])
            ax.set_title(f"ρ = {rho:+.2f}", fontsize=11,
                         fontweight="bold" if abs(rho) > 0.8 else "normal")
            if r == 1:
                ax.set_xlabel(mlab, fontsize=9.5)
            if c == 0:
                ax.set_ylabel(glab, fontsize=10)
    axes[0][0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Which structure metric predicts the composition gain? "
                 "(task-level units; Spearman ρ per panel)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_multislot(rdir, out):
    ks, data = [0, 1, 2, 4, 8], {}
    arms = ["suffix", "dflash", "switch_oracle", "compose", "handoff_oracle"]
    for k in ks:
        p = os.path.join(rdir, f"report_ms_k{k}.json")
        if not os.path.exists(p):
            return
        rep = json.load(open(p))
        for a in arms:
            data.setdefault(a, []).append(rep["arms"][a]["K"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)
    for a in arms:
        ls = "--" if "oracle" in a else "-"
        ax.plot(ks, data[a], ls, color=COLORS[a], marker="o", markersize=5,
                linewidth=2, label=LABELS[a].replace("\n", " "))
    ax.set_xlabel("novel slot count k (multislot bench)", fontsize=11)
    ax.set_ylabel("K (mean accepted / round)", fontsize=11)
    ax.set_title("Controlled multislot sweep: composition's edge does NOT grow "
                 "with slot count", fontsize=11.5)
    ax.set_xticks(ks)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp_validation")
    ap.add_argument("--units", default="results/interp_validation/units.json")
    args = ap.parse_args()
    units = json.load(open(args.units))
    if "raw" in args.dir:
        LABELS["compose"] = "Compose\n(head+tail chain)"
        LABELS["switch_real"] = "Switch (binary per-step,\nraw signals)"
    fdir = os.path.join(args.dir, "figures")
    os.makedirs(fdir, exist_ok=True)
    fig_ladder(units, os.path.join(fdir, "ladder_6arm.png"))
    fig_scatter(units, os.path.join(fdir, "gain_vs_structure.png"))
    fig_multislot(args.dir, os.path.join(fdir, "multislot_sweep.png"))


if __name__ == "__main__":
    main()
