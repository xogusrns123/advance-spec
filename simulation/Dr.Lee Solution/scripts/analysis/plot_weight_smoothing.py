#!/usr/bin/env python3
r"""Grouped-bar MAT figures (x=5 workloads, y=MAT, one bar per arm/workload).

FIG1 weight analysis: raw | head-weight | tail-weight | head+tail-weight, each at
its BEST scalar (auto-picked = max 5-wl mean within the family; chosen weight
annotated at top). Colors: raw=gray, head=blue, tail=orange, head+tail=green.

FIG2/3/4 smoothing+calibration: registry-based; each arm keeps ONE consistent
color across all figures so bars are comparable between panels.

English labels (house rule). Run inside sglang-bench:
  docker exec sglang-bench bash -lc 'cd /workspace/simulation/Dr.Lee\ Solution && \
    PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace python3 scripts/analysis/plot_weight_smoothing.py [1|2|3|4|both]'
"""
from __future__ import annotations
import re
from pathlib import Path

RLOG = Path("readable_outputs/figures/replay_logs")
OUT = Path("results/headweight_tailsmooth")
DSS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
K_RE = re.compile(r"^\s*(\w+):\s*K=([0-9.]+)")

HW = "mat_{ds}_hwts_%s_split.replay.txt"           # hwts arm template
CAL = "mat_{ds}_4way_calib_%s_split.replay.txt"    # older calib arm template

# arm label -> (color, filename template, prop). ONE color per arm, shared across figs.
REG = {
    "raw":                       ("#7f7f7f", CAL % "raw", "calib"),
    "raw + Laplace":             ("#8c564b", CAL % "succrawhead", "calib"),
    "raw + tail iso":            ("#9467bd", CAL % "isotail", "calib"),
    "head-beta + raw":           ("#d62728", HW % "onlbeta_rawtail", "calib"),
    "head-beta + Laplace":       ("#ff7f0e", HW % "onlbeta_succ_raw", "calib"),
    "head-beta + iso(Laplace)":  ("#1f77b4", HW % "onlbeta_succ_iso", "calib"),
    "head-beta + iso":           ("#2ca02c", CAL % "onl_beta_isotonic", "calib"),
    "raw + iso calib":           ("#9467bd", CAL % "isotail", "calib"),
    "raw + tail weight":         ("#17becf", HW % "rawhead_fix0075", "calib"),  # best w=0.075 (tag fix0075 = 0.075!)
    "raw + tail weight after Laplace": ("#e377c2", HW % "rawhead_succscale05", "calib"),  # best w=0.5
    "head-beta + iso calib":     ("#2ca02c", CAL % "onl_beta_isotonic", "calib"),   # alias of head-beta+iso
    "head-beta + tail weight":   ("#17becf", HW % "onlbeta_fixtw08", "calib"),      # best w=0.08
    "raw + tail weight after KT": ("#bcbd22", HW % "rawhead_ktscale03", "calib"),    # best w=0.3
    # offline-beta head x tail treatment (all fit on calib half)
    "beta + fixed scalar (swept)":   ("#17becf", HW % "betahead_fix008", "calib"),   # w=0.08
    "beta + isotonic (fitted)":      ("#2ca02c", HW % "betahead_isofit", "calib"),
    "beta + linear a*s+b (fitted)":  ("#d62728", HW % "betahead_linfit", "calib"),
    "beta + scaled rho*s (fitted)":  ("#9467bd", HW % "betahead_scaled", "calib"),
}

FIGS = {
    "fig2_smoothing_calib":  ("Smoothing + calibration (fig2): MAT by workload",
                              ["raw", "raw + Laplace", "head-beta + raw", "raw + tail iso"]),
    "fig3_smoothing_calib":  ("Smoothing + calibration (fig3): MAT by workload",
                              ["raw", "raw + Laplace", "raw + tail iso",
                               "head-beta + Laplace", "head-beta + iso"]),
    "fig4_smoothing_calib":  ("Smoothing + calibration (fig4): MAT by workload",
                              ["raw", "head-beta + iso", "head-beta + iso(Laplace)"]),
    "fig5_smoothing_calib":  ("Smoothing + calibration (fig5, raw head): MAT by workload",
                              ["raw", "raw + Laplace", "raw + iso calib", "raw + tail weight"]),
    "fig7_smoothing_calib":  ("Smoothing + calibration (fig7, raw head): MAT by workload",
                              ["raw", "raw + tail weight", "raw + tail weight after KT",
                               "raw + tail weight after Laplace"]),
    "fig6_smoothing_calib":  ("Smoothing + calibration (fig6, online-beta head): MAT by workload",
                              ["head-beta + raw", "head-beta + Laplace",
                               "head-beta + iso calib", "head-beta + tail weight"]),
    "fig8_beta_tail_fit":    ("Fitted tail calibration vs swept scalar (fig8, offline-beta head): MAT by workload",
                              ["beta + fixed scalar (swept)", "beta + isotonic (fitted)",
                               "beta + linear a*s+b (fitted)", "beta + scaled rho*s (fitted)"]),
}


def read_arm(tmpl, prop="calib"):
    out = {}
    for ds in DSS:
        p = RLOG / tmpl.format(ds=ds)
        v = None
        if p.exists():
            for ln in p.read_text().splitlines():
                m = K_RE.match(ln)
                if m and m.group(1) == prop:
                    v = float(m.group(2)); break
        out[ds] = v
    return out


def mean5(d):
    vs = [v for v in d.values() if v is not None]
    return sum(vs) / len(vs) if len(vs) == len(DSS) else None


def best_of(cands):
    scored = []
    for wl, tmpl in cands:
        ks = read_arm(tmpl); m = mean5(ks)
        if m is not None:
            scored.append((m, wl, ks))
    if not scored:
        return None
    scored.sort(reverse=True)
    m, wl, ks = scored[0]
    return wl, ks, m


def grouped_bar(ax, arms, title):
    import numpy as np
    x = np.arange(len(DSS)); n = len(arms); w = 0.8 / n
    for i, (label, color, ks) in enumerate(arms):
        vals = [ks[ds] if ks[ds] is not None else 0 for ds in DSS]
        off = (i - (n - 1) / 2) * w
        mv = [v for v in ks.values() if v is not None]
        lab = f"{label}  ({sum(mv)/len(mv):.3f})" if len(mv) == len(DSS) else f"{label}  (partial)"
        bars = ax.bar(x + off, vals, w, label=lab, color=color)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([d.upper() for d in DSS], fontsize=9)
    ax.set_ylabel("MAT (mean accepted tokens / round)")
    ax.set_title(title, fontsize=12)
    top = max((v for _, _, ks in arms for v in ks.values() if v), default=4.5)
    ax.set_ylim(0, top * 1.18)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.4)


def draw_reg(outname, title, labels):
    import matplotlib.pyplot as plt
    arms = [(lab, REG[lab][0], read_arm(REG[lab][1], REG[lab][2])) for lab in labels]
    fig, ax = plt.subplots(figsize=(11, 6))
    grouped_bar(ax, arms, title)
    fig.tight_layout(); fig.savefig(OUT / f"{outname}.png", dpi=150)
    print(f"{outname} ->", OUT / f"{outname}.png")
    for lab, _, ks in arms:
        m = mean5(ks)
        print(f"  {lab:26s} mean={m if m is None else round(m,3)}  " +
              " ".join(f"{ds}={ks[ds]}" for ds in DSS))


def fig1():
    import matplotlib.pyplot as plt
    head_c = [(f"u={u}", HW % a) for u, a in
              [("1.2", "hs12_rawtail"), ("1.4", "hs14_rawtail"), ("1.6", "hs16_rawtail"),
               ("1.8", "hs18_rawtail"), ("2.0", "hs20_rawtail"), ("2.5", "hs25_rawtail")]]
    # NOTE: fix005/fix0075 tags are w=0.05/0.075 (the finer-sweep driver's values);
    # they were once mislabeled 0.005/0.0075 — smooth interior peak at w=0.075.
    tail_c = [(f"w={w}", HW % a) for w, a in
              [("0.05", "rawhead_fix005"), ("0.075", "rawhead_fix0075"),
               ("0.10", "rawhead_fix010"), ("0.125", "rawhead_fix0125"),
               ("0.15", "rawhead_fix015"), ("0.20", "rawhead_fix020")]]
    ht_c = [(l, HW % a) for l, a in
            [("u1.4/w0.05", "h14w005"), ("u1.4/w0.075", "h14w0075"), ("u1.4/w0.10", "h14w010"),
             ("u1.4/w0.125", "twoscalar_h14w0125"), ("u1.6/w0.05", "h16w005"),
             ("u1.6/w0.075", "h16w0075"), ("u1.6/w0.10", "h16w010")]]
    raw_ks = read_arm(CAL % "raw")
    fam = [("raw compose", "#7f7f7f", ("", raw_ks, mean5(raw_ks))),
           ("head weight", "#1f77b4", best_of(head_c)),
           ("tail weight", "#ff7f0e", best_of(tail_c)),
           ("head+tail weight", "#2ca02c", best_of(ht_c))]
    fig, ax = plt.subplots(figsize=(11, 6))
    arms, ann = [], []
    for name, color, res in fam:
        if res is None:
            ann.append(f"{name}=incomplete"); continue
        wl, ks, m = res
        arms.append((name, color, ks))
        ann.append(f"{name}={wl}" if wl else f"{name}")
    grouped_bar(ax, arms, "Weight analysis (fig1): MAT by workload  (each arm at its best scalar)")
    ax.text(0.005, 0.985, "best scalar —  " + "  |  ".join(ann), transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.tight_layout(); fig.savefig(OUT / "fig1_weight_analysis.png", dpi=150)
    print("fig1 ->", OUT / "fig1_weight_analysis.png")
    for name, _, res in fam:
        if res:
            wl, ks, m = res
            print(f"  {name:18s} best {wl:14s} mean={m:.3f}  " +
                  " ".join(f"{ds}={ks[ds]}" for ds in DSS))


if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "1"):
        fig1()
    if which in ("both", "2"):
        draw_reg("fig2_smoothing_calib", *FIGS["fig2_smoothing_calib"])
    if which in ("both", "3"):
        draw_reg("fig3_smoothing_calib", *FIGS["fig3_smoothing_calib"])
    if which in ("both", "4"):
        draw_reg("fig4_smoothing_calib", *FIGS["fig4_smoothing_calib"])
    if which in ("both", "5"):
        draw_reg("fig5_smoothing_calib", *FIGS["fig5_smoothing_calib"])
    if which in ("both", "6"):
        draw_reg("fig6_smoothing_calib", *FIGS["fig6_smoothing_calib"])
    if which in ("both", "7"):
        draw_reg("fig7_smoothing_calib", *FIGS["fig7_smoothing_calib"])
    if which in ("both", "8"):
        draw_reg("fig8_beta_tail_fit", *FIGS["fig8_beta_tail_fit"])
