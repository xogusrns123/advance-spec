#!/usr/bin/env python3
"""Correction diagnostics for the three deployable methodologies
(readable_outputs/figures/correction(deployable)/{online,twoscalar,succession}/):

per workload x side (head/tail) x stage:
  *_raw.png        x = raw signal (head conf / tail arctic score), bars =
                   empirical conditional accept rate (head) or mean accept
                   length (tail) per bin; RED line = the method's fitted /
                   derived transform.
  *_corrected.png  x = corrected signal, bars = empirical values (should hug
                   the dashed y=x reference when the correction is right).

Data: deployable-split TEST halves — head pairs (conf, match) accept-
conditioned from the capture records; tail per-draft (raw score, succession
score, realized accept) from /workspace/tmp/peredge_test_{ds}.npz (k=0 probes).

  python3 scripts/plot/plot_correction_diag.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from replay_extension import _ad, _fit_logistic, split_parity

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "readable_outputs" / "figures" / "correction(deployable)"
REC = {"specbench": "results/perpos_specbench_alleval/specbench_4way.jsonl",
       "bfcl": "results/perpos_bfcl_alleval/bfcl_4way.jsonl",
       "swebench": "results/perpos_swebench_alleval/swebench_4way.jsonl",
       "spider": "results/perpos_spider_alleval/spider_4way.jsonl",
       "tau2": "results/perpos_tau2_alleval/tau2_4way.jsonl"}
WL_NAME = {"specbench": "Spec-Bench", "bfcl": "BFCL v4", "swebench": "SWE-bench Verified",
           "spider": "Spider2-DBT", "tau2": "tau2-bench"}
NOTE = "deployable split, TEST half"
C_HEAD = "#4C78A8"   # DFlash blue (MAT figure convention)
C_TAIL = "#F58518"   # Suffix orange
RED = "#d62728"

U, W = 1.4, 0.125            # two-scalar constants (calib-half derived)


def head_pairs(ds):
    """accept-conditioned (conf, match) on the TEST half, stream order."""
    rec = BASE.parent / REC[ds]
    rec = Path("/workspace/simulation/Dr.Lee Solution") / REC[ds]
    traces = json.load(open(str(rec).replace(".jsonl", ".traces.json")))
    ev = {t["rid"]: t for t in traces["eval_traces"]}
    _, par = split_parity(ev, "convlabel")
    C, Y, V = [], [], []
    for l in open(rec):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if par.get(r["rid"], 0) != 1:
            continue
        cf, mt = r["dflash_conf"], r["dflash_match"]
        cv = r.get("conv", r["rid"])
        ad = _ad(mt)
        for d in range(min(ad + 1, len(cf))):
            C.append(float(cf[d])); Y.append(int(mt[d])); V.append(cv)
    return np.asarray(C), np.asarray(Y, float), np.asarray(V)


def tail_pairs(ds):
    z = np.load(f"/workspace/tmp/peredge_test_{ds}.npz")
    s, A, Ss = z["D_s"], z["D_A"].astype(float), z["D_S_succ"]
    m = s > 0
    return s[m], A[m], Ss[m]


def _render(ax, positions, groups, width, color, style):
    """style='err': per-bin mean +- 1 std errorbar (lower clipped at 0).
    style='bar': per-bin mean histogram bar. Point/bar height = the mean,
    which is exactly what the correction targets."""
    means = np.asarray([np.mean(g) for g in groups], float)
    if style == "bar":
        ax.bar(positions, means, width=width, color=color, edgecolor="k",
               linewidth=0.4, zorder=2, label="empirical mean")
    else:
        stds = np.asarray([np.std(g) for g in groups], float)
        lo = np.minimum(stds, means)          # clip mean-std at 0
        ax.errorbar(positions, means, yerr=[lo, stds], fmt="o", ms=6,
                    color=color, ecolor=color, elinewidth=1.6, capsize=4,
                    markeredgecolor="k", markeredgewidth=0.5, zorder=2,
                    label="empirical mean ± 1 std")


def head_conv_groups(edges, xs, ys, convs, min_pairs=10, min_convs=5):
    """per bin: distribution over CONVERSATIONS of the conv accept rate
    (y is binary per pair, so the box-able unit is the conv-level rate)."""
    pos, groups = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (xs >= lo) & (xs < hi)
        if m.sum() < 100:
            continue
        agg = {}
        for c, y in zip(convs[m], ys[m]):
            a = agg.setdefault(c, [0, 0])
            a[0] += y; a[1] += 1
        rates = [s_ / n_ for s_, n_ in agg.values() if n_ >= min_pairs]
        if len(rates) < min_convs:
            continue
        pos.append((lo + hi) / 2); groups.append(rates)
    return pos, groups


def _diag(ax, hi):
    ax.plot([0, hi], [0, hi], ls="--", color="#888888", lw=1.4, zorder=3,
            label="y = x (target)")


def _ylab(base, style):
    return f"{base} ({'mean' if style == 'bar' else 'mean ± std'})"


def head_figs(mdir, ds, C, Y, convs, transform, tname, identity=False,
              style="err", fit=True):
    edges = np.linspace(0, 1, 13)
    sfx = ("_bar" if style == "bar" else "")
    # ---- raw ----
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    pos, groups = head_conv_groups(edges, C, Y, convs)
    _render(ax, pos, groups, (edges[1] - edges[0]) * 0.85, C_HEAD, style)
    if fit:
        xx = np.linspace(0.01, 0.99, 200)
        ax.plot(xx, [transform(v) for v in xx], color=RED, lw=2.2, zorder=4,
                label=f"correction: {tname}")
    _diag(ax, 1.0)
    ax.set_xlabel("raw DFlash probability (conf)", fontsize=11)
    ax.set_ylabel(_ylab("conditional accept rate", style), fontsize=11)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{WL_NAME[ds]} — HEAD: raw conf vs accept rate\n{NOTE}", fontsize=12)
    ax.legend(fontsize=9.5, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fsfx = "" if fit else "_nofit"
    fig.tight_layout(); fig.savefig(mdir / f"head_{ds}_raw{fsfx}{sfx}.png", dpi=140)
    plt.close(fig)
    if not fit:
        return                     # nofit only produces the raw panel
    # ---- corrected ----
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    Cc = np.asarray([transform(v) for v in C])
    pos, groups = head_conv_groups(edges, Cc, Y, convs)
    _render(ax, pos, groups, (edges[1] - edges[0]) * 0.85, C_HEAD, style)
    _diag(ax, 1.0)
    ax.set_xlabel(f"corrected probability ({tname})", fontsize=11)
    ax.set_ylabel(_ylab("conditional accept rate", style), fontsize=11)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    extra = "  (head kept RAW — no correction)" if identity else ""
    ax.set_title(f"{WL_NAME[ds]} — HEAD: corrected vs accept rate{extra}\n{NOTE}",
                 fontsize=12)
    ax.legend(fontsize=9.5, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(mdir / f"head_{ds}_corrected{sfx}.png", dpi=140)
    plt.close(fig)


def tail_figs(mdir, ds, s, A, corrected, tname, redline, style="err", fit=True):
    """NUMERIC equal-scale axes so the gray dashed y=x diagonal (the alignment
    target) is a true 45-degree line."""
    sfx = ("_bar" if style == "bar" else "")
    edges = np.arange(0, 32.01, 2.0)          # UNIFORM bins
    # ---- raw ----
    fig, ax = plt.subplots(figsize=(6.9, 6.9))
    cs, rl, groups = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum() < 60:
            continue
        cs.append((lo + hi) / 2); groups.append(A[m])
        if callable(redline):
            rl.append(np.mean([redline(v) for v in s[m]]))
        else:
            rl.append(redline[1][m].mean())
    _render(ax, cs, groups, 2.0 * 0.85, C_TAIL, style)
    if fit:
        ax.plot(cs, rl, color=RED, lw=2.2, zorder=4, label=f"correction: {tname}")
    _diag(ax, 32)
    ax.set_xlabel("raw arctic score", fontsize=11)
    ax.set_ylabel(_ylab("realized accept length", style), fontsize=11)
    ax.set_xlim(0, 32.5); ax.set_ylim(0, 32.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{WL_NAME[ds]} — TAIL: raw score vs accept length\n{NOTE}", fontsize=12)
    ax.legend(fontsize=9.5, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fsfx = "" if fit else "_nofit"
    fig.tight_layout(); fig.savefig(mdir / f"tail_{ds}_raw{fsfx}{sfx}.png", dpi=140)
    plt.close(fig)
    if not fit:
        return
    # ---- corrected ----
    cmax = float(np.clip(np.ceil(np.percentile(corrected, 99.9) * 2) / 2 + 0.5,
                         3.0, 12.0))          # data-driven square range
    cedges = np.arange(0, cmax + 0.01, 0.5)   # UNIFORM bins
    fig, ax = plt.subplots(figsize=(6.9, 6.9))
    cs, groups = [], []
    for lo, hi in zip(cedges[:-1], cedges[1:]):
        m = (corrected >= lo) & (corrected < hi)
        if m.sum() < 60:
            continue
        cs.append((lo + hi) / 2); groups.append(A[m])
    _render(ax, cs, groups, 0.5 * 0.85, C_TAIL, style)
    _diag(ax, cmax)
    ax.set_xlabel(f"corrected score ({tname})", fontsize=11)
    ax.set_ylabel(_ylab("realized accept length", style), fontsize=11)
    ax.set_xlim(0, cmax * 1.02); ax.set_ylim(0, cmax * 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{WL_NAME[ds]} — TAIL: corrected score vs accept length\n{NOTE}",
                 fontsize=12)
    ax.legend(fontsize=9.5, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(mdir / f"tail_{ds}_corrected{sfx}.png", dpi=140)
    plt.close(fig)


def main():
    for method in ("online", "twoscalar", "succession"):
        (OUT / method).mkdir(parents=True, exist_ok=True)
    for ds in REC:
        C, Y, V = head_pairs(ds)
        s, A, Ss = tail_pairs(ds)
        # ---- online: converged window fits on the test-half stream ----------
        from sklearn.isotonic import IsotonicRegression
        lg = _fit_logistic(C.tolist(), Y.astype(int).tolist())
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(s, A)
        iso = lambda v, _ir=ir: float(_ir.predict([max(0.0, float(v))])[0])  # noqa: E731
        SPECS = [
            ("online", lg, "online logistic (converged)", False,
             np.asarray(ir.predict(s), float), "online isotonic (converged)", iso),
            ("twoscalar", (lambda c: min(1.0, U * float(c))), f"min(1, {U}*conf)", False,
             W * s, f"{W}*score", (lambda v: W * float(v))),
            ("succession", (lambda c: float(c)), "identity (raw head)", True,
             Ss, "succession rescore", ("binned", Ss)),
        ]
        # each methodology: for style in {errorbar, bar}, draw the fitted raw+
        # corrected panels AND the fit-free raw panels (head & tail).
        for name, htf, htn, hid, tcorr, ttn, trl in SPECS:
            d = OUT / name
            for style in ("err", "bar"):
                head_figs(d, ds, C, Y, V, htf, htn, identity=hid, style=style, fit=True)
                tail_figs(d, ds, s, A, tcorr, ttn, trl, style=style, fit=True)
                head_figs(d, ds, C, Y, V, htf, htn, identity=hid, style=style, fit=False)
                tail_figs(d, ds, s, A, tcorr, ttn, trl, style=style, fit=False)
        print(f"[{ds}] done  (head pairs {len(C):,}, tail drafts {len(s):,})", flush=True)
    print("ALL FIGURES ->", OUT)


if __name__ == "__main__":
    main()
