#!/usr/bin/env python3
"""SMOOTHING REFERENCE (dfprior): DFlash-informed Beta prior tail rescore
q=(k + q0*s)/(n + s), q0=min(1,u*conf) from DFlash head confidence, strength s.
Sweeps prior strength s per workload (raw head, TEST-half convlabel split). The
a/b-ratio analog: q0 is the DFlash-driven prior MEAN, s the prior STRENGTH.

Companion to fig11_ab_ablation (uniform-prior genbeta). Shows DFlash-as-prior is
workload-dependent: strong prior (s>=8) helps DFlash-dominant workloads
(specbench), weak prior (s<=1) is best where suffix carries (swebench/tau2).

  docker exec sglang-bench bash -lc 'cd /workspace/simulation/Dr.Lee\ Solution && \
    PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace python3 scripts/analysis/plot_dfprior_reference.py'
"""
from __future__ import annotations
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "plot"))
import plot_method_families as M

S = [("s025", 0.25), ("s05", 0.5), ("s1", 1.0), ("s2", 2.0), ("s4", 4.0),
     ("s8", 8.0), ("s16", 16.0), ("s32", 32.0)]
DS_LABEL = {"specbench": "Spec-Bench", "bfcl": "BFCL v4",
            "swebench": "SWE-bench", "spider": "Spider2-DBT", "tau2": "τ²-bench"}
COL = {"specbench": "#4C78A8", "bfcl": "#F58518", "swebench": "#54A24B",
       "spider": "#9467BD", "tau2": "#E45756"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K, _, _ = M.load_base()
    fig, ax = plt.subplots(figsize=(11, 6))
    for ds in M.DS:
        xs, ys = [], []
        for tag, s in S:
            d = M.arm_data(ds, f"rawhead_dfprior_{tag}")
            if d:
                xs.append(s); ys.append(d[0])
        if not xs:
            continue
        ax.plot(xs, ys, "-o", color=COL[ds], lw=2, ms=5, label=DS_LABEL[ds])
        bi = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot([xs[bi]], [ys[bi]], "*", color=COL[ds], ms=16, zorder=5)
        ax.annotate(f"s*={xs[bi]:g}\n{ys[bi]:.2f}", (xs[bi], ys[bi]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8,
                    color=COL[ds])
        cr = K[ds].get("compose_raw", (0, 0))[0]         # raw compose floor
        ax.axhline(cr, ls=":", color=COL[ds], lw=0.8, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([s for _, s in S])
    ax.set_xticklabels([f"{s:g}" for _, s in S])
    ax.set_xlabel("DFlash-prior strength  s   (pseudo-count in (k+q₀·s)/(n+s), q₀=min(1,u·conf))")
    ax.set_ylabel("MAT (accepted tok / round)")
    ax.set_title("Smoothing reference — DFlash-informed Beta prior (raw head)\n"
                 "★ = per-workload best strength; dotted = compose(raw) floor")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=9, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = M.OUT / "MAT_smoothing_dfprior_reference.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    for ds in M.DS:
        vals = [(s, M.arm_data(ds, f"rawhead_dfprior_{t}")) for t, s in S]
        vals = [(s, d[0]) for s, d in vals if d]
        if vals:
            b = max(vals, key=lambda x: x[1])
            print(f"  {ds}: best s={b[0]:g} MAT={b[1]:.2f}")


if __name__ == "__main__":
    main()
