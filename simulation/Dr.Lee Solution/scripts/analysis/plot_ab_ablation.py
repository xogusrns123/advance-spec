#!/usr/bin/env python3
r"""fig11: genbeta (k+a)/(n+b) a,b ablation (raw head). Two one-factor-at-a-time
panels using representative + optimal points only (not the full grid):
  A) strength b ablation at fixed prior mean m=a/b=0.25
  B) prior-mean m ablation at fixed strength b=2
with reference lines: raw+raw floor, succession (=a1,b2, uniform prior),
raw+tail-weight (the fixed-scalar target). Best genbeta cell starred; the ~3.71
optimum ridge (several (a,b) tie) noted. English labels.

  docker exec sglang-bench bash -lc 'cd /workspace/simulation/Dr.Lee\ Solution && \
    PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace python3 scripts/analysis/plot_ab_ablation.py'
"""
from __future__ import annotations
import re
from pathlib import Path

RLOG = Path("readable_outputs/figures/replay_logs")
OUT = Path("results/headweight_tailsmooth")
DSS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
K_RE = re.compile(r"^\s*calib:\s*K=([0-9.]+)")
SUCC, TAILW, RAW = 3.638, 3.862, 3.208     # raw-head reference MATs


def mean_ab(at, bt):
    vals = []
    for ds in DSS:
        p = RLOG / f"mat_{ds}_hwts_rawhead_genbeta_a{at}_b{bt}_split.replay.txt"
        k = None
        if p.exists():
            for ln in p.read_text().splitlines():
                m = K_RE.match(ln)
                if m:
                    k = float(m.group(1)); break
        vals.append(k)
    done = [v for v in vals if v is not None]
    return (sum(done) / 5) if len(done) == 5 else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Panel A: strength ablation at mean m=0.25  (a = 0.25*b)
    stren = [(1.0, "025", "1"), (1.5, "0375", "15"), (2.0, "05", "2"),
             (3.0, "075", "3"), (8.0, "2", "8")]
    A = [(b, mean_ab(at, bt)) for (b, at, bt) in stren]
    A = [(b, m) for b, m in A if m is not None]

    # Panel B: prior-mean ablation at strength b=2  (a = m*2)
    meanln = [(0.1, "02", "2"), (0.15, "03", "2"), (0.2, "04", "2"),
              (0.25, "05", "2"), (0.3, "06", "2"), (0.35, "07", "2"), (0.4, "08", "2"),
              (0.5, "1", "2"), (1.0, "2", "2"), (2.0, "4", "2")]
    B = [(m, mean_ab(at, bt)) for (m, at, bt) in meanln]
    B = [(m, v) for m, v in B if v is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def refs(ax):
        ax.axhline(TAILW, ls="--", color="#17becf", lw=1.5, label=f"raw+tail-weight (target) {TAILW:.3f}")
        ax.axhline(SUCC, ls="--", color="#8c564b", lw=1.2, label=f"succession/Laplace {SUCC:.3f}")
        ax.axhline(RAW, ls=":", color="#7f7f7f", lw=1.2, label=f"raw+raw (no prior) {RAW:.3f}")

    # A
    ax1.plot([b for b, _ in A], [m for _, m in A], "-o", color="#1f77b4", lw=2, ms=6)
    if A:
        bstar, mstar = max(A, key=lambda t: t[1])
        ax1.plot([bstar], [mstar], "*", color="#d62728", ms=18, zorder=5,
                 label=f"best  b={bstar:g} (MAT {mstar:.3f})")
        for b, m in A:
            ax1.annotate(f"{m:.3f}", (b, m), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    refs(ax1)
    ax1.set_xlabel("prior strength b  (a=0.25·b)")
    ax1.set_ylabel("MAT (accepted tok / round)")
    ax1.set_title("A) strength ablation  (prior mean fixed = 0.25)")
    ax1.legend(fontsize=7.5, loc="lower center"); ax1.grid(True, ls=":", alpha=0.4)

    # B
    ax2.plot([m for m, _ in B], [v for _, v in B], "-o", color="#2ca02c", lw=2, ms=6)
    if B:
        mstar, vstar = max(B, key=lambda t: t[1])
        ax2.plot([mstar], [vstar], "*", color="#d62728", ms=18, zorder=5,
                 label=f"best  m={mstar:g} (MAT {vstar:.3f})")
        for m, v in B:
            ax2.annotate(f"{v:.3f}", (m, v), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    refs(ax2)
    ax2.set_xlabel("prior mean m = a/b  (strength fixed b=2)")
    ax2.set_title("B) prior-mean ablation  (strength fixed = 2)")
    ax2.legend(fontsize=7.5, loc="lower center"); ax2.grid(True, ls=":", alpha=0.4)

    # ridge note: top genbeta cells
    ridge = sorted(
        [(f"a={a} b={b}", mean_ab(at, bt))
         for (a, at, b, bt) in [(0.5, "05", 2, "2"), (0.75, "075", 3, "3"),
                                (4, "4", 8, "8"), (0.25, "025", 1, "1"), (2, "2", 8, "8")]],
        key=lambda t: -(t[1] or 0))
    txt = "genbeta ridge (top cells):  " + " | ".join(f"{n} {v:.3f}" for n, v in ridge if v)
    fig.suptitle("fig11: genbeta (k+a)/(n+b) ablation, raw head — optimum ~3.71 vs tail-weight 3.862",
                 fontsize=13)
    fig.text(0.5, 0.005, txt + f"   (gap to tail-weight: {(max(v for _,v in ridge if v)-TAILW):+.3f})",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT / "fig11_ab_ablation.png", dpi=150)
    print("fig11 ->", OUT / "fig11_ab_ablation.png")
    print("A strength@m0.25:", A)
    print("B mean@b2:", B)
    print("ridge:", ridge)


if __name__ == "__main__":
    main()
