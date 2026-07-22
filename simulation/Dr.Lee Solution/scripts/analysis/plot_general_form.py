#!/usr/bin/env python3
r"""fig13 (results/headweight_tailsmooth/fig13_general_form.png). Same layout as
fig12 (control | experiment, both MAT-vs-w). Head raw fixed.
  CONTROL   = raw head + tail weight        : w · (raw arctic score)
  EXPERIMENT= general form at its optimum   : w · Σ Π (k+a)/(n+b), a=0, b=0.125
              (the a>0 floor and other b are dominated — see fig13b / ablation)
Unlike Laplace (fig12), the general form's a=0 deflation BEATS the flat scalar.
"""
from __future__ import annotations
import re
from pathlib import Path

RLOG = Path("readable_outputs/figures/replay_logs")
OUT = Path("results/headweight_tailsmooth")
DSS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
K_RE = re.compile(r"^\s*calib:\s*K=([0-9.]+)")


def mean_of(fname):
    vals = []
    for ds in DSS:
        p = RLOG / fname.format(ds=ds)
        k = None
        if p.exists():
            for ln in p.read_text().splitlines():
                m = K_RE.match(ln)
                if m:
                    k = float(m.group(1)); break
        vals.append(k)
    d = [v for v in vals if v is not None]
    return (sum(d) / 5) if len(d) == 5 else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # CONTROL: raw head + tail weight, w on raw-score scale
    ctl = [(0.05, "mat_{ds}_hwts_rawhead_fix005_split.replay.txt"),
           (0.06, "mat_{ds}_hwts_rawhead_fix006_split.replay.txt"),
           (0.075, "mat_{ds}_hwts_rawhead_fix0075_split.replay.txt"),
           (0.085, "mat_{ds}_hwts_rawhead_fix0085_split.replay.txt"),
           (0.10, "mat_{ds}_hwts_rawhead_fix010_split.replay.txt"),
           (0.125, "mat_{ds}_hwts_rawhead_fix0125_split.replay.txt"),
           (0.15, "mat_{ds}_hwts_rawhead_fix015_split.replay.txt"),
           (0.20, "mat_{ds}_hwts_rawhead_fix020_split.replay.txt")]
    C = [(w, mean_of(f)) for w, f in ctl]
    C = [(w, m) for w, m in C if m is not None]

    # EXPERIMENT: general form a=0, b=0.125 (deflation), w-sweep
    exp = [(0.08, "w008"), (0.10, "w010"), (0.12, "w012"), (0.15, "w015"),
           (0.17, "w017"), (0.20, "w020"), (0.22, "w022")]
    E = [(w, mean_of(f"mat_{{ds}}_hwts_rawhead_gbscl2_b0125_{t}_split.replay.txt")) for w, t in exp]
    E = [(w, m) for w, m in E if m is not None]

    ce = max(C, key=lambda t: t[1]); ee = max(E, key=lambda t: t[1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax1.plot([w for w, _ in C], [m for _, m in C], "-o", color="#17becf", lw=2, ms=6)
    ax1.plot([ce[0]], [ce[1]], "*", ms=20, color="#d62728", zorder=5, label=f"peak {ce[1]:.3f} @ w={ce[0]}")
    for w, m in C:
        ax1.annotate(f"{m:.3f}", (w, m), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    ax1.set_title("CONTROL: raw head + tail weight\n w · (raw arctic score)")
    ax1.set_xlabel("weight w"); ax1.set_ylabel("MAT (5-wl mean)")
    ax1.legend(fontsize=9, loc="lower center"); ax1.grid(True, ls=":", alpha=0.4)

    ax2.plot([w for w, _ in E], [m for _, m in E], "-o", color="#2ca02c", lw=2, ms=6)
    ax2.plot([ee[0]], [ee[1]], "*", ms=20, color="#d62728", zorder=5, label=f"peak {ee[1]:.3f} @ w={ee[0]}")
    for w, m in E:
        ax2.annotate(f"{m:.3f}", (w, m), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    ax2.axhline(ce[1], ls="--", color="#17becf", lw=1.3, label=f"control peak {ce[1]:.3f}")
    ax2.set_title("EXPERIMENT: raw head + general form\n w · Σ Π (k+a)/(n+b),  a=0, b=0.125")
    ax2.set_xlabel("weight w")
    ax2.legend(fontsize=9, loc="lower center"); ax2.grid(True, ls=":", alpha=0.4)

    d = ee[1] - ce[1]
    verb = "BREAKS" if d > 0 else "misses"
    fig.suptitle(f"fig13: general tail form {verb} the scalar ceiling — exp peak {ee[1]:.3f} "
                 f"vs control {ce[1]:.3f} ({d:+.3f})   [optimum is a=0, small b; any floor a>0 collapses]",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig13_general_form.png", dpi=150)
    print("fig13 ->", OUT / "fig13_general_form.png")
    print("control:", C); print("experiment (a=0,b=0.125):", E)


if __name__ == "__main__":
    main()
