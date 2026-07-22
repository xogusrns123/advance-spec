#!/usr/bin/env python3
r"""fig12: head raw fixed. CONTROL = tail weight w*(raw arctic score);
EXPERIMENT = Laplace-smooth the tail first, THEN weight = w*(Laplace score).
Big-unit w sweep, mean over 5 workloads. Tests whether pre-Laplace helps the
tail-weight (it does not — the +1 numerator floor over-prices dead edges).
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
    done = [v for v in vals if v is not None]
    return (sum(done) / 5) if len(done) == 5 else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # experiment: Laplace -> weight  (w on Laplace-score scale, O(1))
    exp = [(0.10, "01"), (0.25, "025"), (0.35, "035"), (0.40, "04"), (0.45, "045"),
           (0.50, "05"), (0.55, "055"), (0.60, "06e"), (0.65, "065"), (0.75, "075"),
           (1.00, "10"), (1.50, "15"), (2.00, "20")]
    E = [(w, mean_of(f"mat_{{ds}}_hwts_rawhead_succscale{t}_split.replay.txt")) for w, t in exp]
    E = [(w, m) for w, m in E if m is not None]
    # control: raw-score weight  (w on raw-arctic scale, O(0.1))
    ctl = [(0.05, "fix005"), (0.06, "fix006"), (0.075, "fix0075"), (0.085, "fix0085"),
           (0.10, "fix010"), (0.125, "fix0125"), (0.15, "fix015"), (0.20, "fix020")]
    C = [(w, mean_of(f"mat_{{ds}}_hwts_rawhead_{t}_split.replay.txt")) for w, t in ctl]
    C = [(w, m) for w, m in C if m is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # panel: raw magnitude differs, so separate x-axes but shared y
    ce, ee = max(C, key=lambda t: t[1]), max(E, key=lambda t: t[1])
    ax1.plot([w for w, _ in C], [m for _, m in C], "-o", color="#17becf", lw=2, ms=6)
    ax1.plot([ce[0]], [ce[1]], "*", ms=20, color="#d62728", zorder=5,
             label=f"peak {ce[1]:.3f} @ w={ce[0]}")
    for w, m in C:
        ax1.annotate(f"{m:.3f}", (w, m), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    ax1.set_title("CONTROL: raw head + tail weight\n w · (raw arctic score)")
    ax1.set_xlabel("weight w  (raw-score scale)"); ax1.set_ylabel("MAT (5-wl mean)")
    ax1.legend(fontsize=9, loc="lower center"); ax1.grid(True, ls=":", alpha=0.4)

    ax2.plot([w for w, _ in E], [m for _, m in E], "-o", color="#8c564b", lw=2, ms=6)
    ax2.plot([ee[0]], [ee[1]], "*", ms=20, color="#d62728", zorder=5,
             label=f"peak {ee[1]:.3f} @ w={ee[0]}")
    for w, m in E:
        ax2.annotate(f"{m:.3f}", (w, m), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    ax2.axhline(ce[1], ls="--", color="#17becf", lw=1.3, label=f"control peak {ce[1]:.3f}")
    ax2.set_title("EXPERIMENT: raw head + Laplace → weight\n w · (Laplace (k+1)/(n+2) score)")
    ax2.set_xlabel("weight w  (Laplace-score scale)")
    ax2.legend(fontsize=9, loc="lower center"); ax2.grid(True, ls=":", alpha=0.4)

    fig.suptitle(f"fig12: pre-Laplace HURTS the tail weight — exp peak {ee[1]:.3f} < control {ce[1]:.3f} "
                 f"({ee[1]-ce[1]:+.3f})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig12_laplace_then_weight.png", dpi=150)
    print("fig12 ->", OUT / "fig12_laplace_then_weight.png")
    print("control:", C); print("experiment:", E)


if __name__ == "__main__":
    main()
