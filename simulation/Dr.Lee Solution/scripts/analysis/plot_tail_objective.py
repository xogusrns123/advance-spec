#!/usr/bin/env python3
r"""fig9: the two objectives on one w-axis. LEFT y = MSE(w) = mean (w*score - realized)^2,
the quantity the L2 tail fits (scaled/linear/isotonic) minimize; RIGHT y = MAT(w),
the replay speed we actually want. If argmin MSE != argmax MAT, that gap IS why
fitting misses the optimum. One panel per workload + an aggregate. MAT(w) taken
from the online-beta fixed-tail sweep (mat_{ds}_hwts_onlbeta_fixtw{wt}). English labels.

  docker exec sglang-bench bash -lc 'cd /workspace/simulation/Dr.Lee\ Solution && \
    PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace python3 scripts/analysis/plot_tail_objective.py'
"""
from __future__ import annotations
import json
import re
from pathlib import Path

RLOG = Path("readable_outputs/figures/replay_logs")
TOBJ = Path("results/headweight_tailsmooth/tail_objective")
OUT = Path("results/headweight_tailsmooth")
DSS = ["specbench", "bfcl", "swebench", "spider", "tau2"]
K_RE = re.compile(r"^\s*calib:\s*K=([0-9.]+)")

# online-beta fixed-tail sweep: wt-tag -> w value (from the two launches)
BETA_MATW = {"0075": 0.0075, "0125": 0.0125, "02": 0.02, "035": 0.035, "06": 0.06,
             "08": 0.08, "10": 0.10, "125": 0.125, "15": 0.15, "20": 0.20}


def mat_w(ds):
    """MAT(w) for online-beta + fixed tail, this workload."""
    out = {}
    for tag, w in BETA_MATW.items():
        p = RLOG / f"mat_{ds}_hwts_onlbeta_fixtw{tag}_split.replay.txt"
        if p.exists():
            for ln in p.read_text().splitlines():
                m = K_RE.match(ln)
                if m:
                    out[w] = float(m.group(1)); break
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    objs = {ds: json.load(open(TOBJ / f"{ds}.json")) for ds in DSS if (TOBJ / f"{ds}.json").exists()}
    if not objs:
        print("no tail_objective json yet"); return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, ds in enumerate([d for d in DSS if d in objs]):
        ax = axes[i]; o = objs[ds]
        mse = {float(k): v for k, v in o["mse_curve"].items()}
        ws = sorted(mse)
        ax.plot(ws, [mse[w] for w in ws], color="#d62728", lw=2, label="MSE(w) = mean (w·score − accept)²")
        wmse = o["w_mse_min_gridded"]; rho = o["rho_scaled_meanmatch"]
        ax.axvline(wmse, ls="--", color="#d62728", lw=1)
        ax.axvline(rho, ls=":", color="#8c564b", lw=1.2)
        ax.set_xlabel("tail scalar w"); ax.set_ylabel("MSE (tail-estimation error)", color="#d62728")
        ax.tick_params(axis="y", labelcolor="#d62728")
        ax.set_title(f"{ds.upper()}", fontsize=11)

        ax2 = ax.twinx()
        matd = mat_w(ds)
        if matd:
            mw = sorted(matd)
            ax2.plot(mw, [matd[w] for w in mw], color="#1f77b4", lw=2, marker="o",
                     ms=3, label="MAT(w) (replay, online-beta head)")
            wmat = max(matd, key=matd.get)
            ax2.axvline(wmat, ls="--", color="#1f77b4", lw=1)
            ax2.set_ylabel("MAT (accepted tok / round)", color="#1f77b4")
            ax2.tick_params(axis="y", labelcolor="#1f77b4")
            ax.text(0.97, 0.05,
                    f"argmin MSE  w≈{wmse:.2f}  (ρ={rho:.2f})\nargmax MAT  w≈{wmat:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    # legend in the last (6th) empty axis
    axes[-1].axis("off")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#d62728", lw=2, label="MSE(w): what scaled/linear/iso minimize"),
        Line2D([0], [0], color="#d62728", ls="--", label="argmin MSE  (= fitted slope w*)"),
        Line2D([0], [0], color="#8c564b", ls=":", label="ρ = E[accept]/E[score]  (scaled fit)"),
        Line2D([0], [0], color="#1f77b4", lw=2, marker="o", label="MAT(w): what we want (online-beta)"),
        Line2D([0], [0], color="#1f77b4", ls="--", label="argmax MAT  (swept optimum)"),
    ]
    axes[-1].legend(handles=handles, loc="center", fontsize=10,
                    title="fig9: fitting minimizes MSE, not MAT")
    fig.suptitle("Why fitted tail calibration misses the MAT optimum: "
                 "MSE-min (fit target) vs MAT-max (goal) sit at different w", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig9_tail_objective_gap.png", dpi=140)
    print("fig9 ->", OUT / "fig9_tail_objective_gap.png")
    for ds, o in objs.items():
        matd = mat_w(ds)
        wmat = max(matd, key=matd.get) if matd else None
        print(f"  {ds:10s} argmin-MSE w≈{o['w_mse_min_gridded']:.3f} (ρ={o['rho_scaled_meanmatch']:.3f})"
              f"  argmax-MAT w≈{wmat}  E[score]={o['mean_score']:.1f} E[realized]={o['mean_realized']:.2f}")


if __name__ == "__main__":
    main()
