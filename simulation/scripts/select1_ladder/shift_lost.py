"""reliability_lost format, one column PER SHIFT: manually push the suffix 0.5-atom
score up by 0.1 (no calibration) and watch the correct-but-LOST bars change.

Per cell: rows = [model (EAGLE3/MTP), Suffix], cols = shifts {+0.0..+0.5}. Same bars as
reliability_lost (total gray / won colored / LOST orange-hatched). Selection = raw
argmax(eagle_p, suffix_score), suffix_score = suffix_p + shift for the 0.5 atom (clip 1).
Suffix row is binned by the SHIFTED score (so the 0.5 mass bar slides right with the
shift); model row binned by raw prob (its score is unchanged, only its LOST changes as
suffix steals picks).

Run (host): python3 simulation/scripts/select1_ladder/shift_lost.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_count import ROOT, CENT, load_chains, loopy_rids  # noqa: E402
from reliability_lost import bin_counts, LOST_COLOR  # noqa: E402

OUTDIR = Path("simulation/results/calib_reliability/figures")
SHIFTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
CELLS = {
    "qwen3_14b_2way": ("qwen3_14b_ar", "EAGLE3", "#1f77b4", "eagle_token", "eagle_p"),
    "qwen35_27b_2way": ("qwen35_27b_ar", "MTP", "#1f77b4", "eagle_token", "eagle_p"),
}
SUF = "#d62728"


def collect(chains, mtok, mpk):
    ep, ec, sp, sc = [], [], [], []
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            mt, mpv = r.get(mtok), r.get(mpk)
            st, spv = r.get("suffix_token"), r.get("suffix_p")
            e_av = mt is not None and mpv is not None
            s_av = st is not None and spv is not None
            if not (e_av or s_av):
                continue
            ep.append(float(mpv) if e_av else np.nan)
            ec.append(1 if (e_av and gt is not None and mt == gt) else 0)
            sp.append(float(spv) if s_av else np.nan)
            sc.append(1 if (s_av and gt is not None and st == gt) else 0)
            hit = ((e_av and mt == gt) or (s_av and st == gt)) if gt is not None else False
            if gt is not None and not hit:
                alive = False
    return (np.array(ep), np.array(ec, bool), np.array(sp), np.array(sc, bool))


def picks(ep, sp, sscore):
    e_av = ~np.isnan(ep); s_av = ~np.isnan(sp); both = e_av & s_av
    pick_s = np.zeros(len(ep), bool)
    pick_s[both] = sscore[both] > ep[both]
    pick_s[s_av & ~e_av] = True
    pick_e = (e_av & ~s_av) | (both & ~pick_s)
    return e_av, s_av, pick_s, pick_e


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for tag, (dirname, mname, mcol, mtok, mpk) in CELLS.items():
        d = f"{ROOT}/{dirname}"
        chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
        bad = loopy_rids(d)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        ep, ec, sp, sc = collect(chains, mtok, mpk)
        atom = (~np.isnan(sp)) & (np.round(sp, 4) == 0.5)
        ep_b = np.where(np.isnan(ep), -1.0, ep)

        # precompute per-shift bar data + shared ymax
        data = {}; ymax = 0.0
        for s in SHIFTS:
            sscore = np.where(atom, np.minimum(sp + s, 1.0), sp)
            ss_b = np.where(np.isnan(sp), -1.0, sscore)
            e_av, s_av, pick_s, pick_e = picks(ep, sp, sscore)
            # model row (bin by raw eagle prob)
            e_won = e_av & ec & pick_e
            e_lost = e_av & ec & (~pick_e) & pick_s & (~sc)
            te, we, le = bin_counts(ep_b, e_av, e_won, e_lost)
            # suffix row (bin by SHIFTED score)
            s_won = s_av & sc & pick_s
            s_lost = s_av & sc & (~pick_s) & pick_e & (~ec)
            ts, ws, ls = bin_counts(ss_b, s_av, s_won, s_lost)
            ymax = max(ymax, te.max(), ts.max())
            data[s] = ((te, we, le, int(e_lost.sum())), (ts, ws, ls, int(s_lost.sum())))

        ncol = len(SHIFTS)
        fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.4), squeeze=False)
        rows = [(mname, mcol, 0), ("Suffix", SUF, 1)]
        for s_i, s in enumerate(SHIFTS):
            for (nm, color, ri) in rows:
                tot, won, lost, L = data[s][ri]
                ax = axes[ri][s_i]
                ax.bar(CENT, tot, width=0.045, color="#cccccc", alpha=0.6, zorder=1)
                ax.bar(CENT, won, width=0.045, color=color, alpha=0.8, zorder=2)
                ax.bar(CENT, lost, width=0.045, bottom=won, color=LOST_COLOR,
                       alpha=0.9, hatch="///", edgecolor="white", lw=0.0, zorder=3)
                ax.set_ylim(0, ymax * 1.05); ax.set_xlim(0, 1)
                ax.set_title(f"{nm}  shift +{s:.1f}   LOST={L}", fontsize=8.5)
                ax.grid(axis="y", alpha=0.25)
                if s_i == 0:
                    ax.set_ylabel(f"{nm}\ncount / bin", fontsize=8)
                if ri == 1:
                    ax.set_xlabel("suffix score (0.5-atom shifted)" if nm == "Suffix"
                                  else "prob", fontsize=7.5)
        handles = [
            Patch(facecolor="#cccccc", alpha=0.6, label="total proposals / bin"),
            Patch(facecolor="#666666", alpha=0.8, label="correct & SELECTED (won) [color=proposer]"),
            Patch(facecolor=LOST_COLOR, alpha=0.9, hatch="///", edgecolor="white",
                  label="correct but LOST (not selected & pick wrong)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.005))
        tot_lost = {s: data[s][0][3] + data[s][1][3] for s in SHIFTS}
        fig.suptitle(f"reliability_lost vs manual 0.5-atom shift (no calibration) — {tag}\n"
                     f"suffix row binned by shifted score (0.5 bar slides right); "
                     f"total LOST: " + "  ".join(f"+{s:.1f}={tot_lost[s]}" for s in SHIFTS),
                     fontsize=10)
        fig.tight_layout(rect=[0, 0.04, 1, 0.93])
        outp = OUTDIR / f"shift_lost_{tag}.png"
        fig.savefig(outp, dpi=135); plt.close(fig)
        print(f"{tag}: total LOST " + " ".join(f"+{s:.1f}={tot_lost[s]}" for s in SHIFTS))
        print(f"  wrote {outp}")


if __name__ == "__main__":
    main()
