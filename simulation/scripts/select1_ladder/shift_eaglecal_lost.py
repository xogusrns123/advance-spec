"""eagle CALIBRATED (isotonic) + suffix RAW with the 0.5-atom shifted up by 0.1 steps.

Hybrid of the two ideas: the model proposer gets an honest isotonic calibration; the
suffix 0.5-atom is manually lifted (no calibration). selection = argmax(eagle_cal,
suffix_p + shift@0.5atom). Drawn in the reliability_lost format (cols = shifts;
eagle row binned by eagle_cal, suffix row by the shifted raw score). stdout compares
eagle-RAW vs eagle-CAL across shifts and the both-calibrated reference.

Run (host): python3 simulation/scripts/select1_ladder/shift_eaglecal_lost.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_count import ROOT, CENT, load_chains, loopy_rids, oof_isotonic  # noqa: E402
from reliability_lost import bin_counts, LOST_COLOR  # noqa: E402

OUTDIR = Path("simulation/results/calib_reliability/figures")
SHIFTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
CELLS = {
    "qwen3_14b_2way": ("qwen3_14b_ar", "EAGLE3", "#1f77b4", "eagle_token", "eagle_p"),
    "qwen35_27b_2way": ("qwen35_27b_ar", "MTP", "#1f77b4", "eagle_token", "eagle_p"),
}
SUF = "#d62728"


def collect(chains, mtok, mpk):
    ep, ec, sp, sc, rid = [], [], [], [], []
    for (r_id, ds), rs in chains.items():
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
            rid.append(r_id)
            hit = ((e_av and mt == gt) or (s_av and st == gt)) if gt is not None else False
            if gt is not None and not hit:
                alive = False
    return (np.array(ep), np.array(ec, bool), np.array(sp), np.array(sc, bool),
            np.array(rid))


def calibrate(vals, corr, rid, av):
    """OOF isotonic mapped back to all positions; non-available -> nan."""
    out = np.full(len(vals), np.nan)
    out[av] = oof_isotonic(vals[av], corr[av].astype(float), rid[av])
    return out


def picks(escore, sscore):
    e_av = ~np.isnan(escore); s_av = ~np.isnan(sscore); both = e_av & s_av
    pick_s = np.zeros(len(escore), bool)
    pick_s[both] = sscore[both] > escore[both]
    pick_s[s_av & ~e_av] = True
    pick_e = (e_av & ~s_av) | (both & ~pick_s)
    return e_av, s_av, pick_s, pick_e


def metrics(escore, sscore, ec, sc):
    e_av, s_av, pick_s, pick_e = picks(escore, sscore)
    both = e_av & s_av
    dec = both & (ec != sc)
    pick_correct = np.where(pick_s, sc, ec)
    selacc = float(pick_correct[dec].mean())
    e_lost = e_av & ec & (~pick_e) & pick_s & (~sc)
    s_lost = s_av & sc & (~pick_s) & pick_e & (~ec)
    return selacc, int(e_lost.sum()), int(s_lost.sum())


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for tag, (dirname, mname, mcol, mtok, mpk) in CELLS.items():
        d = f"{ROOT}/{dirname}"
        chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
        bad = loopy_rids(d)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        ep, ec, sp, sc, rid = collect(chains, mtok, mpk)
        e_av0 = ~np.isnan(ep); s_av0 = ~np.isnan(sp)
        atom = s_av0 & (np.round(sp, 4) == 0.5)
        ecal = calibrate(ep, ec, rid, e_av0)
        scal = calibrate(sp, sc, rid, s_av0)

        # references
        print(f"\n=== {tag} ===")
        sa, el, sl = metrics(ep, sp, ec, sc)
        print(f"  both RAW (shift0):        selacc={sa:.4f}  LOST={el+sl} (e{el}+s{sl})")
        sa, el, sl = metrics(ecal, scal, ec, sc)
        print(f"  both CALIBRATED:          selacc={sa:.4f}  LOST={el+sl} (e{el}+s{sl})")
        print(f"  shift  | eagle-RAW selacc/LOST | eagle-CAL selacc/LOST")
        eraw_l, ecal_l, ecal_sa = [], [], []
        bardata = {}
        for s in SHIFTS:
            ss = np.where(atom, np.minimum(sp + s, 1.0), sp)
            sa_r, el_r, sl_r = metrics(ep, ss, ec, sc)      # eagle raw
            sa_c, el_c, sl_c = metrics(ecal, ss, ec, sc)    # eagle calibrated
            eraw_l.append(el_r + sl_r); ecal_l.append(el_c + sl_c); ecal_sa.append(sa_c)
            print(f"  +{s:.1f}   |   {sa_r:.4f} / {el_r+sl_r:5d}     "
                  f"|   {sa_c:.4f} / {el_c+sl_c:5d}  (e{el_c}+s{sl_c})")
            # bars for eagle-CAL figure
            ss_b = np.where(np.isnan(sp), -1.0, ss)
            ec_b = np.where(np.isnan(ecal), -1.0, ecal)
            e_av, s_av, pick_s, pick_e = picks(ecal, ss)
            te, we, le = bin_counts(ec_b, e_av, e_av & ec & pick_e,
                                    e_av & ec & (~pick_e) & pick_s & (~sc))
            ts, ws, ls = bin_counts(ss_b, s_av, s_av & sc & pick_s,
                                    s_av & sc & (~pick_s) & pick_e & (~ec))
            bardata[s] = ((te, we, le, int((e_av & ec & (~pick_e) & pick_s & (~sc)).sum())),
                          (ts, ws, ls, int((s_av & sc & (~pick_s) & pick_e & (~ec)).sum())))

        ymax = max(max(b[0][0].max(), b[1][0].max()) for b in bardata.values())
        ncol = len(SHIFTS)
        fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.4), squeeze=False)
        rows = [(f"{mname} (calibrated)", mcol, 0), ("Suffix (raw, 0.5 shifted)", SUF, 1)]
        for s_i, s in enumerate(SHIFTS):
            for nm, color, ri in rows:
                tot, won, lost, L = bardata[s][ri]
                ax = axes[ri][s_i]
                ax.bar(CENT, tot, width=0.045, color="#cccccc", alpha=0.6, zorder=1)
                ax.bar(CENT, won, width=0.045, color=color, alpha=0.8, zorder=2)
                ax.bar(CENT, lost, width=0.045, bottom=won, color=LOST_COLOR,
                       alpha=0.9, hatch="///", edgecolor="white", lw=0.0, zorder=3)
                ax.set_ylim(0, ymax * 1.05); ax.set_xlim(0, 1)
                ax.set_title(f"{nm.split()[0]}  +{s:.1f}  LOST={L}", fontsize=8.5)
                ax.grid(axis="y", alpha=0.25)
                if s_i == 0:
                    ax.set_ylabel(f"{nm}\ncount / bin", fontsize=7.5)
                if ri == 1:
                    ax.set_xlabel("shifted suffix score" if "Suffix" in nm
                                  else "calibrated prob", fontsize=7.5)
        handles = [
            Patch(facecolor="#cccccc", alpha=0.6, label="total / bin"),
            Patch(facecolor="#666666", alpha=0.8, label="correct & SELECTED (won)"),
            Patch(facecolor=LOST_COLOR, alpha=0.9, hatch="///", edgecolor="white",
                  label="correct but LOST"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.005))
        fig.suptitle(f"eagle CALIBRATED + suffix RAW 0.5-atom shifted — {tag}\n"
                     f"total LOST: " + "  ".join(f"+{s:.1f}={ecal_l[i]}"
                                                 for i, s in enumerate(SHIFTS)),
                     fontsize=10)
        fig.tight_layout(rect=[0, 0.04, 1, 0.93])
        outp = OUTDIR / f"shift_eaglecal_lost_{tag}.png"
        fig.savefig(outp, dpi=135); plt.close(fig)
        print(f"  wrote {outp}")


if __name__ == "__main__":
    main()
