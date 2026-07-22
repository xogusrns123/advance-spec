"""Per prob-bin count decomposition incl. "correct but LOST" — raw rule vs calibrated rule.

Companion to reliability_count.py. For each proposer, per prob bin (raw prob on the
left panel, OOF-calibrated prob on the right), stack three counts:
  total proposals (gray)
  correct & SELECTED (won)  -- proposer right AND the rule picked it
  correct but LOST          -- proposer right, rule picked someone ELSE, and that
                               someone was WRONG (an avoidable chain death = the
                               recoverable selection headroom).
"Lost-but-harmless" (right but another right proposer was taken) is NOT counted as
lost -- it costs nothing. The LOST bar is what a better selector could recover; it
should SHRINK from the raw panel to the calibrated panel wherever calibration helps.

Selection rule: raw panel = argmax raw prob; calibrated panel = argmax OOF-isotonic
calibrated prob (per-proposer, GroupKFold by rid). Population = accept-conditioned.

Run (host): python3 simulation/scripts/select1_ladder/reliability_lost.py
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_count import (  # noqa: E402
    ROOT, CELLS, EDGES, CENT, NB, load_chains, loopy_rids, oof_isotonic, oof_atom,
    bin_stats,
)

OUTDIR = Path("simulation/results/calib_reliability/figures")
LOST_COLOR = "#ff7f0e"


def collect_positions(chains, props):
    """accept-conditioned positions: each = {rid, av:{nm:(raw_p, correct)}}."""
    names = [p[0] for p in props]
    positions = []
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            av = {}
            for nm, tkf, pkf, _ in props:
                t = r.get(tkf); p = r.get(pkf)
                if t is not None and p is not None:
                    av[nm] = (float(p), 1 if (gt is not None and t == gt) else 0)
            if av:
                positions.append({"rid": rid, "av": av})
            hits = [nm for nm, (p, c) in av.items() if c == 1]
            if gt is not None and av and len(hits) == 0:
                alive = False
    return positions, names


def bin_counts(vals, totmask, wonmask, lostmask):
    """3 per-bin count vectors given a prob array `vals` and boolean masks."""
    idx = np.clip(np.digitize(vals, EDGES) - 1, 0, NB - 1)
    tot = np.zeros(NB); won = np.zeros(NB); lost = np.zeros(NB)
    for b in range(NB):
        m = idx == b
        tot[b] = (m & totmask).sum()
        won[b] = (m & wonmask).sum()
        lost[b] = (m & lostmask).sum()
    return tot, won, lost


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for tag, (dirname, props) in CELLS.items():
        d = f"{ROOT}/{dirname}"
        chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
        bad = loopy_rids(d)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        positions, names = collect_positions(chains, props)
        npos = len(positions)

        # OOF-calibrate each proposer; map calibrated prob back to (pos_idx, nm)
        rowsby = defaultdict(list)  # nm -> [(pos_idx, raw_p, correct, rid)]
        for i, pos in enumerate(positions):
            for nm, (p, c) in pos["av"].items():
                rowsby[nm].append((i, p, c, pos["rid"]))
        calp = {}; acalp = {}
        for nm, rows in rowsby.items():
            P = np.array([r[1] for r in rows]); Y = np.array([r[2] for r in rows])
            G = np.array([r[3] for r in rows])
            iso = oof_isotonic(P, Y, G)
            # atom-calibration only for the discrete suffix; eagle/mtp/dflash keep
            # isotonic (per-value lookup overfits their continuous floats).
            atom = oof_atom(P, Y, G) if nm.lower() == "suffix" else iso
            for r, pi, pa in zip(rows, iso, atom):
                calp[(r[0], nm)] = float(pi); acalp[(r[0], nm)] = float(pa)

        # per-position raw / isotonic / atom picks
        for i, pos in enumerate(positions):
            av = pos["av"]
            pos["raw_pick"] = max(av, key=lambda nm: av[nm][0])
            pos["cal_pick"] = max(av, key=lambda nm: calp[(i, nm)])
            pos["atom_pick"] = max(av, key=lambda nm: acalp[(i, nm)])

        nprop = len(props)
        fig, axes = plt.subplots(nprop, 4, figsize=(19, 3.0 * nprop), squeeze=False)
        print(f"\n=== {tag} (dir={dirname}) positions={npos} ===")

        # build per-proposer arrays once
        prepped = []; ymax = 0.0
        tot_correct = tot_lost_raw = tot_lost_cal = tot_lost_atom = 0
        for nm, _, _, color in props:
            rows = rowsby[nm]
            idxs = np.array([r[0] for r in rows])
            rawp = np.array([r[1] for r in rows])
            corr = np.array([r[2] for r in rows], bool)
            calv = np.array([calp[(r[0], nm)] for r in rows])
            acalv = np.array([acalp[(r[0], nm)] for r in rows])
            totm = np.ones(len(rows), bool)

            def won_lost(pick_key, sel_vals):
                sel = np.array([positions[i][pick_key] == nm for i in idxs])
                pick_corr = np.array(
                    [positions[i]["av"][positions[i][pick_key]][1] == 1 for i in idxs])
                won = corr & sel
                lost = corr & (~sel) & (~pick_corr)
                return bin_counts(sel_vals, totm, won, lost), int(lost.sum())

            (tr, wr, lr), Lr = won_lost("raw_pick", rawp)
            (tc, wc, lc), Lc = won_lost("cal_pick", calv)
            (ta, wa, la), La = won_lost("atom_pick", acalv)
            # single-proposer reference: this proposer used ALONE -> always selected,
            # so every correct token is captured (won) and NOTHING is lost to
            # selection (LOST=0). Binned by raw prob (its native, uncalibrated score).
            ts, ws, ls = bin_counts(rawp, totm, corr, np.zeros(len(rows), bool))
            ymax = max(ymax, ts.max(), tr.max(), tc.max(), ta.max())
            prepped.append((nm, color, (ts, ws, ls), (tr, wr, lr), (tc, wc, lc),
                            (ta, wa, la), int(corr.sum()), Lr, Lc, La))
            tot_correct += int(corr.sum()); tot_lost_raw += Lr
            tot_lost_cal += Lc; tot_lost_atom += La
            print(f"  {nm:8} correct={int(corr.sum()):6}  LOST raw={Lr:5} -> "
                  f"iso={Lc:5} -> atom={La:5}  (iso recov {Lr-Lc:+5}, atom recov {Lr-La:+5})")

        for r, (nm, color, single3, raw3, cal3, atom3, C, Lr, Lc, La) in enumerate(prepped):
            for col, (tot, won, lost), rule, L in (
                    (0, single3, "single", 0), (1, raw3, "raw", Lr),
                    (2, cal3, "isotonic", Lc), (3, atom3, "atom", La)):
                ax = axes[r][col]
                ax.bar(CENT, tot, width=0.045, color="#cccccc", alpha=0.6, zorder=1)
                ax.bar(CENT, won, width=0.045, color=color, alpha=0.8, zorder=2)
                ax.bar(CENT, lost, width=0.045, bottom=won, color=LOST_COLOR,
                       alpha=0.9, hatch="///", edgecolor="white", lw=0.0, zorder=3)
                ax.set_ylim(0, ymax * 1.05)
                ax.set_xlim(0, 1)
                xl = {"single": "raw prob", "raw": "raw prob",
                      "isotonic": "calibrated prob", "atom": "atom-calibrated prob"}[rule]
                ax.set_xlabel(xl, fontsize=8)
                ax.set_ylabel("count / bin", fontsize=8)
                if rule == "single":
                    ax.set_title(f"{nm}: SINGLE proposer (no selection)   "
                                 f"correct={C}, LOST=0", fontsize=8.5)
                else:
                    ax.set_title(f"{nm}: {rule.upper()} select-1   correct={C}, "
                                 f"LOST={L} ({L/max(C,1):.0%} of correct)", fontsize=8.5)
                ax.grid(axis="y", alpha=0.25)
        # one legend for the figure
        handles = [
            Patch(facecolor="#cccccc", alpha=0.6, label="total proposals / bin"),
            Patch(facecolor="#666666", alpha=0.8, label="correct & SELECTED (won)  [color=proposer]"),
            Patch(facecolor=LOST_COLOR, alpha=0.9, hatch="///", edgecolor="white",
                  label="correct but LOST (not selected & the pick was wrong)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"Correct-but-LOST decomposition: single vs raw / isotonic / atom "
                     f"selection — {tag}\n(accept-conditioned; LOST = recoverable selection "
                     f"headroom; cell total LOST raw {tot_lost_raw} -> iso {tot_lost_cal} "
                     f"-> atom {tot_lost_atom})", fontsize=11)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        outp = OUTDIR / f"reliability_lost_{tag}.png"
        fig.savefig(outp, dpi=140); plt.close(fig)
        print(f"  TOTAL correct={tot_correct}  LOST raw={tot_lost_raw} -> "
              f"iso={tot_lost_cal} -> atom={tot_lost_atom}")
        print(f"  wrote {outp}")


if __name__ == "__main__":
    main()
