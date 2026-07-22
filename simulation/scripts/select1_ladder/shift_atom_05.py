"""Manual experiment (NO calibration): push the suffix 0.5-atom score up by 0.1 steps
and re-run raw argmax selection. Tests the user's idea: "lift the 0.5 suffix to the
right so they stop being lost."

2-way cells. accept-conditioned positions. selection = argmax(eagle_p, suffix_score)
where suffix_score = suffix_p + shift  iff round(suffix_p,4)==0.5  (else raw suffix_p),
clipped to 1.0. NO calibration on either side.

Reports per shift s in {0,.1,.2,.3,.4,.5}:
  decisive selacc (both avail, exactly one correct -> pick correct?)
  total LOST (correct-but-not-selected & pick wrong, full pool)
  at the 0.5 atom: #suffix-right vs #eagle-right (decisive) and how the picks flip.

Run (host): python3 simulation/scripts/select1_ladder/shift_atom_05.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_count import ROOT, load_chains, loopy_rids  # noqa: E402

OUTDIR = Path("simulation/results/calib_reliability/figures")
CELLS = {
    "qwen3_14b_2way": ("qwen3_14b_ar", "EAGLE3", "eagle_token", "eagle_p"),
    "qwen35_27b_2way": ("qwen35_27b_ar", "MTP", "eagle_token", "eagle_p"),
}
SHIFTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def collect(chains, mtok, mpk):
    """accept-conditioned positions: (eagle_p|nan, e_correct, suffix_p|nan, s_correct)."""
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


def evaluate(ep, ec, sp, sc, shift):
    """returns (decisive_selacc, total_LOST, n_dec, flips_at_atom dict)."""
    e_av = ~np.isnan(ep); s_av = ~np.isnan(sp)
    atom = s_av & (np.round(sp, 4) == 0.5)
    s_score = np.where(atom, np.minimum(sp + shift, 1.0), sp)
    # pick: 'e' or 's'. both avail -> argmax; one avail -> that one.
    both = e_av & s_av
    pick_s = np.zeros(len(ep), bool)
    pick_s[both] = s_score[both] > ep[both]           # tie -> eagle
    pick_s[s_av & ~e_av] = True
    pick_e = (e_av & ~s_av) | (both & ~pick_s)
    # LOST (full pool): correct & not selected & the pick was wrong
    suffix_lost = s_av & sc & (~pick_s) & pick_e & (~ec)
    eagle_lost = e_av & ec & (~pick_e) & pick_s & (~sc)
    lost = int(suffix_lost.sum() + eagle_lost.sum())
    # decisive selacc (both avail, exactly one correct)
    dec = both & (ec != sc)
    pick_correct = np.where(pick_s, sc, ec)
    selacc = float(pick_correct[dec].mean())
    # 0.5-atom decisive breakdown
    atom_dec = dec & atom
    info = dict(n_atom_dec=int(atom_dec.sum()),
                s_right=int(sc[atom_dec].sum()), e_right=int(ec[atom_dec].sum()),
                picked_s=int(pick_s[atom_dec].sum()))
    return selacc, lost, int(dec.sum()), info


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CELLS), figsize=(7 * len(CELLS), 4.8), squeeze=False)
    for k, (tag, (dirname, mname, mtok, mpk)) in enumerate(CELLS.items()):
        d = f"{ROOT}/{dirname}"
        chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
        bad = loopy_rids(d)
        chains = {kk: v for kk, v in chains.items() if kk[0] not in bad}
        ep, ec, sp, sc = collect(chains, mtok, mpk)
        b = evaluate(ep, ec, sp, sc, 0.0)
        print(f"\n=== {tag} ===  decisive={b[2]}  "
              f"0.5-atom decisive: n={b[3]['n_atom_dec']} "
              f"suffix-right={b[3]['s_right']} eagle-right={b[3]['e_right']}")
        print(f"  shift  selacc   LOST   (atom picks_suffix / n_atom_dec)")
        las, sas = [], []
        for s in SHIFTS:
            selacc, lost, ndec, info = evaluate(ep, ec, sp, sc, s)
            las.append(lost); sas.append(selacc)
            print(f"  +{s:.1f}   {selacc:.4f}  {lost:5d}   "
                  f"({info['picked_s']:5d}/{info['n_atom_dec']})")
        ax = axes[0][k]
        ax.plot(SHIFTS, las, "o-", color="#ff7f0e", label="total LOST")
        ax.set_xlabel("shift added to suffix 0.5-atom score (no calibration)")
        ax.set_ylabel("total LOST (correct but not selected)", color="#ff7f0e")
        ax.tick_params(axis="y", colors="#ff7f0e")
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax2 = ax.twinx()
        ax2.plot(SHIFTS, sas, "s--", color="#1f77b4", label="decisive selacc")
        ax2.set_ylabel("decisive selacc", color="#1f77b4")
        ax2.tick_params(axis="y", colors="#1f77b4")
        nr, er = b[3]["s_right"], b[3]["e_right"]
        ax.set_title(f"{tag}: lifting the suffix 0.5-atom (no calib)\n"
                     f"0.5-atom decisive: suffix-right {nr} vs eagle-right {er} "
                     f"-> lifting suffix trades {er} eagle-right for {nr} suffix-right",
                     fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle("Manually pushing the suffix 0.5-atom RIGHT (no calibration): "
                 "does it recover the 'disappearing' suffix?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    outp = OUTDIR / "shift_atom_05.png"
    fig.savefig(outp, dpi=140); plt.close(fig)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
