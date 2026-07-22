"""Can we save the LOST suffix-0.5 tokens by an INTUITIVE *conditional* rule (not a
global shift)? The only lever that can beat the calibrated average is conditioning on
a feature that separates right-from-wrong WITHIN the contested atom.

On the suffix_p=0.5 decisive positions (2-way), split by EAGLE prob: does P(suffix is
the right one | eagle_p) cross 0.5 anywhere? Where it does (eagle weak), routing those
to suffix is CORRECT and recovers tokens the always-eagle calibration loses.

Reports, on the 0.5 atom:
  always-eagle (what calibration does)  vs  conditional-on-eagle_p  vs  oracle
  -> tokens recoverable by the best eagle_p-conditional rule (the realistic ceiling).
Also prints the FULL-decisive joint-GBM recovery (the general feature ceiling) and
frames the only complete escape (multi-candidate verify).

Run (host): python3 simulation/scripts/select1_ladder/save_lost.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_count import ROOT, load_chains, loopy_rids  # noqa: E402

OUTDIR = Path("simulation/results/calib_reliability/figures")
CELLS = {
    "qwen3_14b_2way": ("qwen3_14b_ar", "EAGLE3", "eagle_token", "eagle_p"),
    "qwen35_27b_2way": ("qwen35_27b_ar", "MTP", "eagle_token", "eagle_p"),
}
NB = 10
EDG = np.linspace(0, 1, NB + 1)
CEN = 0.5 * (EDG[:-1] + EDG[1:])


def collect_decisive(chains, mtok, mpk):
    """decisive 2-way rows: eagle_p, suffix_p, depth, s_right(bool), rid."""
    rows = []
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
            hits = []
            if e_av and gt is not None and mt == gt: hits.append("e")
            if s_av and gt is not None and st == gt: hits.append("s")
            if e_av and s_av and len(hits) == 1:
                rows.append((float(mpv), float(spv), int(r["depth"]),
                             1 if hits[0] == "s" else 0, rid))
            if gt is not None and (e_av or s_av) and len(hits) == 0:
                alive = False
    return rows


def oof_gbm(X, y, g):
    pred = np.full(len(y), float(y.mean()))
    if len(y) < 30 or len(set(y.tolist())) < 2:
        return pred
    for tr, te in GroupKFold(min(5, len(set(g.tolist())))).split(X, y, g):
        if len(set(y[tr].tolist())) < 2:
            pred[te] = y[tr].mean(); continue
        m = HGB(max_depth=3, max_iter=200, learning_rate=0.06,
                l2_regularization=1.0).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def analyze(tag, dirname, mname, mtok, mpk, ax):
    d = f"{ROOT}/{dirname}"
    chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
    bad = loopy_rids(d)
    chains = {k: v for k, v in chains.items() if k[0] not in bad}
    rows = collect_decisive(chains, mtok, mpk)
    ep = np.array([r[0] for r in rows]); sp = np.array([r[1] for r in rows])
    dep = np.array([r[2] for r in rows]); sy = np.array([r[3] for r in rows])
    g = np.array([r[4] for r in rows])
    N = len(rows)

    # ---- general feature ceiling: joint-GBM over ALL decisive ----
    raw_sel = float(np.where(sp > ep, sy, 1 - sy).mean())
    pj = oof_gbm(np.c_[ep, sp, dep], sy, g)
    jnt_sel = float(np.where(pj > 0.5, sy, 1 - sy).mean())

    # ---- 0.5 atom: condition on eagle_p ----
    atom = np.round(sp, 4) == 0.5
    ea = ep[atom]; sa = sy[atom]; ga = g[atom]
    n_atom = atom.sum()
    s_right = int(sa.sum()); e_right = n_atom - s_right
    # per eagle_p bin: P(suffix right)
    idx = np.clip(np.digitize(ea, EDG) - 1, 0, NB - 1)
    pbin = np.full(NB, np.nan); nbin = np.zeros(NB)
    for b in range(NB):
        m = idx == b; nbin[b] = m.sum()
        if m.sum() >= 20:
            pbin[b] = sa[m].mean()
    # always-eagle (calibration's choice at the atom): correct = e_right
    sel_eagle = e_right / n_atom
    # conditional per-bin oracle (upper bound): pick the bin-majority
    cond_correct = 0
    for b in range(NB):
        m = idx == b
        if not m.sum():
            continue
        cond_correct += max(sa[m].sum(), m.sum() - sa[m].sum())
    sel_cond = cond_correct / n_atom
    # HONEST OOF: 1D rule on eagle_p (pick suffix iff GBM(eagle_p)>0.5)
    pa = oof_gbm(ea.reshape(-1, 1), sa, ga)
    sel_cond_oof = float(np.where(pa > 0.5, sa, 1 - sa).mean())
    recov_oof = int(round((sel_cond_oof - sel_eagle) * n_atom))

    print(f"\n=== {tag} ===  decisive N={N}")
    print(f"  ALL-decisive selacc: raw={raw_sel:.4f}  joint-GBM(ep,sp,depth)={jnt_sel:.4f}"
          f"  (+{jnt_sel-raw_sel:.4f} = feature ceiling)")
    print(f"  0.5-ATOM (n={n_atom}): suffix-right={s_right} eagle-right={e_right}")
    print(f"    always-eagle (calib)      selacc={sel_eagle:.4f}")
    print(f"    conditional-on-eagle_p OOF selacc={sel_cond_oof:.4f}  "
          f"(+{sel_cond_oof-sel_eagle:.4f} -> recovers ~{recov_oof} tokens)")
    print(f"    per-bin oracle (upper bnd) selacc={sel_cond:.4f}")

    # figure: P(suffix right at 0.5 atom) vs eagle_p
    ok = ~np.isnan(pbin)
    axc = ax.twinx()
    axc.bar(CEN, nbin, width=0.085, color="#cccccc", alpha=0.5, zorder=1)
    axc.set_ylabel("# 0.5-atom positions / bin", fontsize=8, color="#777")
    axc.tick_params(axis="y", labelsize=7, colors="#777")
    ax.axhline(0.5, color="k", ls=":", lw=1.0, zorder=3)
    ax.plot(CEN[ok], pbin[ok], "o-", color="#d62728", lw=1.8, ms=5, zorder=4)
    # shade where suffix wins (recoverable by routing to suffix)
    for b in range(NB):
        if not np.isnan(pbin[b]) and pbin[b] > 0.5:
            ax.axvspan(EDG[b], EDG[b + 1], color="#d62728", alpha=0.08, zorder=0)
    ax.set_zorder(axc.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("eagle prob (at suffix_p=0.5 positions)", fontsize=9)
    ax.set_ylabel("P(suffix is the right one)", fontsize=9)
    ax.set_title(f"{tag}: 0.5-atom — shaded = eagle weak, suffix wins (recoverable)\n"
                 f"always-eagle {sel_eagle:.3f} -> cond-on-eagle_p {sel_cond_oof:.3f} "
                 f"(+~{recov_oof} tok); full joint +{jnt_sel-raw_sel:.3f}", fontsize=9)
    ax.grid(alpha=0.25)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CELLS), figsize=(7 * len(CELLS), 4.8), squeeze=False)
    for k, (tag, cfg) in enumerate(CELLS.items()):
        analyze(tag, *cfg, axes[0][k])
    fig.suptitle("Saving LOST tokens by a CONDITIONAL rule (route suffix-0.5 only where "
                 "eagle is weak) — the only lever beyond the calibrated average",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    outp = OUTDIR / "save_lost.png"
    fig.savefig(outp, dpi=140); plt.close(fig)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
