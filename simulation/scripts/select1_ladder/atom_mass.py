"""Does the MASS at each suffix prob-atom help? (user's clarified hypothesis)

suffix_p = c/n is DISCRETE: proposals pile at rational atoms (0.5, 0.667, 0.75, ...).
The user asks: give an ADVANTAGE to high-mass atoms (where many proposals share the
same prob value). Key fact tested empirically: the mass at a prob value is a
DETERMINISTIC FUNCTION of that prob value, so given the prob it carries no extra
discriminative info -- and calibration already maps each atom to its own accept rate.

On accept-conditioned 2-way pools, label = token==gt:
  (1) atom table: top suffix_p values by mass, their accept rate (= what calibration
      already assigns each atom; computed over huge samples -> near-zero variance).
  (2) decisive selacc: suffix score = OOF-GBM P(correct|feats) for {prob,depth} vs
      +log(atom-mass). lift quantifies whether mass adds anything (expect ~0).
  (3) DIRECT advantage test: boost suffix's calibrated score by alpha * z(log mass)
      and sweep alpha>=0; selacc should PEAK at alpha=0 (mass-advantage can't help).

Run (host): python3 simulation/scripts/select1_ladder/atom_mass.py
"""
import sys
from pathlib import Path
from collections import defaultdict, Counter
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


def collect(chains, mtok, mpk):
    suf = []   # (prob_rounded, depth, correct, rid)
    dec = []   # (model_p, suffix_p_rounded, depth, model_correct, rid)
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            mt, mpv = r.get(mtok), r.get(mpk)
            st, spv = r.get("suffix_token"), r.get("suffix_p")
            d = int(r["depth"])
            if spv is not None and st is not None:
                suf.append((round(float(spv), 4), d,
                            1 if (gt is not None and st == gt) else 0, rid))
            avail = [x for x in (("m", mt, mpv), ("s", st, spv)) if x[1] is not None and x[2] is not None]
            hits = [a[0] for a in avail if gt is not None and a[1] == gt]
            if len(avail) == 2 and len(hits) == 1:
                dec.append((float(mpv), round(float(spv), 4), d,
                            1 if hits[0] == "m" else 0, rid))
            if gt is not None and avail and len(hits) == 0:
                alive = False
    return suf, dec


def oof(X, y, g):
    pred = np.full(len(y), float(np.mean(y)))
    if len(y) < 30 or len(set(y.tolist())) < 2:
        return pred
    for tr, te in GroupKFold(min(5, len(set(g.tolist())))).split(X, y, g):
        if len(set(y[tr].tolist())) < 2:
            pred[te] = y[tr].mean(); continue
        m = HGB(max_depth=3, max_iter=200, learning_rate=0.06, l2_regularization=1.0).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def analyze(tag, dirname, mname, mtok, mpk):
    d = f"{ROOT}/{dirname}"
    chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
    bad = loopy_rids(d)
    chains = {k: v for k, v in chains.items() if k[0] not in bad}
    suf, dec = collect(chains, mtok, mpk)
    N = len(suf)
    vals = np.array([r[0] for r in suf]); sy = np.array([r[2] for r in suf])
    mass = Counter(vals.tolist())
    accrate = {}
    for v in mass:
        m = vals == v
        accrate[v] = sy[m].mean()
    print(f"\n{'='*70}\n{tag}  suffix pool={N}  distinct prob-atoms={len(mass)}")
    top = sorted(mass.items(), key=lambda kv: -kv[1])[:12]
    print(f"  top atoms by mass:  value   count    %pool   accept_rate (=calib value)")
    for v, c in top:
        print(f"     {v:6.4f}  {c:7d}  {100*c/N:5.1f}%   {accrate[v]:.3f}")
    print(f"  -> top {len(top)} atoms cover "
          f"{100*sum(c for _,c in top)/N:.0f}% of all suffix proposals")

    if dec:
        mp = np.array([r[0] for r in dec]); sv = np.array([r[1] for r in dec])
        dd = np.array([r[2] for r in dec]); mc = np.array([r[3] for r in dec])
        g = np.array([r[4] for r in dec])
        suf_y = 1 - mc
        logmass = np.log1p(np.array([mass.get(v, 0) for v in sv], float))
        Pm = oof(np.c_[mp, dd], mc, g)
        raw = float(np.where(sv > mp, suf_y, mc).mean())
        Ps0 = oof(np.c_[sv, dd], suf_y, g)
        Ps1 = oof(np.c_[sv, dd, logmass], suf_y, g)
        s0 = float(np.where(Ps0 > Pm, suf_y, mc).mean())
        s1 = float(np.where(Ps1 > Pm, suf_y, mc).mean())
        print(f"  decisive selacc:  raw={raw:.4f}  |  suffix[prob,depth]={s0:.4f}  "
              f"suffix[prob,depth,+log mass]={s1:.4f}  (mass lift {s1-s0:+.4f})")
        # direct advantage sweep: boost suffix score by alpha * z(log mass)
        z = (logmass - logmass.mean()) / (logmass.std() + 1e-9)
        print(f"  ADVANTAGE sweep (pick suffix iff Ps0 + alpha*z(logmass) > Pm):")
        for a in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5):
            sel = float(np.where(Ps0 + a * z > Pm, suf_y, mc).mean())
            print(f"     alpha={a:4.2f} -> selacc={sel:.4f}")
    return top, accrate, N


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    res = {tag: analyze(tag, *cfg) for tag, cfg in CELLS.items()}
    fig, axes = plt.subplots(1, len(CELLS), figsize=(7 * len(CELLS), 4.6), squeeze=False)
    for k, (tag, (top, accrate, N)) in enumerate(res.items()):
        ax = axes[0][k]
        top = sorted(top, key=lambda kv: kv[0])
        xs = [f"{v:.3f}" for v, _ in top]; cs = [c for _, c in top]
        ar = [accrate[v] for v, _ in top]
        bars = ax.bar(range(len(xs)), cs, color=plt.cm.RdYlBu(ar), edgecolor="#333", lw=0.5)
        for i, (c, a) in enumerate(zip(cs, ar)):
            ax.text(i, c, f"{a:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=45, fontsize=7, ha="right")
        ax.set_xlabel("suffix_p atom value"); ax.set_ylabel("mass (# proposals)")
        ax.set_title(f"{tag}: suffix prob piles at atoms\n"
                     "(bar=mass, number/color=accept rate calibration assigns)", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="RdYlBu", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes[0], fraction=0.04, label="atom accept rate")
    fig.suptitle("suffix prob-atoms: mass and the accept rate calibration already gives each",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    outp = OUTDIR / "atom_mass.png"
    fig.savefig(outp, dpi=140); plt.close(fig)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
