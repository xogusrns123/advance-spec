"""Does suffix COUNT carry signal the prob-average throws away? (user hypothesis)

calibration maps prob -> mean accept rate, averaging over count. Loss concentrates at
the suffix_p=0.5 atom (mostly c/n=1/2, low count). Hypothesis: high trie-count suffix
proposals are more reliable, so a count-aware score could split the atom.

Measured on accept-conditioned pools (2-way cells), label = token==gt:
  (1) Is count CONDITIONALLY informative? AUC(count->correct) overall and WITHIN prob
      bins (esp the 0.5 atom); accept-rate surface over (suffix_p x count bucket).
  (2) Does count lift SELECTION? decisive selacc with suffix score =
      OOF-GBM P(correct | feats) for feats {prob,depth} / +log count / +match_len,
      vs the model's {prob,depth}. raw(prob>prob) and oracle(=1) as refs.

Run (host): python3 simulation/scripts/select1_ladder/count_advantage.py
"""
import sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_count import ROOT, load_chains, loopy_rids  # noqa: E402

OUTDIR = Path("simulation/results/calib_reliability/figures")
CELLS = {
    "qwen3_14b_2way": ("qwen3_14b_ar", "EAGLE3", "eagle_token", "eagle_p"),
    "qwen35_27b_2way": ("qwen35_27b_ar", "MTP", "eagle_token", "eagle_p"),
}
PROB_EDGES = np.linspace(0, 1, 11)
NBUCK = [(2, 2), (3, 3), (4, 4), (5, 6), (7, 10), (11, 20), (21, 10**9)]
NBUCK_LBL = ["2", "3", "4", "5-6", "7-10", "11-20", "21+"]


def collect(chains, mtok, mpk):
    """suffix pool rows + decisive 2-way rows."""
    suf = []   # (prob, c, n, depth, correct, rid)
    dec = []   # (model_p, suffix_p, c, n, depth, model_correct, rid) -- exactly one correct
    mdl = []   # (prob, depth, correct, rid)
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            mt, mpv = r.get(mtok), r.get(mpk)
            st, spv = r.get("suffix_token"), r.get("suffix_p")
            c, n = r.get("suffix_count"), r.get("suffix_total")
            d = int(r["depth"])
            if mpv is not None and mt is not None:
                mdl.append((float(mpv), d, 1 if (gt is not None and mt == gt) else 0, rid))
            if spv is not None and st is not None and n:
                suf.append((float(spv), float(c or 0), float(n), d,
                            1 if (gt is not None and st == gt) else 0, rid))
            avail = [x for x in (("m", mt, mpv), ("s", st, spv)) if x[1] is not None and x[2] is not None]
            hits = [a[0] for a in avail if gt is not None and a[1] == gt]
            if len(avail) == 2 and len(hits) == 1:
                dec.append((float(mpv), float(spv), float(c or 0), float(n or 0), d,
                            1 if hits[0] == "m" else 0, rid))
            if gt is not None and avail and len(hits) == 0:
                alive = False
    return suf, dec, mdl


def auc(y, x):
    y = np.asarray(y); x = np.asarray(x)
    if len(set(y.tolist())) < 2 or len(y) < 20:
        return float("nan")
    return roc_auc_score(y, x)


def oof(X, y, g):
    pred = np.full(len(y), float(np.mean(y)))
    if len(y) < 30 or len(set(y.tolist())) < 2:
        return pred
    ng = len(set(g.tolist()))
    for tr, te in GroupKFold(min(5, ng)).split(X, y, g):
        if len(set(y[tr].tolist())) < 2:
            pred[te] = y[tr].mean(); continue
        m = HGB(max_depth=3, max_iter=200, learning_rate=0.06, l2_regularization=1.0).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def nbucket_idx(n):
    for i, (lo, hi) in enumerate(NBUCK):
        if lo <= n <= hi:
            return i
    return len(NBUCK) - 1


def analyze(tag, dirname, mname, mtok, mpk):
    d = f"{ROOT}/{dirname}"
    chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
    bad = loopy_rids(d)
    chains = {k: v for k, v in chains.items() if k[0] not in bad}
    suf, dec, mdl = collect(chains, mtok, mpk)
    sp = np.array([r[0] for r in suf]); sc = np.array([r[1] for r in suf])
    sn = np.array([r[2] for r in suf]); sy = np.array([r[4] for r in suf])
    print(f"\n{'='*74}\n{tag}  suffix pool={len(suf)}  decisive={len(dec)}")

    # (1) count signal -------------------------------------------------------
    print(f"  overall AUC -> suffix correct: prob={auc(sy,sp):.3f}  "
          f"count_n={auc(sy,sn):.3f}  count_c={auc(sy,sc):.3f}  log_n={auc(sy,np.log1p(sn)):.3f}")
    # within-prob-bin conditional AUC of count (does count add WHERE prob is fixed?)
    print(f"  conditional AUC(count_n -> correct) within prob bins:")
    pb = np.clip(np.digitize(sp, PROB_EDGES) - 1, 0, len(PROB_EDGES) - 2)
    for b in range(len(PROB_EDGES) - 1):
        m = pb == b
        if m.sum() < 100:
            continue
        a = auc(sy[m], sn[m])
        print(f"    prob[{PROB_EDGES[b]:.1f},{PROB_EDGES[b+1]:.1f}) n={m.sum():6} "
              f"acc={sy[m].mean():.3f} AUC(n)={a:.3f}  n[p50={np.median(sn[m]):.0f}]")
    # the 0.5 atom: accept rate by count bucket
    atom = (sp > 0.49) & (sp < 0.51)
    print(f"  --- suffix_p~0.5 ATOM (n={atom.sum()}, {100*atom.sum()/len(suf):.0f}% of suffix) "
          f"accept rate by count bucket: ---")
    for i, lbl in enumerate(NBUCK_LBL):
        bi = np.array([nbucket_idx(x) for x in sn])
        m = atom & (bi == i)
        if m.sum() < 20:
            continue
        print(f"    n={lbl:6} cnt={m.sum():5} accept={sy[m].mean():.3f}")

    # accept-rate surface (prob x count) for the figure
    surf = np.full((len(PROB_EDGES) - 1, len(NBUCK)), np.nan)
    cnts = np.zeros_like(surf)
    bi_all = np.array([nbucket_idx(x) for x in sn])
    for b in range(len(PROB_EDGES) - 1):
        for j in range(len(NBUCK)):
            m = (pb == b) & (bi_all == j)
            cnts[b, j] = m.sum()
            if m.sum() >= 20:
                surf[b, j] = sy[m].mean()

    # (2) selection lift from count ----------------------------------------
    if dec:
        mp = np.array([r[0] for r in dec]); spd = np.array([r[1] for r in dec])
        cd = np.array([r[2] for r in dec]); nd = np.array([r[3] for r in dec])
        dd = np.array([r[4] for r in dec]); mc = np.array([r[5] for r in dec])
        g = np.array([r[6] for r in dec])
        # model score: OOF P(correct|[prob,depth]); suffix correct = 1-mc
        Pm = oof(np.c_[mp, dd], mc, g)
        suf_y = 1 - mc
        variants = {
            "prob": np.c_[spd, dd],
            "prob+logN": np.c_[spd, dd, np.log1p(nd)],
            "prob+logN+logC": np.c_[spd, dd, np.log1p(nd), np.log1p(cd)],
        }
        raw_sel = float((np.where(spd > mp, suf_y, mc)).mean())
        print(f"  decisive selacc:  raw(prob>prob)={raw_sel:.4f}  oracle=1.0")
        for name, Xs in variants.items():
            Ps = oof(Xs, suf_y, g)
            sel = float(np.where(Ps > Pm, suf_y, mc).mean())
            print(f"    suffix={name:16} -> selacc={sel:.4f}")
    return surf, cnts


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    surfs = {}
    for tag, (dirname, mname, mtok, mpk) in CELLS.items():
        surfs[tag] = analyze(tag, dirname, mname, mtok, mpk)
    # figure: accept-rate surface (prob x count) per cell
    fig, axes = plt.subplots(1, len(CELLS), figsize=(6.6 * len(CELLS), 5.2), squeeze=False)
    for k, (tag, (surf, cnts)) in enumerate(surfs.items()):
        ax = axes[0][k]
        im = ax.imshow(surf, origin="lower", aspect="auto", cmap="RdYlBu",
                       vmin=0, vmax=1, extent=[-0.5, len(NBUCK) - 0.5, 0, 1])
        for b in range(surf.shape[0]):
            for j in range(surf.shape[1]):
                if not np.isnan(surf[b, j]):
                    ax.text(j, (b + 0.5) / surf.shape[0], f"{surf[b,j]:.2f}\n{int(cnts[b,j])}",
                            ha="center", va="center", fontsize=6,
                            color="black")
        ax.set_xticks(range(len(NBUCK))); ax.set_xticklabels(NBUCK_LBL, fontsize=8)
        ax.set_xlabel("suffix count n (suffix_total) bucket")
        ax.set_ylabel("suffix_p (c/n)")
        ax.set_title(f"{tag}\nsuffix accept rate over (prob x count)\n"
                     "row-constant => count adds nothing; varies => count is signal", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label="P(suffix token==gt)")
    fig.suptitle("Does suffix COUNT split the prob-average? accept rate vs (prob, count)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outp = OUTDIR / "count_advantage_surface.png"
    fig.savefig(outp, dpi=140); plt.close(fig)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
