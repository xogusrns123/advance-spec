"""DFlash-vs-suffix selection accuracy + decision-surface heatmap, OFFLINE.

Consumes decisions_dflash_suffix.jsonl (capture_dflash_vs_suffix.py): per
(rid, round, depth) the block drafter's top-1 prob/token (dflash_p/dflash_tok),
the suffix trie's chain proposal (suffix_p/suffix_tok), the GT token, and
oracle_hit ∈ {both,eagle(=dflash),suffix,none}. Decisive = exactly one arm
matches GT. Each round's recorded rows are a natural alive chain (depths 0..acc,
DFlash hits until the death depth), so pooling decisive rows == alive-conditioned.

Outputs (parallel to the EAGLE3/MTP chain-hybrid study):
  1) selection-accuracy ladder: raw (suffix_p>dflash_p) / Bayes-ceiling
     (GBM on (dflash_p,suffix_p,depth), GroupKFold-by-rid held-out) / oracle.
  2) (dflash_p, suffix_p) empirical heatmap with raw / Bayes / best-monotone
     boundaries — the DFlash analog of panelB.

Usage (inside sglang-bench as root for figures):
  python3 simulation/scripts/analyze_dflash_suffix.py --cell qwen3_8b
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from scipy.ndimage import distance_transform_edt

ALIVE = {"eagle", "suffix", "both"}
FIGDIR = "simulation/results/calib_why_analysis/figures"


def load(cell):
    p = f"simulation/results/chain_hybrid_perdepth/{cell}_dflash/decisions_dflash_suffix.jsonl"
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return rows


def selacc_ladder(rows):
    """raw / Bayes-ceiling(held-out) / oracle decisive selection accuracy."""
    dec = [r for r in rows if r["oracle_hit"] in ("eagle", "suffix")]
    n = len(dec)
    sw = sum(1 for r in dec if r["oracle_hit"] == "suffix")
    # raw: pick suffix iff suffix_p > dflash_p (suffix absent => never pick suffix)
    raw_ok = 0
    for r in dec:
        sp = r["suffix_p"] if r["suffix_p"] is not None else -1.0
        pick_suffix = sp > r["dflash_p"]
        raw_ok += (pick_suffix and r["oracle_hit"] == "suffix") or \
                  ((not pick_suffix) and r["oracle_hit"] == "eagle")
    raw = raw_ok / max(n, 1)
    # Bayes ceiling: GBM on (dflash_p, suffix_p, depth), held-out OOF by rid
    X = np.array([[r["dflash_p"], (r["suffix_p"] if r["suffix_p"] is not None else 0.0),
                   r["depth"]] for r in dec], float)
    y = np.array([1 if r["oracle_hit"] == "suffix" else 0 for r in dec])
    grp = np.array([r["rid"] for r in dec])
    oof = np.zeros(n)
    ng = len(set(grp.tolist()))
    if ng >= 2:
        for tr, te in GroupKFold(n_splits=min(5, ng)).split(X, y, grp):
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.06)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        pick = oof > 0.5
        bayes = (((pick) & (y == 1)) | ((~pick) & (y == 0))).mean()
    else:
        bayes = float("nan")
    return {"n": n, "suffix_wins_frac": sw / max(n, 1), "raw": raw,
            "bayes_heldout": bayes, "oracle": 1.0}


def heatmap(rows, cell, model_label):
    """Empirical P(suffix==gt | dflash_p, suffix_p, DECISIVE) + 3 boundaries."""
    dec = [r for r in rows if r["oracle_hit"] in ("eagle", "suffix")
           and r["suffix_p"] is not None]
    dp = np.array([r["dflash_p"] for r in dec])
    sp = np.array([r["suffix_p"] for r in dec])
    corr = np.array([1 if r["oracle_hit"] == "suffix" else 0 for r in dec])
    print(f"  heatmap decisive(suffix-present) n={len(dec)} suffix-wins={corr.mean():.3f}")

    BINS, MIN = 20, 5
    s2, xe, ye = np.histogram2d(dp, sp, bins=BINS, range=[[0, 1], [0, 1]], weights=corr)
    c2, _, _ = np.histogram2d(dp, sp, bins=BINS, range=[[0, 1], [0, 1]])
    with np.errstate(invalid="ignore"):
        M = s2 / c2
    M[c2 < MIN] = np.nan
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("#dddddd")

    g = np.linspace(0, 1, 300)
    mono = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.06,
                                          monotonic_cst=[-1, 1]).fit(np.c_[dp, sp], corr)
    GX, GY = np.meshgrid(g, g)
    Zm = mono.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)

    Mf = M.copy(); occ = ~np.isnan(Mf)
    if occ.any() and not occ.all():
        idx = distance_transform_edt(~occ, return_distances=False, return_indices=True)
        Mf = Mf[tuple(idx)]
    B = (np.nan_to_num(Mf) > 0.5).astype(int)
    segs = []; nx_, ny_ = B.shape
    for i in range(nx_):
        for j in range(ny_):
            if i + 1 < nx_ and B[i, j] != B[i + 1, j]:
                segs.append([(xe[i + 1], ye[j]), (xe[i + 1], ye[j + 1])])
            if j + 1 < ny_ and B[i, j] != B[i, j + 1]:
                segs.append([(xe[i], ye[j + 1]), (xe[i + 1], ye[j + 1])])

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(np.ma.masked_invalid(M.T), origin="lower", extent=[0, 1, 0, 1],
                   cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax).set_label("EMPIRICAL mean(suffix==gt) per bin | DECISIVE")
    if segs:
        ax.add_collection(LineCollection(segs, colors="lime", linewidths=2.6))
    ax.plot([], [], color="lime", lw=2.6, label="Bayes boundary (cell-edge staircase, P=0.5)")
    ax.plot([0, 1], [0, 1], "k-", lw=2.4, label="raw boundary {suffix_p = dflash_p}")
    ax.contour(GX, GY, Zm, levels=[0.5], colors="darkorange", linewidths=2.8)
    ax.plot([], [], color="darkorange", lw=2.8, label="best-monotone boundary (calib ceiling)")
    ax.set_xlabel("dflash_p  (block drafter top-1 prob)"); ax.set_ylabel("suffix_p")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"DFlash vs suffix decision surface — {model_label}\n"
                 "decisive, alive-conditioned (offline, GT teacher-forced)")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    out = f"{FIGDIR}/panelB_{cell}_dflash_suffix.png"
    fig.savefig(out, dpi=140); print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="qwen3_8b")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    rows = load(args.cell)
    label = args.label or args.cell
    print(f"=== {args.cell} DFlash-vs-suffix ({len(rows)} rows) ===")
    L = selacc_ladder(rows)
    print(f"  decisive n={L['n']}  suffix-wins frac={L['suffix_wins_frac']:.3f}")
    print(f"  SEL.ACC  raw={L['raw']:.4f}  Bayes(heldout)={L['bayes_heldout']:.4f}  oracle={L['oracle']:.4f}")
    heatmap(rows, args.cell, label)


if __name__ == "__main__":
    main()
