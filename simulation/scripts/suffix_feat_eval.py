"""Evaluate the suffix accept-prediction feature matrix: univariate AUC, baseline
vs full GBM (CV), per-feature incremental gain over baseline, and lift on the
p~0.5 bucket. Baseline = the already-known {headfrac(suffix_p), logn, logc, matchlen}.
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score

BASE = ["headfrac", "logn", "logc", "matchlen"]
GROUPS = None   # set in main; GroupKFold by request to avoid within-request leakage


def cvauc(X, y, seed=0):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                         max_depth=4, random_state=seed)
    cv = GroupKFold(n_splits=5)
    p = cross_val_predict(clf, X, y, cv=cv, groups=GROUPS, method="predict_proba")[:, 1]
    return roc_auc_score(y, p), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n-add", type=int, default=99)
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    X = d["X"]; y = d["y"]; names = list(d["names"])
    global GROUPS
    GROUPS = d["groups"] if "groups" in d else np.arange(len(y))
    idx = {n: i for i, n in enumerate(names)}
    print(f"(GroupKFold by request: {len(np.unique(GROUPS))} groups)")
    print(f"n={len(y)}  accept={y.mean()*100:.1f}%  features={len(names)}\n")

    # univariate AUC
    print("== univariate AUC (|auc-.5| strength), top 20 ==")
    uni = []
    for i, n in enumerate(names):
        col = X[:, i]
        if np.nanstd(col) == 0:
            uni.append((n, 0.5)); continue
        a = roc_auc_score(y, col)
        uni.append((n, max(a, 1 - a)))
    for n, a in sorted(uni, key=lambda t: -t[1])[:20]:
        print(f"   {n:<14} {a:.3f}")

    bi = [idx[b] for b in BASE]
    base_auc, base_p = cvauc(X[:, bi], y)
    full_auc, full_p = cvauc(X, y)
    print(f"\n== models (5-fold CV AUC) ==")
    print(f"   baseline {BASE} : {base_auc:.4f}")
    print(f"   FULL (all {len(names)} feats)      : {full_auc:.4f}   (delta {full_auc-base_auc:+.4f})")

    # per-feature incremental over baseline (add one)
    print(f"\n== add-one-to-baseline CV-AUC gain (top) ==")
    gains = []
    for n in names:
        if n in BASE:
            continue
        cols = bi + [idx[n]]
        a, _ = cvauc(X[:, cols], y)
        gains.append((n, a - base_auc))
    for n, gnl in sorted(gains, key=lambda t: -t[1])[:args.n_add]:
        print(f"   +{n:<14} {gnl:+.4f}")

    # group ablations: baseline + one feature family
    GROUPS_F = {
        "backoff": ["hf_L1", "hf_L2", "hf_Lm1", "argmax_L1", "argmax_L2", "argmax_Lm1", "vote", "entropy_L2"],
        "crosstree": ["g_has", "g_agree", "g_matchlen", "g_headfrac", "g_logn"],
        "autocorr": ["prev_accept", "streak", "recent_acc", "ml_extend", "logpos"],
        "tokenclass": ["head_space", "head_newline", "head_punct", "head_digit", "head_alpha", "head_len", "prev_space", "prev_punct"],
        "geometry": ["nchildren", "entropy", "margin", "secondfrac", "recency_agree", "dist_recent"],
    }
    print(f"\n== baseline + one feature family (CV-AUC) ==")
    for gname, feats in GROUPS_F.items():
        cols = bi + [idx[f] for f in feats if f in idx]
        a, _ = cvauc(X[:, cols], y)
        print(f"   base+{gname:<10} {a:.4f}  (delta {a-base_auc:+.4f})")
    # compact interpretable model: baseline + backoff + crosstree + tokenclass
    comp_feats = GROUPS_F["backoff"] + GROUPS_F["crosstree"] + GROUPS_F["tokenclass"]
    comp_cols = bi + [idx[f] for f in comp_feats if f in idx]
    comp_auc, comp_p = cvauc(X[:, comp_cols], y)
    print(f"   COMPACT (base+backoff+crosstree+tokenclass, {len(comp_cols)} feats): {comp_auc:.4f}  (delta {comp_auc-base_auc:+.4f})")

    # 0.5 bucket lift (out-of-fold predictions)
    hf = X[:, idx["headfrac"]]
    m = np.abs(hf - 0.5) < 1e-3
    if m.sum() > 50:
        ba = roc_auc_score(y[m], base_p[m]); fa = roc_auc_score(y[m], full_p[m])
        ca = roc_auc_score(y[m], comp_p[m])
        print(f"\n== p~0.5 bucket (n={m.sum()}, accept={y[m].mean()*100:.1f}%) ==")
        print(f"   baseline model AUC on subset : {ba:.4f}")
        print(f"   COMPACT model AUC on subset  : {ca:.4f}   (delta {ca-ba:+.4f})")
        print(f"   FULL model AUC on subset     : {fa:.4f}   (delta {fa-ba:+.4f})")


if __name__ == "__main__":
    main()
