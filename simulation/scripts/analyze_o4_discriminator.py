"""Does a JOINT discriminator (calibration + cross-features) beat raw selection?

A per-proposer calibrator maps each proposer's own score to its marginal accept
prob and compares the two. A discriminator instead fits ONE model on BOTH
proposers' features jointly with the comparative label (is suffix the GT pick?).
The user's point is correct: this is just calibration whose features include the
other proposer + depth/count/match_len and whose target is the comparison.

We measure, OFFLINE, on the oracle log's decisive+contested decisions (both
proposers proposed a token, exactly one matches GT), with GroupKFold by rid so
test conversations are disjoint from train, the held-out selection accuracy & AUC
of:
  always-suffix         prob-blind majority
  raw                   suffix_p > eagle_p                       (current rule)
  LR[probs]             logistic on [eagle_p, suffix_p]          (joint, 2 feat)
  LR[+depth]            + depth
  LR[+all]              + log1p(count,total), match_len, score, agreement
  GBDT[all]             gradient-boosted trees on all features   (nonlinear)
  oracle                1.0 ceiling

The ablation (probs -> +depth -> +all) isolates how much INCREMENTAL signal
depth/count/match_len carry beyond the two probabilities. Feature importances
show which ones matter. This gates whether a serving run is worth it; it does NOT
itself prove MAT (trajectory changes) -- it measures the selection ceiling.

Usage:
  python3 simulation/scripts/analyze_o4_discriminator.py \
      --dir simulation/results/o4_perdepth/qwen3_14b_replay --model-label EAGLE3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.ensemble import HistGradientBoostingClassifier as GBDT
    _GBDT_KW = dict(max_iter=200, max_depth=3, learning_rate=0.08)
except Exception:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier as GBDT
    _GBDT_KW = dict(n_estimators=200, max_depth=3, learning_rate=0.08)


def f(x, default=0.0):
    return default if x is None else float(x)


def load(path):
    rows, rids, y = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") != "decision" or r.get("tail"):
                continue
            if r.get("oracle_hit") not in ("eagle", "suffix"):
                continue
            if r.get("eagle_p") is None or r.get("suffix_p") is None:
                continue  # decisive AND contested: a real 2-way choice
            ep, sp = float(r["eagle_p"]), float(r["suffix_p"])
            cnt, tot = f(r.get("suffix_count")), f(r.get("suffix_total"))
            ml = f(r.get("match_len"))
            sc = f(r.get("suffix_score"))
            agr = 1.0 if r.get("agreement") else 0.0
            rows.append(dict(
                eagle_p=ep, suffix_p=sp, margin=sp - ep, depth=float(r["depth"]),
                log_count=math.log1p(cnt), log_total=math.log1p(tot),
                has_count=1.0 if r.get("suffix_count") is not None else 0.0,
                match_len=ml, suffix_score=sc, agreement=agr,
            ))
            rids.append(r.get("rid", "?"))
            y.append(1 if r["oracle_hit"] == "suffix" else 0)
    return rows, np.asarray(rids), np.asarray(y, int)


FEATSETS = {
    "LR[probs]": ["eagle_p", "suffix_p"],
    "LR[+depth]": ["eagle_p", "suffix_p", "depth"],
    "LR[+all]": ["eagle_p", "suffix_p", "depth", "log_count", "log_total",
                 "has_count", "match_len", "suffix_score", "agreement"],
}
ALL_FEATS = FEATSETS["LR[+all]"]


def matrix(rows, feats):
    return np.array([[r[k] for k in feats] for r in rows], float)


def cv_eval(rows, rids, y, feats, model="lr", n_splits=5):
    X = matrix(rows, feats)
    groups = rids
    ng = len(set(groups.tolist()))
    splits = min(n_splits, ng)
    gkf = GroupKFold(n_splits=splits)
    accs, aucs, coefs = [], [], []
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if model == "lr":
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, y[tr])
            coefs.append(clf.coef_[0])
        else:
            clf = GBDT(**_GBDT_KW)
            clf.fit(X[tr], y[tr])  # trees: raw scale
            Xte = X[te]
        prob = clf.predict_proba(Xte)[:, 1]
        accs.append(float(np.mean((prob > 0.5).astype(int) == y[te])))
        aucs.append(roc_auc_score(y[te], prob))
    return (float(np.mean(accs)), float(np.mean(aucs)),
            np.mean(coefs, axis=0) if coefs else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model-label", default="EAGLE3")
    args = ap.parse_args()
    D = Path(args.dir)
    rows, rids, y = load(D / "decisions_select1_oracle.jsonl")
    n = len(y)
    print(f"decisive+contested n={n}  suffix-right={y.mean():.3f}  "
          f"unique rids={len(set(rids.tolist()))}")

    results = []  # (label, acc, auc)
    results.append(("always-suffix", float(max(y.mean(), 1 - y.mean())), float("nan")))
    sp = np.array([r["suffix_p"] for r in rows]); ep = np.array([r["eagle_p"] for r in rows])
    raw_pick = (sp > ep).astype(int)
    results.append(("raw", float(np.mean(raw_pick == y)), roc_auc_score(y, sp - ep)))

    lr_all_coef = None
    for lab, feats in FEATSETS.items():
        acc, auc, coef = cv_eval(rows, rids, y, feats, "lr")
        results.append((lab, acc, auc))
        if lab == "LR[+all]":
            lr_all_coef = coef
    gacc, gauc, _ = cv_eval(rows, rids, y, ALL_FEATS, "gbdt")
    results.append(("GBDT[all]", gacc, gauc))
    results.append(("oracle", 1.0, 1.0))

    print(f"  {'rule':14s} {'sel.acc':>8s} {'AUC':>7s}")
    for lab, a, u in results:
        us = f"{u:7.3f}" if u == u else f"{'   -':>7s}"
        print(f"  {lab:14s} {a:8.3f} {us}")
    if lr_all_coef is not None:
        order = np.argsort(-np.abs(lr_all_coef))
        print("  LR[+all] standardized |coef| (desc):")
        for i in order:
            print(f"    {ALL_FEATS[i]:14s} {lr_all_coef[i]:+.3f}")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    labs = [r[0] for r in results]
    accs = [r[1] for r in results]
    aucs = [r[2] for r in results]
    colors = ["#999", "#1f77b4", "#2ca02c", "#17becf", "#9467bd", "#8c564b", "#e0b400"]
    xb = np.arange(len(labs))
    ax1.bar(xb, accs, color=colors[:len(labs)])
    for i, (a, u) in enumerate(zip(accs, aucs)):
        ax1.text(i, a + 0.006, f"{a:.3f}", ha="center", fontsize=8)
        if u == u and labs[i] not in ("oracle", "always-suffix"):
            ax1.text(i, 0.03, f"AUC\n{u:.3f}", ha="center", fontsize=7, color="white")
    ax1.axhline(results[0][1], color="k", ls=":", lw=0.9,
                label=f"always-suffix={results[0][1]:.3f}")
    ax1.axhline(results[1][1], color="#1f77b4", ls="--", lw=0.9,
                label=f"raw={results[1][1]:.3f}")
    ax1.set_xticks(xb); ax1.set_xticklabels(labs, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("held-out selection accuracy (GroupKFold by rid)")
    ax1.set_ylim(0, 1.02); ax1.grid(axis="y", alpha=0.3); ax1.legend(fontsize=8)
    ax1.set_title("Joint discriminator vs raw — does adding features help?")

    if lr_all_coef is not None:
        order = np.argsort(np.abs(lr_all_coef))
        yb = np.arange(len(ALL_FEATS))
        cols = ["#2ca02c" if lr_all_coef[i] > 0 else "#d62728" for i in order]
        ax2.barh(yb, [lr_all_coef[i] for i in order], color=cols)
        ax2.set_yticks(yb); ax2.set_yticklabels([ALL_FEATS[i] for i in order], fontsize=8)
        ax2.axvline(0, color="k", lw=0.8)
        ax2.set_xlabel("LR[+all] standardized coefficient\n(+ = favors suffix)")
        ax2.set_title("Feature contribution to the joint decision")
        ax2.grid(axis="x", alpha=0.3)
    fig.suptitle(f"O4 discriminator (offline, decisive set) — {args.model_label} "
                 f"(fair/replay-all)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = D / "figures" / "o4_discriminator.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
