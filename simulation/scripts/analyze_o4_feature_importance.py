"""Rank the candidate discriminator features BEFORE wiring the serving arm.

The discriminator predicts P(suffix is the GT pick) on decisive+contested
decisions. This script enumerates every usable feature in the decision log
(excluding leakage: oracle_hit, gt_token, chosen, *_cal) plus a few engineered
ones, and ranks them two collinearity-robust ways:

  (1) UNIVARIATE AUC  -- each feature alone, |AUC-0.5| = standalone power
                         (rank-based, immune to collinearity)
  (2) FORWARD SELECTION -- greedily add the feature that most raises held-out
                         LR AUC (GroupKFold by rid); the increment column is the
                         NON-REDUNDANT contribution, so near-duplicates
                         (suffix_p vs count/total vs jeffrey) don't all score.

Output: a ranked table + figure (univariate power bars + forward-selection AUC
trajectory). No serving run -- this only decides the feature set.

Usage:
  python3 simulation/scripts/analyze_o4_feature_importance.py \
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

EPS = 1e-4


def fnum(x, d=0.0):
    return d if x is None else float(x)


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def load(path):
    feats, rids, y = [], [], []
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
                continue
            ep, sp = float(r["eagle_p"]), float(r["suffix_p"])
            cnt, tot = fnum(r.get("suffix_count")), fnum(r.get("suffix_total"))
            ml, scr = fnum(r.get("match_len")), fnum(r.get("suffix_score"))
            ds = fnum(r.get("decode_step"))
            agr = 1.0 if r.get("agreement") else 0.0
            jeff = (cnt + 0.5) / (tot + 1.0) if tot > 0 else sp
            feats.append(dict(
                # raw logged signals
                suffix_p=sp, eagle_p=ep, suffix_count=cnt, suffix_total=tot,
                suffix_score=scr, match_len=ml, depth=float(r["depth"]),
                decode_step=ds, agreement=agr,
                # engineered
                margin=sp - ep, log_count=math.log1p(cnt),
                log_total=math.log1p(tot), jeffrey_p=jeff,
                suffix_logit=logit(sp), eagle_logit=logit(ep),
                logodds_margin=logit(sp) - logit(ep),
            ))
            rids.append(r.get("rid", "?"))
            y.append(1 if r["oracle_hit"] == "suffix" else 0)
    names = list(feats[0].keys())
    X = np.array([[f[k] for k in names] for f in feats], float)
    return names, X, np.asarray(rids), np.asarray(y, int)


LOGGED = {"suffix_p", "eagle_p", "suffix_count", "suffix_total", "suffix_score",
          "match_len", "depth", "decode_step", "agreement"}


def cv_auc(X, y, groups, cols, n_splits=5):
    Xs = X[:, cols]
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups.tolist()))))
    aucs = []
    for tr, te in gkf.split(Xs, y, groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(Xs[tr])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(sc.transform(Xs[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(Xs[te]))[:, 1]))
    return float(np.mean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model-label", default="EAGLE3")
    args = ap.parse_args()
    D = Path(args.dir)
    names, X, rids, y = load(D / "decisions_select1_oracle.jsonl")
    n, nf = X.shape
    nonnull = {nm: float(np.mean(np.isfinite(X[:, i]))) for i, nm in enumerate(names)}
    print(f"decisive+contested n={n}  suffix-right={y.mean():.3f}  "
          f"unique rids={len(set(rids.tolist()))}  candidate features={nf}")

    # (1) univariate AUC (rank-based, sign tells direction)
    uni = {}
    for i, nm in enumerate(names):
        try:
            a = roc_auc_score(y, X[:, i])
        except Exception:
            a = 0.5
        uni[nm] = a
    print("\n(1) UNIVARIATE AUC  (|AUC-0.5| = standalone power; >0.5 favors suffix)")
    print(f"  {'feature':16s} {'AUC':>6s} {'power':>6s} {'kind':>9s}")
    for nm in sorted(names, key=lambda k: -abs(uni[k] - 0.5)):
        kind = "logged" if nm in LOGGED else "derived"
        print(f"  {nm:16s} {uni[nm]:6.3f} {abs(uni[nm]-0.5):6.3f} {kind:>9s}")

    # (2) forward selection on held-out LR AUC
    print("\n(2) FORWARD SELECTION  (held-out LR AUC, GroupKFold by rid)")
    remaining = list(range(nf))
    chosen, traj = [], []
    base = 0.5
    while remaining:
        best_i, best_a = None, -1
        for i in remaining:
            a = cv_auc(X, y, rids, chosen + [i])
            if a > best_a:
                best_a, best_i = a, i
        gain = best_a - (traj[-1][2] if traj else 0.5)
        chosen.append(best_i); remaining.remove(best_i)
        traj.append((len(chosen), names[best_i], best_a, gain))
        print(f"  +{names[best_i]:16s} AUC={best_a:.4f}  (+{gain:.4f})")
        if gain < 0.001 and len(chosen) >= 3:
            print("  ... further features add < 0.001 AUC (diminishing)")
            # keep going to show full trajectory but mark the knee
    knee = next((k for k, _, _, g in traj if g < 0.001 and k >= 3), len(traj))

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1]})
    order = sorted(names, key=lambda k: abs(uni[k] - 0.5))
    powers = [abs(uni[k] - 0.5) for k in order]
    cols = ["#1f77b4" if k in LOGGED else "#ff7f0e" for k in order]
    yb = np.arange(len(order))
    ax1.barh(yb, powers, color=cols)
    ax1.set_yticks(yb); ax1.set_yticklabels(order, fontsize=8)
    ax1.set_xlabel("univariate power  |AUC - 0.5|")
    ax1.set_title("(1) Standalone discriminative power per feature")
    ax1.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color="#1f77b4", label="logged"),
                        Patch(color="#ff7f0e", label="derived")], fontsize=8)

    ks = [t[0] for t in traj]; aucs = [t[2] for t in traj]
    ax2.plot(ks, aucs, "-o", color="#2ca02c", ms=4)
    for k, nm, a, g in traj:
        ax2.annotate(nm, (k, a), fontsize=6.5, rotation=35,
                     textcoords="offset points", xytext=(2, 4))
    ax2.axvline(knee, color="k", ls="--", lw=0.9, label=f"knee @ {knee} feats")
    ax2.axhline(0.788, color="#1f77b4", ls=":", lw=0.9, label="raw rule AUC=0.788")
    ax2.set_xlabel("# features (added greedily)")
    ax2.set_ylabel("held-out LR AUC")
    ax2.set_title("(2) Forward selection — non-redundant contribution")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle(f"O4 discriminator feature importance — {args.model_label} "
                 f"(fair/replay-all)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = D / "figures" / "o4_feature_importance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
