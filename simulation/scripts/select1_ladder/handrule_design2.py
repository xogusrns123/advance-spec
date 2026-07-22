"""Round 2 transparent rule search — push 14B above raw more clearly. Adds a 2D linear
boundary, a 3-clause combo (margin|count|match_len), and a SHALLOW DECISION TREE
(depth 2-3, OOF) whose learned thresholds are PRINTED as a readable if-then rule (the
transparent hand-rule with more features). Metric = decisive selacc vs raw/bayes/oracle.

Run (host): python3 simulation/scripts/select1_ladder/handrule_design2.py
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_count import ROOT, load_chains, loopy_rids  # noqa: E402
from handrule_design import collect, selacc  # noqa: E402

CELLS = {
    "qwen3_14b_2way": ("qwen3_14b_ar", "eagle_token", "eagle_p"),
    "qwen35_27b_2way": ("qwen35_27b_ar", "eagle_token", "eagle_p"),
}
FEATS = ["ep", "sp", "logn", "ml", "dep"]


def oof_tree(X, y, g, depth):
    pred = np.full(len(y), y.mean())
    gk = GroupKFold(min(5, len(set(g.tolist()))))
    for tr, te in gk.split(X, y, g):
        if len(set(y[tr].tolist())) < 2:
            pred[te] = y[tr].mean(); continue
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=200,
                                   random_state=0).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def main():
    for tag, (dirname, mtok, mpk) in CELLS.items():
        d = f"{ROOT}/{dirname}"
        chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
        bad = loopy_rids(d)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        D = collect(chains, mtok, mpk)
        ep, sp, n, ml, dep, sy, g = (D["ep"], D["sp"], D["n"], D["ml"], D["dep"],
                                     D["sy"], D["g"])
        N = len(sy); logn = np.log1p(n)
        raw = selacc(sp > ep, sy)
        Xfull = np.c_[ep, sp, logn, ml, dep]
        # bayes ceiling
        pj = np.full(N, sy.mean())
        for tr, te in GroupKFold(min(5, len(set(g.tolist())))).split(Xful := Xfull, sy, g):
            if len(set(sy[tr].tolist())) < 2:
                pj[te] = sy[tr].mean(); continue
            pj[te] = HGB(max_depth=3, max_iter=200, learning_rate=0.06,
                         l2_regularization=1.0).fit(Xfull[tr], sy[tr]).predict_proba(Xfull[te])[:, 1]
        bayes = selacc(pj > 0.5, sy)
        print(f"\n{'='*66}\n{tag}  N={N}  raw={raw:.4f}  bayes={bayes:.4f}  oracle=1.0")

        # 2D linear boundary sp > a*ep + b
        bestlin = (-1, None)
        for a in np.arange(0.4, 2.01, 0.1):
            for b in np.arange(-0.25, 0.36, 0.05):
                acc = selacc(sp > a * ep + b, sy)
                if acc > bestlin[0]:
                    bestlin = (acc, (round(a, 2), round(b, 2)))
        print(f"  linear  sp>a*ep+b           selacc={bestlin[0]:.4f}  (a,b)={bestlin[1]}")

        # 3-clause combo: margin | count | match_len
        best3 = (-1, None)
        for M in (.1, .15, .2, .25, .3):
            for C in (5, 8, 12, 20, 10**9):
                for L in (2, 3, 4, 10**9):
                    acc = selacc((sp > ep + M) | ((n >= C) & (sp > ep)) |
                                 ((ml >= L) & (sp > ep)), sy)
                    if acc > best3[0]:
                        best3 = (acc, (M, C, L))
        print(f"  combo3 margin|count|mlen     selacc={best3[0]:.4f}  (M,C,L)={best3[1]}")

        # shallow decision trees (transparent), OOF
        for depth in (2, 3):
            for feats, X in (("ep,sp", np.c_[ep, sp]),
                             ("all", Xfull)):
                acc = selacc(oof_tree(X, sy, g, depth) > 0.5, sy)
                print(f"  tree depth={depth} feats={feats:6s}  selacc={acc:.4f}"
                      f"  {'BEATS raw' if acc>raw+1e-4 else ''}")
        # print the readable depth-3 all-feature tree (full-data fit)
        t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200,
                                   random_state=0).fit(Xfull, sy)
        print("  --- depth-3 tree (full fit, leaf>0.5 => pick suffix) ---")
        print(export_text(t, feature_names=FEATS, show_weights=False,
                          max_depth=3).rstrip())


if __name__ == "__main__":
    main()
