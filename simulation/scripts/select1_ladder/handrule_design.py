"""Design a TRANSPARENT hand-rule for select-1 (to serve as a new arm). Tests rule
families incl. the user's "shift suffix 0.5 right ONLY when count is high" and more
features (eagle-weakness gate, margin, match_len, combos). Metric = decisive selacc
(offline, reliable for RANKING; higher selacc -> fewer chain deaths -> higher MAT).
Compares each family's best to raw, bayes-GBM (joint ceiling) and oracle(=1).

Decisive-contested 2-way features: eagle_p, suffix_p, depth, match_len, count n
(suffix_total), c (suffix_count). Pick the winning transparent rule -> serve it.

Run (host): python3 simulation/scripts/select1_ladder/handrule_design.py
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_count import ROOT, load_chains, loopy_rids  # noqa: E402

CELLS = {
    "qwen3_14b_2way": ("qwen3_14b_ar", "eagle_token", "eagle_p"),
    "qwen35_27b_2way": ("qwen35_27b_ar", "eagle_token", "eagle_p"),
}


def collect(chains, mtok, mpk):
    ep, sp, dep, ml, n, c, sy, g = [], [], [], [], [], [], [], []
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
                ep.append(float(mpv)); sp.append(float(spv)); dep.append(int(r["depth"]))
                ml.append(float(r.get("match_len") or 0))
                n.append(float(r.get("suffix_total") or 0))
                c.append(float(r.get("suffix_count") or 0))
                sy.append(1 if hits[0] == "s" else 0); g.append(rid)
            if gt is not None and (e_av or s_av) and len(hits) == 0:
                alive = False
    return dict(ep=np.array(ep), sp=np.array(sp), dep=np.array(dep),
                ml=np.array(ml), n=np.array(n), c=np.array(c),
                sy=np.array(sy), g=np.array(g))


def selacc(pick_s, sy):
    return float(np.where(pick_s, sy, 1 - sy).mean())


def main():
    for tag, (dirname, mtok, mpk) in CELLS.items():
        d = f"{ROOT}/{dirname}"
        chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
        bad = loopy_rids(d)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        D = collect(chains, mtok, mpk)
        ep, sp, n, c, ml, dep, sy, g = (D["ep"], D["sp"], D["n"], D["c"], D["ml"],
                                        D["dep"], D["sy"], D["g"])
        N = len(sy)
        atom = np.round(sp, 4) == 0.5
        print(f"\n{'='*68}\n{tag}  decisive N={N}  (0.5-atom {atom.sum()})  "
              f"suffix-right base={sy.mean():.3f}")

        raw = selacc(sp > ep, sy)
        # bayes-GBM joint ceiling (OOF)
        X = np.c_[ep, sp, dep, ml, np.log1p(n)]
        pj = np.full(N, sy.mean())
        for tr, te in GroupKFold(min(5, len(set(g.tolist())))).split(X, sy, g):
            if len(set(sy[tr].tolist())) < 2:
                pj[te] = sy[tr].mean(); continue
            pj[te] = HGB(max_depth=3, max_iter=200, learning_rate=0.06,
                         l2_regularization=1.0).fit(X[tr], sy[tr]).predict_proba(X[te])[:, 1]
        bayes = selacc(pj > 0.5, sy)
        print(f"  raw={raw:.4f}   bayes-GBM(full feats)={bayes:.4f}   oracle=1.0")

        best = {}

        def sweep(name, fn, params):
            b = (-1, None)
            for p in params:
                a = selacc(fn(p), sy)
                if a > b[0]:
                    b = (a, p)
            best[name] = b
            tag_p = f"={b[1]}" if not isinstance(b[1], tuple) else f"={b[1]}"
            print(f"  {name:34s} best selacc={b[0]:.4f}  (param{tag_p})  "
                  f"{'BEATS raw' if b[0] > raw + 1e-4 else ''}")

        # R1 margin: suffix must beat eagle by M
        sweep("margin: sp>ep+M", lambda M: sp > ep + M,
              [0, .05, .1, .15, .2, .25, .3])
        # R2 eagle-weakness gate: suffix only if eagle weak
        sweep("eagle-gate: sp>ep & ep<T", lambda T: (sp > ep) & (ep < T),
              [.3, .4, .5, .6, .7, .8, 1.0])
        # R3 count-gate: suffix only if count high (n>=C)
        sweep("count-gate: sp>ep & n>=C", lambda C: (sp > ep) & (n >= C),
              [2, 3, 5, 8, 12, 20, 40])
        # R4 USER count-conditional SHIFT: boost 0.5-atom right only when count high
        sweep("count-shift@0.5: +D if n>=C", lambda pr: (
            np.where(atom & (n >= pr[1]), np.minimum(sp + pr[0], 1.0), sp) > ep),
              [(D_, C_) for D_ in (.1, .2, .3, .5) for C_ in (5, 10, 20, 40)])
        # R4b count-conditional shift on ALL suffix (not just 0.5)
        sweep("count-shift(all): +D if n>=C", lambda pr: (
            np.where(n >= pr[1], np.minimum(sp + pr[0], 1.0), sp) > ep),
              [(D_, C_) for D_ in (.1, .2, .3) for C_ in (5, 10, 20)])
        # R5 match_len gate
        sweep("mlen-gate: sp>ep & mlen>=L", lambda L: (sp > ep) & (ml >= L),
              [1, 2, 3, 4, 5])
        # R6 combo: eagle weak AND (count high OR strong suffix)
        sweep("combo: ep<T & (n>=C | sp>=Q)", lambda pr: (
            (ep < pr[0]) & ((n >= pr[1]) | (sp >= pr[2])) & (sp > ep)),
              [(T_, C_, Q_) for T_ in (.4, .5, .6) for C_ in (5, 12) for Q_ in (.66, .75, .8)])
        # R7 combo margin + count-shift: dominate by M, OR high-count beats eagle
        sweep("combo: sp>ep+M | (n>=C & sp>ep)", lambda pr: (
            (sp > ep + pr[0]) | ((n >= pr[1]) & (sp > ep))),
              [(M_, C_) for M_ in (.1, .15, .2) for C_ in (8, 12, 20)])

        bb = max(best.items(), key=lambda kv: kv[1][0])
        print(f"  -> BEST transparent rule: {bb[0]}  selacc={bb[1][0]:.4f} param={bb[1][1]}"
              f"  (raw {raw:.4f}, bayes {bayes:.4f})")


if __name__ == "__main__":
    main()
