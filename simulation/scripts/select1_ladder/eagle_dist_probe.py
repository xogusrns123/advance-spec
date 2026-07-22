"""Does eagle DISTRIBUTIONAL confidence break the confident-inversion wall?

Consumes an oracle decision log captured with --log-eagle-dist (fields
eagle_entropy, eagle_top2_margin, p_eagle_eagle, p_eagle_suffix present).
Everything is on the DECISIVE, accept-conditioned set with the counterfactual
all-candidate labels (token==gt), same population as irreducible_cases.py.

Three questions, answered head-to-head vs the logged-feature baseline:
  Q1  per-proposer OOF AUC(features -> own correctness): does [+entropy,
      +top2_margin,(+p_eagle_suffix)] beat [prob, depth]? (is the new signal
      informative at all)
  Q2  decisive selacc ceiling: joint-GBM(logged) vs joint-GBM(logged+dist).
      Does the Bayes ceiling rise above the logged-feature value
      (14B ~0.790 / 27B ~0.907)? (does the wall move)
  Q3  region-deadness: in the CONFIDENT-inversion region (raw argmax confidently
      wrong, margin>=0.2), AUC of the NEW features -> losing-but-correct proposer.
      Logged features all sit at ~0.5 there; if entropy/margin/p_eagle_suffix
      clear 0.5, THAT is the signal that closes the gap.

Run (host, sklearn ok):
  python3 simulation/scripts/select1_ladder/eagle_dist_probe.py --dir <capture_dir>
"""
import argparse, json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DIST_FIELDS = ("eagle_entropy", "eagle_top2_margin", "p_eagle_eagle", "p_eagle_suffix")
CONF_MARGIN = 0.20
HIT = ("eagle", "suffix", "both")


def load_decisive(path):
    """Accept-conditioned decisive rows (0<|hits|<|avail|), with all features."""
    chains = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            chains[(o["rid"], o["decode_step"])].append(o)
    for k in chains:
        chains[k].sort(key=lambda r: r["depth"])
    rows = []
    have_dist = 0
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            et, st = r.get("eagle_token"), r.get("suffix_token")
            ep, sp = r.get("eagle_p"), r.get("suffix_p")
            avail = [nm for nm, tk, pr in (("eagle", et, ep), ("suffix", st, sp))
                     if tk is not None and pr is not None]
            hits = [nm for nm, tk in (("eagle", et), ("suffix", st))
                    if tk is not None and gt is not None and tk == gt]
            if 0 < len(hits) < len(avail):
                d = {"rid": rid, "depth": int(r["depth"]), "avail": avail,
                     "hits": set(hits), "eagle_p": ep, "suffix_p": sp,
                     "match_len": r.get("match_len"), "count": r.get("suffix_count")}
                for f in DIST_FIELDS:
                    d[f] = r.get(f)
                if r.get("eagle_entropy") is not None:
                    have_dist += 1
                rows.append(d)
            if gt is not None and len(hits) == 0:
                alive = False
    return rows, have_dist


def oof(X, y, g):
    pred = np.full(len(y), float(y.mean()) if len(y) else 0.0)
    if len(y) < 40 or len(set(y)) < 2:
        return pred
    for tr, te in GroupKFold(min(5, len(set(g)))).split(X, y, g):
        if len(set(y[tr])) < 2:
            pred[te] = y[tr].mean(); continue
        m = HGB(max_depth=3, max_iter=200, learning_rate=0.06,
                l2_regularization=1.0).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def fv(e, feats):
    out = []
    for f in feats:
        if f == "depth":
            out.append(float(e["depth"]))
        elif f == "lcnt":
            out.append(math.log1p(e["count"]) if e.get("count") is not None else np.nan)
        elif f in ("eagle_p", "suffix_p", "match_len"):
            v = e.get(f); out.append(float(v) if v is not None else np.nan)
        else:  # dist fields
            v = e.get(f); out.append(float(v) if v is not None else np.nan)
    return out


def per_proposer(rows, nm, base_feats, dist_feats):
    idx = [i for i, e in enumerate(rows) if nm in e["avail"]]
    if not idx:
        return None
    y = np.array([1 if nm in rows[i]["hits"] else 0 for i in idx])
    g = np.array([rows[i]["rid"] for i in idx])
    Xb = np.array([fv(rows[i], base_feats) for i in idx], float)
    Xd = np.array([fv(rows[i], base_feats + dist_feats) for i in idx], float)
    ib, idd = oof(Xb, y, g), oof(Xd, y, g)
    aucb = roc_auc_score(y, ib) if len(set(y)) == 2 else float("nan")
    aucd = roc_auc_score(y, idd) if len(set(y)) == 2 else float("nan")
    return dict(idx=idx, y=y, base=ib, dist=idd, aucb=aucb, aucd=aucd, n=len(y))


def selacc(rows, pj):
    ok = 0
    for i, e in enumerate(rows):
        pick = max(e["avail"], key=lambda nm: pj[nm][i] if not np.isnan(pj[nm][i]) else -1)
        ok += 1 if pick in e["hits"] else 0
    return ok / max(len(rows), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--log", default="decisions_select1_oracle.jsonl")
    args = ap.parse_args()
    path = str(Path(args.dir) / args.log)
    rows, have_dist = load_decisive(path)
    N = len(rows)
    print(f"decisive rows={N}   rows with eagle-dist fields={have_dist} "
          f"({100*have_dist/max(N,1):.0f}%)")
    if N < 30:
        print("too few decisive rows; abort"); return
    if have_dist == 0:
        print("NO eagle-dist fields present — was the log captured with "
              "--log-eagle-dist? aborting."); return

    EBASE, SBASE = ["eagle_p", "depth"], ["suffix_p", "depth", "match_len", "lcnt"]
    EDIST = ["eagle_entropy", "eagle_top2_margin", "p_eagle_suffix"]
    SDIST = ["p_eagle_suffix", "eagle_entropy"]  # eagle's read on the suffix token

    print("\nQ1  per-proposer OOF AUC(features -> own correctness):")
    N2 = {}
    for nm, base, dist in (("eagle", EBASE, EDIST), ("suffix", SBASE, SDIST)):
        r = per_proposer(rows, nm, base, dist)
        N2[nm] = r
        if r:
            print(f"   {nm:7} logged={r['aucb']:.3f}  +eagle-dist={r['aucd']:.3f}  "
                  f"(delta {r['aucd']-r['aucb']:+.3f}, n={r['n']})")

    # Q2 joint selacc ladder: raw / joint(logged) / joint(logged+dist) / oracle
    print("\nQ2  decisive selacc ceiling:")
    raw = selacc(rows, {nm: np.array([e[nm + "_p"] if e.get(nm + "_p") is not None
                                      else -1 for e in rows], float)
                        for nm in ("eagle", "suffix")})
    pj_log = {nm: np.full(N, np.nan) for nm in ("eagle", "suffix")}
    pj_dist = {nm: np.full(N, np.nan) for nm in ("eagle", "suffix")}
    for nm, base, dist in (("eagle", EBASE, EDIST), ("suffix", SBASE, SDIST)):
        r = N2[nm]
        if not r:
            continue
        for j, i in enumerate(r["idx"]):
            pj_log[nm][i] = r["base"][j]
            pj_dist[nm][i] = r["dist"][j]
    sa_log, sa_dist = selacc(rows, pj_log), selacc(rows, pj_dist)
    print(f"   raw (argmax prob)      = {raw:.4f}")
    print(f"   joint-GBM (logged)     = {sa_log:.4f}  ({sa_log-raw:+.4f} vs raw)")
    print(f"   joint-GBM (+eagle-dist)= {sa_dist:.4f}  ({sa_dist-sa_log:+.4f} vs logged)  <- wall test")
    print(f"   oracle                 = 1.0000")

    # Q3 confident-inversion region-deadness for the NEW features
    print("\nQ3  confident-inversion region (raw confidently wrong, margin>=0.2): "
          "AUC of NEW features -> losing-but-correct proposer")
    for nm in ("eagle", "suffix"):
        sub_y, sub = [], {f: [] for f in DIST_FIELDS}
        for e in rows:
            w = max(e["avail"], key=lambda k: e[k + "_p"])
            if w in e["hits"] or nm not in e["avail"]:
                continue
            bestc = max(e[k + "_p"] for k in e["hits"])
            if e[w + "_p"] - bestc < CONF_MARGIN:
                continue
            sub_y.append(1 if nm in e["hits"] else 0)
            for f in DIST_FIELDS:
                sub[f].append(e.get(f))
        y = np.array(sub_y)
        if len(y) < 30 or len(set(y)) < 2:
            print(f"   {nm:7} (n={len(y)}, degenerate)"); continue
        parts = []
        for f in DIST_FIELDS:
            v = np.array([np.nan if x is None else x for x in sub[f]], float)
            ok = ~np.isnan(v)
            if ok.sum() < 30 or len(set(y[ok])) < 2:
                parts.append(f"{f}=na"); continue
            parts.append(f"{f}={roc_auc_score(y[ok], v[ok]):.3f}")
        print(f"   {nm:7} n={len(y)} P(right)={y.mean():.3f}  AUC: " + ", ".join(parts))
    print("\n  READ: Q2 +eagle-dist delta > ~+0.01 or Q3 AUC clearly >0.55 => the "
          "wall moves and a re-served eagle-dist selector should close more of the gap.")


if __name__ == "__main__":
    main()
