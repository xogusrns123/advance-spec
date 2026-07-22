"""Confound-controlled feature-value study for CHAIN select-1.
Question: does a candidate feature add information about (proposer==gt) GIVEN the
features already used by the selector ([self_p, max_other, depth])? A feature that
is redundant/confounded gives ~0 OOF lift.

Candidates (only the principled, cheaply-available ones):
  n_agree   = # OTHER available proposers proposing the SAME token (3-way only;
              degenerate on 2-way decisive by definition). Ensemble concurrence,
              orthogonal to self-confidence.
  sfx_ev    = [match_len, log1p(suffix_total)] for the SUFFIX arm: the evidence
              strength behind the same count-ratio suffix_p. 0 for non-suffix arms.

Metrics:
  per-arm OOF AUC: base vs base+feat  (clean conditional info, no cross-arm mix)
  selector OOF selacc: argmax of P(==gt) over proposers, base vs +feat vs oracle
All GroupKFold-by-rid OOF on decisive-alive rows.
"""
import json, sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = "simulation/results/chain_hybrid_perdepth"
CELLS = {
    "14B 2-way": {"dir": "qwen3_14b_ar",
        "props": [("eagle3", "eagle_token", "eagle_p"), ("suffix", "suffix_token", "suffix_p")]},
    "27B 2-way": {"dir": "qwen35_27b_ar",
        "props": [("mtp", "eagle_token", "eagle_p"), ("suffix", "suffix_token", "suffix_p")]},
    "8B 3-way": {"dir": "qwen3_8b_dflash_e3_ceiling20",
        "props": [("dflash", "eagle_token", "eagle_p"), ("eagle3", "e3_token", "e3_p"),
                  ("suffix", "suffix_token", "suffix_p")]},
    "27B 3-way": {"dir": "qwen35_27b_3way_real_full",
        "props": [("mtp", "eagle_token", "eagle_p"), ("dflash", "dflash_token", "dflash_p"),
                  ("suffix", "suffix_token", "suffix_p")]},
}


def load_chains(path):
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
    return chains


def loopy_rids(d, thresh=0.5, n=4):
    dd = Path(d); gtf = dd / "gt_tokens.jsonl"
    if not gtf.exists():
        return set()
    reqs = {}
    for line in open(dd / "decisions_select1_oracle.jsonl"):
        o = json.loads(line)
        if o.get("type") == "req":
            reqs[o["rid"]] = tuple(o["input_ids"])
    gt = {}
    for line in open(gtf):
        r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
    bad = set()
    for rid, ids in reqs.items():
        out = gt.get(ids)
        if not out or len(out) < n + 1:
            continue
        g = [tuple(out[i:i + n]) for i in range(len(out) - n + 1)]
        if len(set(g)) / max(len(g), 1) < thresh:
            bad.add(rid)
    return bad


def collect(cell):
    d = f"{ROOT}/{cell['dir']}"
    chains = load_chains(f"{d}/decisions_select1_oracle.jsonl")
    bad = loopy_rids(d)
    chains = {k: v for k, v in chains.items() if k[0] not in bad}
    props = cell["props"]
    rows = []  # each: dict(decision, rid, name, self_p, max_other, depth, n_agree, mlen, ltot, is_sfx, y)
    for (rid, ds), rs in chains.items():
        alive = True
        for r in rs:
            if not alive:
                break
            gt = r.get("gt_token")
            toks = {p[0]: r.get(p[1]) for p in props}
            probs = {p[0]: (r.get(p[2]) if r.get(p[2]) is not None else None) for p in props}
            avail = [p[0] for p in props if toks[p[0]] is not None and probs[p[0]] is not None]
            hits = [nm for nm in avail if gt is not None and toks[nm] == gt]
            if 0 < len(hits) < len(avail):
                did = (rid, ds, r["depth"])
                for nm in avail:
                    others = [probs[q] for q in avail if q != nm]
                    n_agree = sum(1 for q in avail if q != nm and toks[q] == toks[nm])
                    is_sfx = 1.0 if nm == "suffix" else 0.0
                    mlen = float(r.get("match_len") or 0) if nm == "suffix" else 0.0
                    ltot = math.log1p(float(r.get("suffix_total") or 0)) if nm == "suffix" else 0.0
                    rows.append(dict(did=did, rid=rid, name=nm,
                                     self_p=float(probs[nm]),
                                     max_other=(max(others) if others else 0.0),
                                     depth=float(r["depth"]), n_agree=float(n_agree),
                                     mlen=mlen, ltot=ltot, is_sfx=is_sfx,
                                     y=1 if nm in hits else 0))
            if gt is not None and len(hits) == 0:
                alive = False
    return rows, [p[0] for p in props]


def oof_proba(X, y, groups, n_splits=5):
    X = np.asarray(X, float); y = np.asarray(y); groups = np.asarray(groups)
    pred = np.zeros(len(y))
    ng = len(set(groups))
    gkf = GroupKFold(n_splits=min(n_splits, ng))
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            pred[te] = y[tr].mean(); continue
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=120,
                                             learning_rate=0.08, l2_regularization=1.0)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return pred


def selacc_from_scores(rows, scores):
    by = defaultdict(list)
    for r, s in zip(rows, scores):
        by[r["did"]].append((s, r["y"]))
    hit = 0; n = 0
    for did, lst in by.items():
        n += 1
        hit += max(lst, key=lambda t: t[0])[1]
    return hit / max(n, 1)


def main():
    for name, cell in CELLS.items():
        rows, pnames = collect(cell)
        if not rows:
            print(f"\n### {name}: no rows"); continue
        y = [r["y"] for r in rows]; g = [r["rid"] for r in rows]
        three = len(pnames) >= 3
        n_dec = len(set(r["did"] for r in rows))
        print(f"\n### {name}  ({cell['dir']})  decisive={n_dec}  rows={len(rows)}  "
              f"{'3-way' if three else '2-way'}")

        # ---- selector-level selacc: raw / base / +agree(3way) / +sfxev / +both / oracle
        raw = [r["self_p"] for r in rows]
        base_X = [[r["self_p"], r["max_other"], r["depth"]] for r in rows]
        sets = {"base [p,max_other,depth]": base_X}
        if three:
            sets["+ n_agree"] = [x + [r["n_agree"]] for x, r in zip(base_X, rows)]
        sets["+ sfx_ev [mlen,ltot]"] = [x + [r["mlen"], r["ltot"]] for x, r in zip(base_X, rows)]
        if three:
            sets["+ both"] = [x + [r["n_agree"], r["mlen"], r["ltot"]] for x, r in zip(base_X, rows)]

        print(f"  {'selector':28} {'selacc':>8}")
        print(f"  {'raw argmax(self_p)':28} {selacc_from_scores(rows, raw):>8.4f}")
        base_sel = None
        for sname, X in sets.items():
            p = oof_proba(X, y, g)
            sa = selacc_from_scores(rows, p)
            if sname.startswith("base"):
                base_sel = sa
            tag = "" if base_sel is None or sname.startswith("base") else f"  (Δ {sa-base_sel:+.4f})"
            print(f"  {sname:28} {sa:>8.4f}{tag}")
        print(f"  {'oracle':28} {1.0:>8.4f}")

        # ---- clean per-feature conditional AUC (does feat add over its base?)
        # agreement: pooled AUC base[p,max_other,depth] vs +n_agree
        if three:
            ab = oof_proba(base_X, y, g)
            ap = oof_proba([x + [r["n_agree"]] for x, r in zip(base_X, rows)], y, g)
            print(f"  AUC(all arms)  base={roc_auc_score(y,ab):.4f}  "
                  f"+n_agree={roc_auc_score(y,ap):.4f}  (Δ {roc_auc_score(y,ap)-roc_auc_score(y,ab):+.4f})")
        # suffix-evidence: SUFFIX rows only, AUC[suffix_p] vs [suffix_p,mlen,ltot]
        si = [i for i, r in enumerate(rows) if r["name"] == "suffix"]
        if len(si) > 50:
            ys = [rows[i]["y"] for i in si]; gs = [rows[i]["rid"] for i in si]
            if len(set(ys)) == 2:
                b1 = oof_proba([[rows[i]["self_p"]] for i in si], ys, gs)
                b2 = oof_proba([[rows[i]["self_p"], rows[i]["mlen"], rows[i]["ltot"]] for i in si], ys, gs)
                print(f"  AUC(suffix arm, n={len(si)})  [sfx_p]={roc_auc_score(ys,b1):.4f}  "
                      f"+[mlen,ltot]={roc_auc_score(ys,b2):.4f}  (Δ {roc_auc_score(ys,b2)-roc_auc_score(ys,b1):+.4f})")


main()
