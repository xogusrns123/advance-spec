"""Candidate A — leverage-weighted selection training (OFFLINE block-anchored test).

WHY: chain select-1 total selacc is INFO-CAPPED, but MAT = sum_d leverage(d)*survival(d).
A miss at depth d forfeits oracle's expected remaining chain = leverage(d), which falls
~7.6->1 over depth while raw miss-rate is ~flat. So MAT loss concentrates SHALLOW. Refitting
the selector with sample_weight = leverage(d) REALLOCATES the fixed (capped) accuracy budget
toward high-leverage shallow positions -> should raise block-anchored MAT while *lowering*
overall selacc slightly (sacrifices low-leverage deep accuracy). This is the test of that.

Compares, per cell, weight schemes {none, leverage(d), 1/(d+1), maxd-d} x selectors
{gbm prob+depth, gbm prob+depth+suffix, logistic prob+depth} on GroupKFold-by-rid OOF:
  selacc (overall) | selacc d<=2 (high-leverage) | selacc d>=6 (low-leverage) | MAT (block-anchored)
raw + oracle shown as reference. Weights are normalized to mean 1 per proposer so the l2
regularization strength stays comparable across schemes (otherwise leverage ~inflates n_eff).

Run: python3 simulation/scripts/select1_ladder/leverage_weighted.py
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

ROOT = "simulation/results/chain_hybrid_perdepth"
CELLS = [
 dict(key="14B 2-way", dir="qwen3_14b_ar",
      props=[("EAGLE3","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
 dict(key="27B 2-way", dir="qwen35_27b_ar",
      props=[("MTP","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
 dict(key="8B 3-way", dir="qwen3_8b_dflash_e3_ceiling20",
      props=[("DFlash","eagle_token","eagle_p"),("EAGLE3","e3_token","e3_p"),("suffix","suffix_token","suffix_p")]),
 dict(key="27B 3-way", dir="qwen35_27b_3way_real_full",
      props=[("MTP","eagle_token","eagle_p"),("DFlash","dflash_token","dflash_p"),("suffix","suffix_token","suffix_p")]),
]

def load_blocks(path, props):
    raw = defaultdict(list)
    for line in open(path):
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            raw[(o["rid"], o["decode_step"])].append(o)
    blocks = {}
    for k, rs in raw.items():
        rs.sort(key=lambda r: r["depth"]); pos = []
        for r in rs:
            e = {"depth": r["depth"], "gt": r.get("gt_token"), "rid": k[0], "P": {}}
            for nm, tk, pk in props:
                t = r.get(tk); p = r.get(pk)
                if t is not None and p is not None:
                    e["P"][nm] = dict(tok=t, prob=float(p), mlen=float(r.get("match_len") or 0),
                                      lcnt=math.log1p(float(r.get("suffix_count") or 0)))
            pos.append(e)
        blocks[k] = pos
    return blocks

def loopy(d):
    dd = Path(d); gtf = dd / "gt_tokens.jsonl"; reqs = {}
    for line in open(dd / "decisions_select1_oracle.jsonl"):
        o = json.loads(line)
        if o.get("type") == "req": reqs[o["rid"]] = tuple(o["input_ids"])
    bad = set()
    if gtf.exists():
        gt = {}
        for line in open(gtf):
            r = json.loads(line); gt[tuple(r["input_ids"])] = r["output_ids"]
        for rid, ids in reqs.items():
            out = gt.get(ids)
            if not out or len(out) < 5: continue
            g = [tuple(out[i:i+4]) for i in range(len(out)-3)]
            if len(set(g))/max(len(g),1) < 0.5: bad.add(rid)
    return bad

def leverage_curve(blocks, maxd=64):
    """leverage(d) = E[oracle remaining chain | oracle alive at d], from the gt trajectory."""
    lev_sum = np.zeros(maxd); reach = np.zeros(maxd)
    for pos in blocks.values():
        D = len(pos)
        for i, e in enumerate(pos):
            gt = e["gt"]
            if gt is None or not e["P"]: D = i; break
            if not any(v["tok"] == gt for v in e["P"].values()): D = i; break
        for i in range(min(D, maxd)):
            reach[i] += 1; lev_sum[i] += (D - i)
    lev = np.where(reach > 0, lev_sum / np.maximum(reach, 1), 0.0)
    return lev

def fv(e, nm, featset):
    P = e["P"][nm]; v = [P["prob"], e["depth"]]
    if featset == "prob_depth_suffix" and nm == "suffix":
        v += [P["mlen"], P["lcnt"]]
    return v

def weight_of(depth, scheme, lev, maxd):
    if scheme == "none":     return 1.0
    if scheme == "leverage": return float(lev[depth]) if depth < len(lev) and lev[depth] > 0 else 1e-3
    if scheme == "inv":      return 1.0 / (depth + 1)
    if scheme == "linear":   return float(max(maxd - depth, 1))
    raise ValueError(scheme)

def fit_predict(method, Xfull, prob, y, w, fit, pred):
    ytr = y[fit]; wtr = w[fit]
    if len(set(ytr)) < 2:
        return np.full(len(pred), ytr.mean())
    if method == "logistic":
        Xt = Xfull[fit]; m = Xt.mean(0); s = Xt.std(0) + 1e-9
        clf = LogisticRegression(max_iter=1000).fit((Xt - m) / s, ytr, sample_weight=wtr)
        return clf.predict_proba((Xfull[pred] - m) / s)[:, 1]
    if method == "gbm":
        clf = HGB(max_depth=3, max_iter=150, learning_rate=0.08, l2_regularization=1.0)
        clf.fit(Xfull[fit], ytr, sample_weight=wtr)
        return clf.predict_proba(Xfull[pred])[:, 1]
    raise ValueError(method)

def picks(alive, props, featset, method, scheme, lev, maxd):
    names = [p[0] for p in props]
    R = {nm: {"X":[], "p":[], "y":[], "g":[], "d":[], "dep":[]} for nm in names}
    for did, e in alive:
        for nm in e["P"]:
            R[nm]["X"].append(fv(e, nm, featset)); R[nm]["p"].append(e["P"][nm]["prob"])
            R[nm]["y"].append(1 if e["P"][nm]["tok"] == e["gt"] else 0)
            R[nm]["g"].append(e["rid"]); R[nm]["d"].append(did); R[nm]["dep"].append(e["depth"])
    P_of = defaultdict(dict)
    for nm in names:
        X = np.array(R[nm]["X"], float); p = np.array(R[nm]["p"], float)
        y = np.array(R[nm]["y"]); g = np.array(R[nm]["g"]); ds = R[nm]["d"]
        w = np.array([weight_of(dp, scheme, lev, maxd) for dp in R[nm]["dep"]], float)
        w = w * (len(w) / w.sum()) if w.sum() > 0 else np.ones_like(w)  # normalize mean->1
        if len(y) < 10 or len(set(y)) < 2:
            for d, yy in zip(ds, y): P_of[d][nm] = float(yy)
            continue
        prd = np.zeros(len(y))
        for tr, te in GroupKFold(min(5, len(set(g)))).split(X, y, g):
            prd[te] = fit_predict(method, X, p, y, w, tr, te)
        for d, pp in zip(ds, prd): P_of[d][nm] = float(pp)
    return {did: (max(P_of[did], key=lambda nm: P_of[did][nm]) if P_of[did] else None) for did, _ in alive}

def selacc(alive, pk, lo=None, hi=None):
    rows = [(d, e) for d, e in alive if (lo is None or e["depth"] >= lo) and (hi is None or e["depth"] <= hi)]
    if not rows: return float("nan")
    return np.mean([1.0 if (pk.get(d) is not None and e["P"][pk[d]]["tok"] == e["gt"]) else 0.0
                    for d, e in rows])

def run_length(blocks, pick_fn):
    tot = 0; n = 0
    for k, pos in blocks.items():
        n += 1; run = 0
        for e in pos:
            gt = e["gt"]
            if gt is None or not e["P"]: break
            av = list(e["P"]); hits = [nm for nm in av if e["P"][nm]["tok"] == gt]
            if len(hits) == len(av): run += 1; continue
            if len(hits) == 0: break
            nm = pick_fn((k[0], k[1], e["depth"]), e)
            if nm is not None and e["P"][nm]["tok"] == gt: run += 1
            else: break
        tot += run
    return tot / max(n, 1)

SCHEMES = ["none", "leverage", "inv", "linear"]
SELECTORS = [("gbm", "prob_depth"), ("gbm", "prob_depth_suffix"), ("logistic", "prob_depth")]

def run_cell(cell):
    d = f"{ROOT}/{cell['dir']}"; props = cell["props"]; names = [p[0] for p in props]
    blocks = load_blocks(f"{d}/decisions_select1_oracle.jsonl", props)
    bad = loopy(d); blocks = {k: v for k, v in blocks.items() if k[0] not in bad}
    lev = leverage_curve(blocks); maxd = max((e["depth"] for pos in blocks.values() for e in pos), default=16) + 1
    alive = []
    for k, pos in blocks.items():
        al = True
        for e in pos:
            if not al: break
            if not e["P"]: continue
            av = list(e["P"]); hits = [nm for nm in av if e["P"][nm]["tok"] == e["gt"]]
            if 0 < len(hits) < len(av): alive.append(((k[0], k[1], e["depth"]), e))
            if e["gt"] is not None and len(hits) == 0: al = False

    print(f"\n{'='*108}\n### {cell['key']}   blocks={len(blocks)}  decisive={len(alive)}")
    print("  leverage(d 0..9): " + " ".join(f"{x:.2f}" for x in lev[:10]))
    # references
    pick_raw = lambda did, e: max(e["P"], key=lambda nm: e["P"][nm]["prob"]) if e["P"] else None
    raw_pk = {did: pick_raw(did, e) for did, e in alive}
    print(f"  {'reference':36} {'selacc':>8} {'sel d<=2':>9} {'sel d>=6':>9} {'MAT':>8}")
    print(f"  {'raw (prob argmax)':36} {selacc(alive,raw_pk):>8.4f} {selacc(alive,raw_pk,lo=0,hi=2):>9.4f} "
          f"{selacc(alive,raw_pk,lo=6):>9.4f} {run_length(blocks,pick_raw):>8.3f}")
    orc = run_length(blocks, lambda did,e: next((nm for nm in e["P"] if e["P"][nm]["tok"]==e["gt"]), None))
    print(f"  {'oracle':36} {1.0:>8.4f} {1.0:>9.4f} {1.0:>9.4f} {orc:>8.3f}")

    for method, fs in SELECTORS:
        tag = f"{method}({fs.replace('prob_depth','p+d').replace('_suffix','+sfx')})"
        print(f"\n  -- {tag} --   {'scheme':14} {'selacc':>8} {'sel d<=2':>9} {'sel d>=6':>9} {'MAT':>8}  {'dMAT':>7}")
        base_mat = None
        for sch in SCHEMES:
            pk = picks(alive, props, fs, method, sch, lev, maxd)
            mat = run_length(blocks, lambda did, e, _p=pk: _p.get(did))
            if sch == "none": base_mat = mat
            dm = mat - base_mat
            print(f"  {'':14}{'':14}   {sch:14} {selacc(alive,pk):>8.4f} {selacc(alive,pk,lo=0,hi=2):>9.4f} "
                  f"{selacc(alive,pk,lo=6):>9.4f} {mat:>8.3f}  {dm:>+7.3f}")

if __name__ == "__main__":
    for c in CELLS:
        try: run_cell(c)
        except Exception as ex:
            import traceback; print(f"ERR {c['key']}: {ex}"); traceback.print_exc()
