"""DECISIVE TEST: does a serving-causal HISTORY feature ADD information to the static
per-position GBM (prob+depth)? If yes, sequence-history is a NEW in-scope lever that pushes
the per-position info-ceiling; if not, it's redundant with prob+depth.

History feature (block-causal, serving-available): for proposer i at a position in block ds,
hist_i = empirical match-rate of proposer i (token==gt) over the PRIOR blocks (ds'<ds) of the
SAME request. = "how reliable has proposer i been so far this request." (Also a decayed/EWMA
variant.) No within-block leak; no cross-request leak.

Compares per-proposer one-vs-rest GBM under feature sets {prob+depth} vs {prob+depth+hist},
GroupKFold-by-rid OOF, reporting decisive selacc + block-anchored MAT. raw/oracle for reference.
Run: python3 simulation/scripts/select1_ladder/hedge_feature_test.py
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

ROOT = "simulation/results/chain_hybrid_perdepth"
CELLS = [
    ("27B 2-way", "qwen35_27b_ar", [("eagle","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
    ("14B 2-way", "qwen3_14b_ar",  [("eagle","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
]

def load(path, props):
    raw = defaultdict(list)
    for line in open(path):
        o = json.loads(line)
        if o.get("type") == "decision" and not o.get("tail"):
            raw[(o["rid"], o["decode_step"])].append(o)
    by_rid = defaultdict(list)
    for (rid, ds), rs in raw.items():
        rs.sort(key=lambda r: r["depth"]); pos = []
        for r in rs:
            e = {"depth": r["depth"], "gt": r.get("gt_token"), "rid": rid, "ds": ds, "P": {}}
            for nm, tk, pk in props:
                t = r.get(tk); p = r.get(pk)
                if t is not None and p is not None:
                    e["P"][nm] = dict(tok=t, prob=float(p))
            pos.append(e)
        by_rid[rid].append((ds, pos))
    for rid in by_rid:
        by_rid[rid].sort(key=lambda x: x[0])
    return by_rid

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

def attach_history(by_rid, names, delta=1.0, prior=0.5):
    """block-causal per-proposer match-rate over PRIOR blocks of the same rid (EWMA via delta)."""
    for rid, blocks in by_rid.items():
        hit = {nm: 0.0 for nm in names}; tot = {nm: 0.0 for nm in names}
        for ds, pos in blocks:
            for e in pos:                      # assign hist from PRIOR blocks (snapshot)
                e["hist"] = {nm: (hit[nm] / tot[nm] if tot[nm] > 0 else prior) for nm in names}
            for e in pos:                      # then update with THIS block (for next blocks)
                gt = e["gt"]
                if gt is None: continue
                for nm, v in e["P"].items():
                    hit[nm] = delta * hit[nm] + (1.0 if v["tok"] == gt else 0.0)
                    tot[nm] = delta * tot[nm] + 1.0

def flat(by_rid):
    out = []
    for rid, blocks in by_rid.items():
        for ds, pos in blocks:
            out.append((rid, ds, pos))
    return out

def fv(e, nm, featset):
    v = [e["P"][nm]["prob"], e["depth"]]
    if featset == "phist":
        v.append(e.get("hist", {}).get(nm, 0.5))
    return v

def picks_oof(alive, names, featset):
    R = {nm: {"X":[],"y":[],"g":[],"k":[]} for nm in names}
    for key, e in alive:
        for nm in e["P"]:
            R[nm]["X"].append(fv(e,nm,featset)); R[nm]["y"].append(1 if e["P"][nm]["tok"]==e["gt"] else 0)
            R[nm]["g"].append(e["rid"]); R[nm]["k"].append(key)
    P_of = defaultdict(dict)
    for nm in names:
        X=np.array(R[nm]["X"],float); y=np.array(R[nm]["y"]); g=np.array(R[nm]["g"]); ks=R[nm]["k"]
        if len(y)<10 or len(set(y))<2:
            for k,yy in zip(ks,y): P_of[k][nm]=float(yy);
            continue
        prd=np.zeros(len(y))
        for tr,te in GroupKFold(min(5,len(set(g)))).split(X,y,g):
            if len(set(y[tr]))<2: prd[te]=y[tr].mean(); continue
            m=HGB(max_depth=3,max_iter=150,learning_rate=0.08,l2_regularization=1.0).fit(X[tr],y[tr])
            prd[te]=m.predict_proba(X[te])[:,1]
        for k,pp in zip(ks,prd): P_of[k][nm]=float(pp)
    return {key:(max(P_of[key],key=lambda nm:P_of[key][nm]) if P_of[key] else None) for key,_ in alive}

def selacc(alive, pk):
    return float(np.mean([1.0 if (pk[k] is not None and e["P"][pk[k]]["tok"]==e["gt"]) else 0.0 for k,e in alive]))

def run_length(by_rid, pick_of):
    tot=n=0
    for rid, blocks in by_rid.items():
        for ds, pos in blocks:
            run=0
            for e in pos:
                gt=e["gt"]; P=e["P"]
                if gt is None or not P: break
                av=list(P); hits=[nm for nm in av if P[nm]["tok"]==gt]
                if len(hits)==len(av): run+=1; continue
                if len(hits)==0: break
                nm = pick_of.get((e["rid"],e["ds"],e["depth"]))
                if nm is not None and P[nm]["tok"]==gt: run+=1
                else: break
            tot+=run; n+=1
    return tot/max(n,1)

def main():
    for name, d, props in CELLS:
        names=[p[0] for p in props]; dd=f"{ROOT}/{d}"
        by_rid=load(f"{dd}/decisions_select1_oracle.jsonl",props)
        bad=loopy(dd); by_rid={k:v for k,v in by_rid.items() if k not in bad}
        attach_history(by_rid, names, delta=1.0)
        # decisive alive positions, keyed by (rid,ds,depth)
        alive=[]
        for rid, blocks in by_rid.items():
            for ds, pos in blocks:
                al=True
                for e in pos:
                    if not al: break
                    if not e["P"]: continue
                    av=list(e["P"]); hits=[nm for nm in av if e["P"][nm]["tok"]==e["gt"]]
                    if 0<len(hits)<len(av): alive.append(((e["rid"],e["ds"],e["depth"]), e))
                    if e["gt"] is not None and len(hits)==0: al=False
        raw_pick={(e["rid"],e["ds"],e["depth"]):(max(e["P"],key=lambda nm:e["P"][nm]["prob"]) if e["P"] else None) for _,e in alive}
        orc=run_length(by_rid, {(e["rid"],e["ds"],e["depth"]):next((nm for nm in e["P"] if e["P"][nm]["tok"]==e["gt"]),None) for _,e in alive})
        print(f"\n### {name}  decisive={len(alive)}  [OOF GroupKFold-by-rid, block-anchored]")
        rsa=selacc(alive,raw_pick); rmat=run_length(by_rid,raw_pick)
        print(f"  {'selector':28} {'selacc':>8} {'MAT':>8}")
        print(f"  {'raw (prob)':28} {rsa:>8.4f} {rmat:>8.3f}")
        for fs,lbl in [("pd","GBM prob+depth"),("phist","GBM prob+depth+HIST")]:
            pk=picks_oof(alive,names,fs)
            print(f"  {lbl:28} {selacc(alive,pk):>8.4f} {run_length(by_rid,pk):>8.3f}")
        print(f"  {'oracle':28} {1.0:>8.4f} {orc:>8.3f}")

if __name__ == "__main__":
    main()
