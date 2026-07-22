"""IN-SAMPLE vs HELD-OUT (OOF) selection analysis on the SAME 2-way eval logs (offline,
block-anchored). For each arm {raw, calib best-of-4, bayes GBM, oracle} we fit the selector
on the eval decision log and score it BOTH ways:
  - in-sample : fit on all eval rows, predict the same rows (optimistic)
  - OOF       : GroupKFold-by-rid held-out
and report decisive selacc + block-anchored MAT (run-length on the gt trajectory).
The in-sample - OOF gap = the selector's overfitting/generalization gap. (Realized SERVED
numbers are a separate train/eval split; this isolates the in-sample effect on one substrate.)

Run: python3 simulation/scripts/select1_ladder/insample_vs_heldout_2way.py
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

ROOT = "simulation/results/chain_hybrid_perdepth"
CELLS = [
    dict(key="14B 2-way", dir="qwen3_14b_ar", props=[("EAGLE3","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
    dict(key="27B 2-way", dir="qwen35_27b_ar", props=[("MTP","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
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

def fv(e, nm, featset):
    P = e["P"][nm]; v = [P["prob"], e["depth"]]
    if featset == "prob_depth_suffix" and nm == "suffix":
        v += [P["mlen"], P["lcnt"]]
    return v

def fit_predict(method, Xfull, prob, y, fit, pred):
    ytr = y[fit]
    if len(set(ytr)) < 2: return np.full(len(pred), ytr.mean())
    if method == "logistic":
        Xt = Xfull[fit]; m = Xt.mean(0); s = Xt.std(0)+1e-9
        clf = LogisticRegression(max_iter=1000).fit((Xt-m)/s, ytr)
        return clf.predict_proba((Xfull[pred]-m)/s)[:,1]
    if method == "gbm":
        clf = HGB(max_depth=3, max_iter=150, learning_rate=0.08, l2_regularization=1.0).fit(Xfull[fit], ytr)
        return clf.predict_proba(Xfull[pred])[:,1]
    p = prob
    if method == "iso":
        return IsotonicRegression(out_of_bounds="clip").fit(p[fit], ytr).predict(p[pred])
    if method == "beta":
        F = np.c_[np.log(np.clip(p,1e-6,1)), np.log(np.clip(1-p,1e-6,1))]
        return LogisticRegression(max_iter=1000).fit(F[fit], ytr).predict_proba(F[pred])[:,1]
    if method == "hist":
        edges = np.unique(np.quantile(p[fit], np.linspace(0,1,11)))
        if len(edges) < 3: return np.full(len(pred), ytr.mean())
        bt = np.clip(np.digitize(p[fit], edges[1:-1]), 0, len(edges)-2)
        means = np.array([ytr[bt==b].mean() if (bt==b).any() else ytr.mean() for b in range(len(edges)-1)])
        return means[np.clip(np.digitize(p[pred], edges[1:-1]), 0, len(edges)-2)]
    raise ValueError(method)

def picks(alive, props, featset, method, insample):
    names = [p[0] for p in props]
    R = {nm: {"X":[],"p":[],"y":[],"g":[],"d":[]} for nm in names}
    for did, e in alive:
        for nm in e["P"]:
            R[nm]["X"].append(fv(e,nm,featset)); R[nm]["p"].append(e["P"][nm]["prob"])
            R[nm]["y"].append(1 if e["P"][nm]["tok"]==e["gt"] else 0)
            R[nm]["g"].append(e["rid"]); R[nm]["d"].append(did)
    P_of = defaultdict(dict)
    for nm in names:
        X=np.array(R[nm]["X"],float); p=np.array(R[nm]["p"],float); y=np.array(R[nm]["y"]); g=np.array(R[nm]["g"]); ds=R[nm]["d"]
        if len(y)<10 or len(set(y))<2:
            for d,yy in zip(ds,y): P_of[d][nm]=float(yy)
            continue
        prd=np.zeros(len(y))
        if insample:
            prd=fit_predict(method,X,p,y,np.arange(len(y)),np.arange(len(y)))
        else:
            for tr,te in GroupKFold(min(5,len(set(g)))).split(X,y,g):
                prd[te]=fit_predict(method,X,p,y,tr,te)
        for d,pp in zip(ds,prd): P_of[d][nm]=float(pp)
    return {did:(max(P_of[did],key=lambda nm:P_of[did][nm]) if P_of[did] else None) for did,_ in alive}

def selacc(alive, pk):
    return float(np.mean([1.0 if (pk[d] is not None and e["P"][pk[d]]["tok"]==e["gt"]) else 0.0 for d,e in alive])) if alive else float("nan")

def run_length(blocks, pick_fn):
    tot=n=0
    for k,pos in blocks.items():
        n+=1; run=0
        for e in pos:
            gt=e["gt"]
            if gt is None or not e["P"]: break
            av=list(e["P"]); hits=[nm for nm in av if e["P"][nm]["tok"]==gt]
            if len(hits)==len(av): run+=1; continue
            if len(hits)==0: break
            nm=pick_fn((k[0],k[1],e["depth"]),e)
            if nm is not None and e["P"][nm]["tok"]==gt: run+=1
            else: break
        tot+=run
    return tot/max(n,1)

def best_calib(blocks, alive, props, featset, insample):
    best=None
    for m in ("hist","iso","beta","logistic"):
        pk=picks(alive,props,featset,m,insample); sa=selacc(alive,pk)
        if best is None or sa>best[1]: best=(m,sa,pk)
    return best[0], best[1], run_length(blocks, lambda did,e,_p=best[2]:_p.get(did))

def run_cell(cell):
    d=f"{ROOT}/{cell['dir']}"; props=cell["props"]; names=[p[0] for p in props]
    blocks=load_blocks(f"{d}/decisions_select1_oracle.jsonl",props)
    bad=loopy(d); blocks={k:v for k,v in blocks.items() if k[0] not in bad}
    alive=[]
    for k,pos in blocks.items():
        al=True
        for e in pos:
            if not al: break
            if not e["P"]: continue
            av=list(e["P"]); hits=[nm for nm in av if e["P"][nm]["tok"]==e["gt"]]
            if 0<len(hits)<len(av): alive.append(((k[0],k[1],e["depth"]),e))
            if e["gt"] is not None and len(hits)==0: al=False
    raw_pk={did:(max(e["P"],key=lambda nm:e["P"][nm]["prob"]) if e["P"] else None) for did,e in alive}
    orc=run_length(blocks, lambda did,e: next((nm for nm in e["P"] if e["P"][nm]["tok"]==e["gt"]),None))
    print(f"\n### {cell['key']}  decisive={len(alive)}   [OFFLINE block-anchored; in-sample vs OOF on SAME eval log]")
    print(f"  {'arm':34s} {'selacc IN':>10} {'selacc OOF':>11} | {'MAT IN':>8} {'MAT OOF':>9}")
    rsa=selacc(alive,raw_pk); rmat=run_length(blocks,lambda did,e,_p=raw_pk:_p.get(did))
    print(f"  {'raw (prob argmax)':34s} {rsa:>10.4f} {rsa:>11.4f} | {rmat:>8.3f} {rmat:>9.3f}   (rule-based: same)")
    for fs,lbl in [("prob_depth","calib best-of-4 (prob+depth)"),("prob_depth_suffix","calib best-of-4 (prob+depth+sfx)")]:
        mi,si,mati=best_calib(blocks,alive,props,fs,True); mo,so,mato=best_calib(blocks,alive,props,fs,False)
        print(f"  {lbl+' ['+mi+'/'+mo+']':34s} {si:>10.4f} {so:>11.4f} | {mati:>8.3f} {mato:>9.3f}")
    for fs,lbl in [("prob_depth","bayes GBM (prob+depth)"),("prob_depth_suffix","bayes GBM (prob+depth+sfx)")]:
        pi=picks(alive,props,fs,"gbm",True); po=picks(alive,props,fs,"gbm",False)
        si=selacc(alive,pi); so=selacc(alive,po)
        mati=run_length(blocks,lambda did,e,_p=pi:_p.get(did)); mato=run_length(blocks,lambda did,e,_p=po:_p.get(did))
        print(f"  {lbl:34s} {si:>10.4f} {so:>11.4f} | {mati:>8.3f} {mato:>9.3f}")
    print(f"  {'oracle':34s} {1.0:>10.4f} {1.0:>11.4f} | {orc:>8.3f} {orc:>9.3f}")

if __name__ == "__main__":
    for c in CELLS:
        try: run_cell(c)
        except Exception as ex:
            import traceback; print(f"ERR {c['key']}: {ex}"); traceback.print_exc()
