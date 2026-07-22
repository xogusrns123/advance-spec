"""OOF (held-out, GroupKFold-by-rid) vs IN-SAMPLE (fit & predict same rows) for the
fitted select-1 policies. single/raw/oracle are rule-based -> identical, shown once.
The OOF<in-sample gap = overfitting/variance (why unconstrained bayes-0.5 can lose to a
simpler calib OOF)."""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

ROOT = "simulation/results/chain_hybrid_perdepth"
CELLS = [
 dict(key="14B 2-way", dir="qwen3_14b_ar", nway=2,
      props=[("EAGLE3","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
 dict(key="27B 2-way", dir="qwen35_27b_ar", nway=2,
      props=[("MTP","eagle_token","eagle_p"),("suffix","suffix_token","suffix_p")]),
 dict(key="8B 3-way", dir="qwen3_8b_dflash_e3_ceiling20", nway=3,
      props=[("DFlash","eagle_token","eagle_p"),("EAGLE3","e3_token","e3_p"),("suffix","suffix_token","suffix_p")]),
 dict(key="27B 3-way", dir="qwen35_27b_3way_real_full", nway=3,
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
                                      ltot=math.log1p(float(r.get("suffix_total") or 0)))
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

def n_agree(e, nm):
    t = e["P"][nm]["tok"]; return sum(1 for q,v in e["P"].items() if q!=nm and v["tok"]==t)

def fv(e, nm, variant, nway):
    P = e["P"][nm]
    if variant == "prob": return [P["prob"]]
    if variant == "prob_depth": return [P["prob"], e["depth"]]
    if variant == "bayes":
        o = [v["prob"] for q,v in e["P"].items() if q!=nm]; return [P["prob"], max(o) if o else 0.0, e["depth"]]
    v = [P["prob"], e["depth"]]
    if nway == 3: v.append(float(n_agree(e, nm)))
    if nm == "suffix": v += [P["mlen"], P["ltot"]]
    return v

def fit_pick(alive, props, variant, nway, insample):
    names = [p[0] for p in props]
    R = {nm: {"X":[],"y":[],"g":[],"d":[]} for nm in names}
    for did, e in alive:
        for nm in e["P"]:
            R[nm]["X"].append(fv(e,nm,variant,nway)); R[nm]["y"].append(1 if e["P"][nm]["tok"]==e["gt"] else 0)
            R[nm]["g"].append(e["rid"]); R[nm]["d"].append(did)
    P_of = defaultdict(dict)
    for nm in names:
        X=np.array(R[nm]["X"],float); y=np.array(R[nm]["y"]); g=np.array(R[nm]["g"]); ds=R[nm]["d"]
        if len(y)<10 or y.min()==y.max():
            for d,yy in zip(ds,y): P_of[d][nm]=float(yy)
            continue
        pred=np.zeros(len(y))
        mk=lambda: HGB(max_depth=3,max_iter=150,learning_rate=0.08,l2_regularization=1.0)
        if insample:
            pred=mk().fit(X,y).predict_proba(X)[:,1]
        else:
            for tr,te in GroupKFold(min(5,len(set(g)))).split(X,y,g):
                if len(set(y[tr]))<2: pred[te]=y[tr].mean(); continue
                pred[te]=mk().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
        for d,pp in zip(ds,pred): P_of[d][nm]=float(pp)
    pick={}
    for did,_ in alive:
        c=P_of[did]; pick[did]=max(c,key=lambda nm:c[nm]) if c else None
    return pick

def run_length(blocks, pick_fn):
    tot=0;n=0
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

def selacc(alive, pick):
    return np.mean([1.0 if (pick[d] is not None and e["P"][pick[d]]["tok"]==e["gt"]) else 0.0
                    for d,e in alive]) if alive else float("nan")

for cell in CELLS:
    d=f"{ROOT}/{cell['dir']}"; props=cell["props"]; nway=cell["nway"]
    blocks=load_blocks(f"{d}/decisions_select1_oracle.jsonl",props); bad=loopy(d)
    blocks={k:v for k,v in blocks.items() if k[0] not in bad}
    alive=[]
    for k,pos in blocks.items():
        al=True
        for e in pos:
            if not al: break
            if not e["P"]: continue
            av=list(e["P"]); hits=[nm for nm in av if e["P"][nm]["tok"]==e["gt"]]
            if 0<len(hits)<len(av): alive.append(((k[0],k[1],e["depth"]),e))
            if e["gt"] is not None and len(hits)==0: al=False
    print(f"\n### {cell['key']}  decisive={len(alive)}")
    print(f"  {'policy':40} {'selacc OOF':>11} {'in-sample':>10} | {'MAT OOF':>9} {'in-sample':>10}")
    variants=[("prob","calib (prob)"),("prob_depth","calib (prob+depth)"),
              ("prob_depth_other","calib (prob+depth+other)"),("bayes","bayes-0.5 (prob+max_other+depth)")]
    for var,lbl in variants:
        po=fit_pick(alive,props,var,nway,False); pi=fit_pick(alive,props,var,nway,True)
        so=selacc(alive,po); si=selacc(alive,pi)
        mo=run_length(blocks,lambda did,e,_p=po:_p.get(did)); mi=run_length(blocks,lambda did,e,_p=pi:_p.get(did))
        print(f"  {lbl:40} {so:>11.4f} {si:>10.4f} | {mo:>9.3f} {mi:>10.3f}")
