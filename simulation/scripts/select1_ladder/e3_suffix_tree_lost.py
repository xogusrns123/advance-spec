"""reliability_lost (EAGLE3 vs Suffix), suffix from two trees: AGNOSTIC = the EXISTING
ArcticInference combine (max-PATH-score over local/global) then calibrate, vs PER-TREE =
calibrate local/global separately and pick the higher-calibrated first token. Same
either-tree population, same calibrator family [prob, log n, match_len] OOF GBM — only
the COMBINE differs. Validates the replay's max-path-score suffix against the decision
log's served suffix_token. EAGLE3 joined from the decision log (committed positions),
calibrated by OOF isotonic(prob).

Run (IN docker): docker exec sglang-bench python3 /workspace/simulation/scripts/select1_ladder/e3_suffix_tree_lost.py
"""
import json, math
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold
from arctic_inference.suffix_decoding.cache import SuffixDecodingCache, SuffixTree

D="simulation/results/chain_hybrid_perdepth/qwen3_14b_ar"
LOG=f"{D}/decisions_select1_oracle.jsonl"
OUT="simulation/results/calib_reliability/figures/e3_suffix_tree_lost.png"
MAXD=64; FACTOR=4.0; OFFSET=0.0; MINP=0.1
NB=20; EDGES=np.linspace(0,1,NB+1); CENT=0.5*(EDGES[:-1]+EDGES[1:]); LOST="#ff7f0e"; ECOL="#1f77b4"; SCOL="#d62728"

# ---- parse decision log; reconstruct ordered committed decisions per request ----
req_ids={}; steps={}; decs=defaultdict(list); rid_order=[]
for line in open(LOG):
    o=json.loads(line); t=o.get("type")
    if t=="req":
        req_ids[o["rid"]]=list(o["input_ids"])
        if o["rid"] not in rid_order: rid_order.append(o["rid"])
    elif t=="step": steps[(o["rid"],o["decode_step"])]=o["accept_len"]
    elif t=="decision" and not o.get("tail"): decs[(o["rid"],o["decode_step"])].append(o)

def spec(t,ctx):
    try: return t.speculate(ctx,MAXD,FACTOR,OFFSET,MINP,False)
    except Exception: return SuffixTree.speculate(t,np.array(ctx,np.int32),MAXD,FACTOR,OFFSET,MINP,False)
def top(d):
    if not d.token_ids: return None
    c=int(d.counts[0]); p=float(d.probs[0])
    return dict(tok=int(d.token_ids[0]),n=int(round(c/p)) if p>0 else 0,p=p,ml=int(d.match_len),score=float(d.score))

sc=SuffixDecodingCache(max_tree_depth=MAXD)
pos=[]; n_serv=0; serv_match=0
for rid in rid_order:
    prompt=req_ids.get(rid)
    if prompt is None: continue
    dss=sorted(ds for (r,ds) in steps if r==rid)
    committed=[]
    for ds in dss:
        al=steps[(rid,ds)]
        for r in sorted(decs.get((rid,ds),[]), key=lambda x:x["depth"]):
            if r["depth"]<al and r.get("gt_token") is not None: committed.append(r)
    if not committed: continue
    sc.start_request(rid, list(prompt)); lt=sc._local_trees[rid]; gtree=sc._global_tree; seq=list(prompt); gen=[]
    for r in committed:
        gt=r["gt_token"]; ep=r.get("eagle_p"); et=r.get("eagle_token")
        dl=top(spec(lt,seq[-MAXD:])); dg=top(spec(gtree,seq[-MAXD:]))
        if ep is not None and et is not None and (dl is not None or dg is not None):
            # served suffix = max PATH-score combine (ArcticInference: ties -> local)
            if dg is None: served=dl
            elif dl is None: served=dg
            else: served = dl if dl["score"]>=dg["score"] else dg
            served=dict(served); served["corr"]=int(served["tok"]==gt)
            for x in (dl,dg):
                if x is not None: x["corr"]=int(x["tok"]==gt)
            # validate against the decision log's served suffix_token (if present)
            st_log=r.get("suffix_token")
            if st_log is not None:
                n_serv+=1; serv_match+= (served["tok"]==st_log)
            pos.append(dict(rid=rid, ep=float(ep), ec=int(et==gt), gt=gt, l=dl, g=dg, served=served))
        seq.append(gt); gen.append(gt)
    sc.add_active_response(rid, gen); sc.stop_request(rid)
print(f"either-tree suffix positions={len(pos)}   replay-served vs decision-log suffix_token match={100*serv_match/max(n_serv,1):.1f}% (n={n_serv})")

# ---- calibrators (OOF by rid, features [prob, log n, match_len]) ----
g=np.array([p["rid"] for p in pos])
def oof(X,y,grp,iso=False):
    pred=np.full(len(y),float(np.mean(y)))
    if len(y)<50 or len(set(y.tolist()))<2: return pred
    for tr,te in GroupKFold(5).split(X,y,grp):
        if len(set(y[tr].tolist()))<2: pred[te]=y[tr].mean(); continue
        if iso: pred[te]=IsotonicRegression(out_of_bounds="clip").fit(X[tr,0],y[tr]).predict(X[te,0])
        else: pred[te]=HGB(max_depth=3,max_iter=200,learning_rate=0.06,l2_regularization=1.0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return pred
def feat(d): return [d["p"], math.log1p(d["n"]), d["ml"]]
# eagle: isotonic(prob)
ecal=oof(np.array([[p["ep"]] for p in pos]),np.array([p["ec"] for p in pos]),g,iso=True)
# AGNOSTIC: one GBM on the SERVED (max-path-score) suffix first-token
svd=[(i,p["served"]) for i,p in enumerate(pos)]
sv_ag=oof(np.array([feat(d) for _,d in svd]),np.array([d["corr"] for _,d in svd]),np.array([pos[i]["rid"] for i,_ in svd]))
# PER-TREE: separate GBMs on local rows and global rows
loc=[(i,p["l"]) for i,p in enumerate(pos) if p["l"] is not None]
glo=[(i,p["g"]) for i,p in enumerate(pos) if p["g"] is not None]
cl=oof(np.array([feat(d) for _,d in loc]),np.array([d["corr"] for _,d in loc]),np.array([pos[i]["rid"] for i,_ in loc]))
cg=oof(np.array([feat(d) for _,d in glo]),np.array([d["corr"] for _,d in glo]),np.array([pos[i]["rid"] for i,_ in glo]))
for p in pos: p["ecal"]=p["sv_ag"]=p["cl"]=p["cg"]=None
for i,v in enumerate(ecal): pos[i]["ecal"]=float(v)
for (i,_),v in zip(svd,sv_ag): pos[i]["sv_ag"]=float(v)
for (i,_),v in zip(loc,cl): pos[i]["cl"]=float(v)
for (i,_),v in zip(glo,cg): pos[i]["cg"]=float(v)

def suffix_for(p, method):
    """returns (calibrated_val, corr) of the suffix first-token under the method."""
    if method=="ag":
        return p["sv_ag"], p["served"]["corr"]
    # per-tree: argmax over available trees' calibrated value
    cands=[]
    if p["l"] is not None: cands.append((p["cl"], p["l"]["corr"]))
    if p["g"] is not None: cands.append((p["cg"], p["g"]["corr"]))
    return max(cands, key=lambda c:c[0])

def bars(method):
    out={'e':[np.zeros(NB) for _ in range(3)], 's':[np.zeros(NB) for _ in range(3)]}; TL=0
    for p in pos:
        sval,scorr=suffix_for(p,method); eval_=p["ecal"]; ecorr=p["ec"]
        sel_s = sval>eval_
        be=min(int(eval_*NB),NB-1); out['e'][0][be]+=1
        if not sel_s and ecorr: out['e'][1][be]+=1
        if ecorr and sel_s and not scorr: out['e'][2][be]+=1; TL+=1
        bs=min(int(sval*NB),NB-1); out['s'][0][bs]+=1
        if sel_s and scorr: out['s'][1][bs]+=1
        if scorr and not sel_s and not ecorr: out['s'][2][bs]+=1; TL+=1
    return out,TL

cols=[("ag","AGNOSTIC = existing combine (max-path-score)"),("pt","PER-TREE calib")]
data={m:bars(m) for m,_ in cols}
ymax=max(data[m][0][w][0].max() for m,_ in cols for w in ('e','s'))
fig,axes=plt.subplots(2,2,figsize=(12,7.2),squeeze=False)
rows=[('e',"EAGLE3",ECOL),('s',"Suffix",SCOL)]
for ci,(m,ml) in enumerate(cols):
    b,TL=data[m]
    for ri,(who,wl,c) in enumerate(rows):
        tot,won,lost=b[who]; ax=axes[ri][ci]
        ax.bar(CENT,tot,width=0.045,color="#cccccc",alpha=0.6)
        ax.bar(CENT,won,width=0.045,color=c,alpha=0.8)
        ax.bar(CENT,lost,width=0.045,bottom=won,color=LOST,alpha=0.9,hatch="///",edgecolor="white",lw=0)
        ax.set_ylim(0,ymax*1.05); ax.set_xlim(0,1)
        ax.set_title(f"{wl} — {ml}\nLOST={int(lost.sum())}",fontsize=8.5)
        ax.set_xlabel("calibrated value",fontsize=8); ax.set_ylabel("count/bin",fontsize=8); ax.grid(axis="y",alpha=0.25)
    print(f"{ml}: total LOST={TL} (eagle {int(b['e'][2].sum())} + suffix {int(b['s'][2].sum())})")
handles=[Patch(facecolor="#cccccc",alpha=0.6,label="total / bin"),
         Patch(facecolor="#666",alpha=0.8,label="SELECTED & correct (won) [blue=E3, red=suffix]"),
         Patch(facecolor=LOST,alpha=0.9,hatch="///",edgecolor="white",label="correct but LOST")]
fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.01))
fig.suptitle("reliability_lost (EAGLE3 vs Suffix) — AGNOSTIC (existing max-path-score combine) vs PER-TREE calib (14B)\n"
             f"total LOST: agnostic {data['ag'][1]} | per-tree {data['pt'][1]}",fontsize=11)
fig.tight_layout(rect=[0,0.04,1,0.95]); fig.savefig(OUT,dpi=140); print("wrote",OUT)
