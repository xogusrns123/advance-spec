"""reliability_lost format for the LOCAL-vs-GLOBAL tree selection, comparing
tree-AGNOSTIC calibration (one shared calibrator on both trees) vs PER-TREE calibration
(separate calibrators). Rows = {LOCAL, GLOBAL} proposers; cols = {agnostic, per-tree}.
Bars: total proposals (gray) / SELECTED & correct = won (color) / correct but LOST
(not selected & the selected tree was wrong, orange-hatched). Binned by the CALIBRATED
first-token prob the selection used. Both methods use the SAME features [prob, log n,
match_len] OOF GBM — only the tree-split differs.

Run (IN docker): docker exec sglang-bench python3 /workspace/simulation/scripts/select1_ladder/global_tree_lost.py
"""
import json, math
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from arctic_inference.suffix_decoding.cache import SuffixDecodingCache, SuffixTree
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

GT="simulation/results/chain_hybrid_perdepth/qwen3_14b_ar/gt_tokens.jsonl"
OUT="simulation/results/calib_reliability/figures/global_tree_lost.png"
MAXD=64; FACTOR=4.0; OFFSET=0.0; MINP=0.1
NB=20; EDGES=np.linspace(0,1,NB+1); CENT=0.5*(EDGES[:-1]+EDGES[1:]); LOST="#ff7f0e"
LCOL="#1f77b4"; GCOL="#2ca02c"

def spec(t,ctx):
    try: return t.speculate(ctx,MAXD,FACTOR,OFFSET,MINP,False)
    except Exception: return SuffixTree.speculate(t,np.array(ctx,np.int32),MAXD,FACTOR,OFFSET,MINP,False)
def top(d):
    if not d.token_ids: return None
    c=int(d.counts[0]); p=float(d.probs[0])
    return dict(tok=int(d.token_ids[0]),c=c,n=int(round(c/p)) if p>0 else 0,p=p,ml=int(d.match_len))

recs=[json.loads(l) for l in open(GT)]
sc=SuffixDecodingCache(max_tree_depth=MAXD); rows=[]
for ri,rec in enumerate(recs):
    rid=f"r{ri}"; prompt=list(rec.get("input_ids") or []); gen=list(rec.get("output_ids") or [])
    if not gen: continue
    sc.start_request(rid,prompt); lt=sc._local_trees[rid]; gtree=sc._global_tree; seq=list(prompt)
    for tok in gen:
        dl=top(spec(lt,seq[-MAXD:])); dg=top(spec(gtree,seq[-MAXD:]))
        if dl is not None:
            for d in (dl,dg):
                if d is not None: d["corr"]=int(d["tok"]==tok)
            rows.append(dict(rid=rid,l=dl,g=dg))
        seq.append(tok)
    sc.add_active_response(rid,gen); sc.stop_request(rid)

def feats(d): return [d["p"], math.log1p(d["n"]), d["ml"]]
def oof(sub):
    X=np.array([feats(r) for r in sub],float); y=np.array([r["corr"] for r in sub]); g=np.array([r["rid"] for r in sub])
    pred=np.full(len(y),y.mean())
    if len(set(y.tolist()))>1 and len(y)>50:
        for tr,te in GroupKFold(5).split(X,y,g):
            if len(set(y[tr].tolist()))<2: pred[te]=y[tr].mean(); continue
            pred[te]=HGB(max_depth=3,max_iter=200,learning_rate=0.06,l2_regularization=1.0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return pred
# build subsets
loc=[dict(r["l"],rid=r["rid"],_i=i) for i,r in enumerate(rows)]
glo=[dict(r["g"],rid=r["rid"],_i=i) for i,r in enumerate(rows) if r["g"] is not None]
# per-tree calibrators
cl_pt=oof(loc); cg_pt=oof(glo)
# agnostic: one calibrator on pooled rows
pool=loc+glo; cp=oof(pool)
for r in rows: r["cl_pt"]=r["cg_pt"]=r["cl_ag"]=r["cg_ag"]=None
for s,v in zip(loc,cl_pt): rows[s["_i"]]["cl_pt"]=float(v)
for s,v in zip(glo,cg_pt): rows[s["_i"]]["cg_pt"]=float(v)
# assign agnostic preds back (first len(loc) are local rows, rest global)
for s,v in zip(loc,cp[:len(loc)]): rows[s["_i"]]["cl_ag"]=float(v)
for s,v in zip(glo,cp[len(loc):]): rows[s["_i"]]["cg_ag"]=float(v)

def lost_bars(method):
    """returns {'local':(tot,won,lost), 'global':(...)} for a method ('ag' or 'pt')."""
    cl=f"cl_{method}"; cg=f"cg_{method}"
    out={k:[np.zeros(NB) for _ in range(3)] for k in ("local","global")}
    tl=0
    for r in rows:
        l=r["l"]; g=r["g"]
        vl=r[cl]; vg=r[cg] if g is not None else None
        # selection: higher calibrated value (>= -> local)
        sel_local = (g is None) or (vl>=vg)
        for who,d,val,sel in (("local",l,vl,sel_local),("global",g,vg,(g is not None and not sel_local))):
            if d is None: continue
            b=min(int(val*NB),NB-1)
            out[who][0][b]+=1                       # total
            if sel and d["corr"]: out[who][1][b]+=1  # won
            # lost: this tree correct, not selected, selected tree wrong
            if d["corr"] and not sel:
                seld = l if sel_local else g
                if seld is None or seld["corr"]==0:
                    out[who][2][b]+=1; tl+=1
    return out, tl

fig,axes=plt.subplots(2,2,figsize=(12,7.2),squeeze=False)
methods=[("ag","tree-AGNOSTIC calib (one shared)"),("pt","PER-TREE calib (separate)")]
data={m:lost_bars(m) for m,_ in methods}
ymax=max(d[0][who][0].max() for d in data.values() for who in ("local","global"))
for ci,(m,mlabel) in enumerate(methods):
    bars,tl=data[m]
    for ri,(who,col) in enumerate((("local",LCOL),("global",GCOL))):
        tot,won,lost=bars[who]; ax=axes[ri][ci]
        ax.bar(CENT,tot,width=0.045,color="#cccccc",alpha=0.6)
        ax.bar(CENT,won,width=0.045,color=col,alpha=0.8)
        ax.bar(CENT,lost,width=0.045,bottom=won,color=LOST,alpha=0.9,hatch="///",edgecolor="white",lw=0)
        ax.set_ylim(0,ymax*1.05); ax.set_xlim(0,1)
        ax.set_title(f"{who.upper()} tree — {mlabel}\nLOST={int(lost.sum())}",fontsize=9)
        ax.set_xlabel("calibrated first-token prob",fontsize=8); ax.set_ylabel("count / bin",fontsize=8)
        ax.grid(axis="y",alpha=0.25)
    print(f"{mlabel}: total LOST = {tl}")
handles=[Patch(facecolor="#cccccc",alpha=0.6,label="total proposals / bin"),
         Patch(facecolor="#666",alpha=0.8,label="SELECTED & correct (won) [blue=local, green=global]"),
         Patch(facecolor=LOST,alpha=0.9,hatch="///",edgecolor="white",label="correct but LOST (not selected & selected tree wrong)")]
fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.01))
fig.suptitle("reliability_lost: LOCAL vs GLOBAL tree selection — tree-agnostic vs per-tree calibration\n"
             f"(14B, first-token; total LOST agnostic {data['ag'][1]} vs per-tree {data['pt'][1]})",fontsize=11)
fig.tight_layout(rect=[0,0.04,1,0.95]); fig.savefig(OUT,dpi=140)
print("wrote",OUT)
