"""reliability_lost in the EAGLE3-vs-suffix select-1 system, comparing the EXISTING
prob-only calibration vs the EVIDENCE-COUNT calibration (suffix: P(correct|prob,log n,
match_len); eagle: P(correct|prob,depth)) — the lever found in the local/global probe,
brought back to the real proposers. Rows = EAGLE3, Suffix. Cols = raw | calib(prob-only)
| calib(+evidence). Bars: total / SELECTED&correct (won) / correct-but-LOST. accept-
conditioned, OOF by rid, 14B decision log.

Run (host): python3 simulation/scripts/select1_ladder/suffix_calib_lost.py
"""
import sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_count import ROOT, CENT, NB, EDGES, load_chains, loopy_rids  # noqa: E402

OUT="simulation/results/calib_reliability/figures/suffix_calib_lost.png"
DIR="qwen3_14b_ar"; ECOL="#1f77b4"; SCOL="#d62728"; LOST="#ff7f0e"

def collect(chains):
    P=[]
    for rs in chains.values():
        alive=True
        for r in rs:
            if not alive: break
            gt=r.get("gt_token"); et,ep=r.get("eagle_token"),r.get("eagle_p")
            st,sp=r.get("suffix_token"),r.get("suffix_p")
            eav=et is not None and ep is not None; sav=st is not None and sp is not None
            if eav or sav:
                P.append(dict(rid=r["rid"], dep=int(r["depth"]),
                    ep=(float(ep) if eav else None), ec=int(eav and gt is not None and et==gt),
                    sp=(float(sp) if sav else None), sc=int(sav and gt is not None and st==gt),
                    sn=float(r.get("suffix_total") or 0), sml=float(r.get("match_len") or 0)))
            hits=[]
            if eav and gt is not None and et==gt: hits.append('e')
            if sav and gt is not None and st==gt: hits.append('s')
            if gt is not None and (eav or sav) and not hits: alive=False
    return P

def oof(X,y,g,iso=False):
    pred=np.full(len(y),float(np.mean(y)))
    if len(y)<50 or len(set(y.tolist()))<2: return pred
    for tr,te in GroupKFold(5).split(X,y,g):
        if len(set(y[tr].tolist()))<2: pred[te]=y[tr].mean(); continue
        if iso:
            m=IsotonicRegression(out_of_bounds="clip").fit(X[tr,0],y[tr]); pred[te]=m.predict(X[te,0])
        else:
            pred[te]=HGB(max_depth=3,max_iter=200,learning_rate=0.06,l2_regularization=1.0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return pred

d=f"{ROOT}/{DIR}"; ch=load_chains(f"{d}/decisions_select1_oracle.jsonl")
bad=loopy_rids(d); ch={k:v for k,v in ch.items() if k[0] not in bad}
P=collect(ch); N=len(P)
g=np.array([p["rid"] for p in P])
e_idx=[i for i,p in enumerate(P) if p["ep"] is not None]
s_idx=[i for i,p in enumerate(P) if p["sp"] is not None]
def col(idx,key): return np.array([P[i][key] for i in idx],float)
# calibrators
ev_iso=oof(col(e_idx,"ep").reshape(-1,1), col(e_idx,"ec"), g[e_idx], iso=True)
sv_iso=oof(col(s_idx,"sp").reshape(-1,1), col(s_idx,"sc"), g[s_idx], iso=True)
ev_mf =oof(np.c_[col(e_idx,"ep"),col(e_idx,"dep")], col(e_idx,"ec"), g[e_idx])
sv_mf =oof(np.c_[col(s_idx,"sp"),np.log1p(col(s_idx,"sn")),col(s_idx,"sml")], col(s_idx,"sc"), g[s_idx])
for arr in P: arr["ev_iso"]=arr["sv_iso"]=arr["ev_mf"]=arr["sv_mf"]=None
for j,i in enumerate(e_idx): P[i]["ev_iso"]=float(ev_iso[j]); P[i]["ev_mf"]=float(ev_mf[j])
for j,i in enumerate(s_idx): P[i]["sv_iso"]=float(sv_iso[j]); P[i]["sv_mf"]=float(sv_mf[j])

def bars(method):
    """method: 'raw'|'iso'|'mf'. returns {'e':(tot,won,lost,binval),'s':...}, total_lost."""
    val={'raw':('ep','sp'),'iso':('ev_iso','sv_iso'),'mf':('ev_mf','sv_mf')}[method]
    out={k:[np.zeros(NB) for _ in range(3)] for k in ('e','s')}; TL=0
    for p in P:
        ve=p[val[0]] if p["ep"] is not None else None
        vs=p[val[1]] if p["sp"] is not None else None
        sel_s = (ve is None) or (vs is not None and vs>ve)
        sel_e = (vs is None) or (ve is not None and not sel_s)
        for who,av,vv,corr,sel in (('e',p["ep"] is not None,ve,p["ec"],sel_e),
                                   ('s',p["sp"] is not None,vs,p["sc"],sel_s)):
            if not av: continue
            b=min(int(vv*NB),NB-1)
            out[who][0][b]+=1
            if sel and corr: out[who][1][b]+=1
            if corr and not sel:
                other_corr = p["sc"] if who=='e' else p["ec"]
                if other_corr==0: out[who][2][b]+=1; TL+=1
    return out,TL

cols=[("raw","RAW (uncalibrated)"),("iso","calib PROB-ONLY (existing)"),("mf","calib +EVIDENCE (prob,n,match_len)")]
data={m:bars(m) for m,_ in cols}
ymax=max(data[m][0][w][0].max() for m,_ in cols for w in ('e','s'))
fig,axes=plt.subplots(2,3,figsize=(16,7.2),squeeze=False)
rows=[('e',"EAGLE3",ECOL),('s',"Suffix",SCOL)]
for ci,(m,mlabel) in enumerate(cols):
    b,TL=data[m]
    for ri,(who,wlabel,c) in enumerate(rows):
        tot,won,lost=b[who]; ax=axes[ri][ci]
        ax.bar(CENT,tot,width=0.045,color="#cccccc",alpha=0.6)
        ax.bar(CENT,won,width=0.045,color=c,alpha=0.8)
        ax.bar(CENT,lost,width=0.045,bottom=won,color=LOST,alpha=0.9,hatch="///",edgecolor="white",lw=0)
        ax.set_ylim(0,ymax*1.05); ax.set_xlim(0,1)
        ax.set_title(f"{wlabel} — {mlabel}\nLOST={int(lost.sum())}",fontsize=8.5)
        ax.set_xlabel("calibrated value" if m!='raw' else "raw prob",fontsize=8); ax.set_ylabel("count/bin",fontsize=8)
        ax.grid(axis="y",alpha=0.25)
    print(f"{mlabel}: total LOST={TL}  (eagle {int(b['e'][2].sum())} + suffix {int(b['s'][2].sum())})")
handles=[Patch(facecolor="#cccccc",alpha=0.6,label="total proposals / bin"),
         Patch(facecolor="#666",alpha=0.8,label="SELECTED & correct (won) [blue=EAGLE3, red=Suffix]"),
         Patch(facecolor=LOST,alpha=0.9,hatch="///",edgecolor="white",label="correct but LOST (not selected & the pick was wrong)")]
fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.01))
fig.suptitle("reliability_lost (EAGLE3 vs Suffix select-1) — existing prob-only calib vs +evidence-count calib (14B)\n"
             f"total LOST: raw {data['raw'][1]} | prob-only {data['iso'][1]} | +evidence {data['mf'][1]}",fontsize=11)
fig.tight_layout(rect=[0,0.04,1,0.95]); fig.savefig(OUT,dpi=140); print("wrote",OUT)
