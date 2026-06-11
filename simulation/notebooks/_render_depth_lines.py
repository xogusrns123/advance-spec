import json, numpy as np, matplotlib, time
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
R=Path('/home/muchwater/advance-spec/simulation/results')
OUT=Path('/home/muchwater/advance-spec/simulation/notebooks/figures/anchor_depth')
CACHE=Path('/home/muchwater/advance-spec/simulation/notebooks/.depthmean_cache'); CACHE.mkdir(exist_ok=True)
BLUE,ORANGE,PURPLE='#1f77b4','#ff7f0e','#7b1fa2'
MODELS={'qwen3_14b':('Qwen3-14B','EAGLE3','explorations_qwen3_14b'),
        'qwen35_27b':('Qwen3.5-27B','MTP','explorations_qwen35_27b')}
WL={'specbench':'SpecBench','bfcl_v4':'BFCLv4','swebench_verified':'SWE-Bench Verified'}

def depth_means(mk,wl,path):
    cf=CACHE/f'{mk}_{wl}.npz'
    if cf.exists():
        z=np.load(cf); return z['ds'],z['eg'],z['sf'],z['ex'],z['n']
    rec=defaultdict(dict)
    for line in open(path):
        r=json.loads(line); m=r['method']; key=(r['request_id'],r.get('call_idx',0),r['step_id'])
        if m=='extension:4.0:0.0':
            src=r.get('tree_source') or []; acc=r.get('tree_is_accepted') or []
            if not src: continue
            k=sum(1 for i in range(len(acc)) if acc[i] and i<len(src) and src[i]=='eagle')
            rec[key]['k']=k; rec[key]['ext']=int(r.get('accepted',0))
        elif m=='single:eagle3': rec[key]['eg']=int(r.get('accepted',0))
        elif m=='single:suffix:4.0:0.0': rec[key]['sf']=int(r.get('accepted',0))
    g=defaultdict(lambda:{'eg':[],'sf':[],'ext':[]})
    for v in rec.values():
        if 'k' not in v or 'eg' not in v or 'sf' not in v: continue
        if v['k']<=8:
            g[v['k']]['eg'].append(v['eg']); g[v['k']]['sf'].append(v['sf']); g[v['k']]['ext'].append(v['ext'])
    ds=list(range(9)); eg=[];sf=[];ex=[];n=[]
    for d in ds:
        a=g.get(d)
        if a and len(a['ext'])>0:
            eg.append(np.mean(a['eg'])); sf.append(np.mean(a['sf'])); ex.append(np.mean(a['ext'])); n.append(len(a['ext']))
        else: eg.append(np.nan); sf.append(np.nan); ex.append(np.nan); n.append(0)
    ds=np.array(ds); eg=np.array(eg); sf=np.array(sf); ex=np.array(ex); n=np.array(n)
    np.savez(cf,ds=ds,eg=eg,sf=sf,ex=ex,n=n); return ds,eg,sf,ex,n

t0=time.time()
fig,axes=plt.subplots(2,3,figsize=(18,10),sharex=True)
for ri,(mk,(mt,bb,sub)) in enumerate(MODELS.items()):
    for ci,(wl,wll) in enumerate(WL.items()):
        ax=axes[ri][ci]
        p=R/sub/'anchor_depth'/f'per_step_{wl}.jsonl'
        if not p.exists(): ax.text(0.5,0.5,'missing',ha='center',transform=ax.transAxes); continue
        ds,eg,sf,ex,n=depth_means(mk,wl,p); print(f'{mk}/{wl} @{time.time()-t0:.0f}s',flush=True)
        # EAGLE3-only / Suffix-only = root-start single methods -> MARGINAL mean
        # over ALL steps (a single fixed value, drawn as a horizontal line, NOT
        # per anchor depth). Only Extension varies by anchor depth.
        w=n.astype(float); tot=w.sum()
        eg_marg=float(np.nansum(np.nan_to_num(eg)*w)/tot); sf_marg=float(np.nansum(np.nan_to_num(sf)*w)/tot)
        ax.axhline(eg_marg,color=BLUE,lw=2.2,ls='--',label=f'{bb} only (all)={eg_marg:.2f}')
        ax.axhline(sf_marg,color=ORANGE,lw=2.2,ls='--',label=f'Suffix only (all)={sf_marg:.2f}')
        ax.plot(ds,ex,color=PURPLE,lw=2.8,marker='D',ms=6,label='Extension (anchor=d)')
        ax.fill_between(ds,eg_marg,ex,where=~np.isnan(ex),color=PURPLE,alpha=0.10)
        ax.set_title(f'{mt} / {wll}',fontsize=12,fontweight='bold')
        ax.grid(alpha=0.3); ax.set_xticks(range(9)); ax.set_xlim(-0.3,8.3); ax.set_ylim(bottom=0)
        if ci==0: ax.set_ylabel('mean accepted tokens (E[L])',fontsize=11)
        if ri==1: ax.set_xlabel('anchor depth d  (eagle3 accepted exactly to d)',fontsize=11)
        if ri==0 and ci==0: ax.legend(loc='upper left',fontsize=10)
fig.suptitle('Mean accepted tokens per anchor depth — EAGLE3/MTP only vs Suffix only vs Extension '
             '(EAGLE3/Suffix MARGINAL (dashed, same value), Extension conditional per anchor depth)',y=1.01,fontsize=13,fontweight='bold')
plt.tight_layout(); fp=OUT/'survival_by_depth_singles_vs_extension.png'
plt.savefig(fp,dpi=120,bbox_inches='tight'); print('saved',fp,flush=True); print('ALL DONE',flush=True)
