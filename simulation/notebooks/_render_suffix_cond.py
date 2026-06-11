import json, numpy as np, matplotlib, time
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
R=Path('/home/muchwater/advance-spec/simulation/results')
OUT=Path('/home/muchwater/advance-spec/simulation/notebooks/figures/anchor_depth')
CACHE=Path('/home/muchwater/advance-spec/simulation/notebooks/.ktau_cache'); CACHE.mkdir(exist_ok=True)
C_EXTENSION='#7b1fa2'   # purple = extension (matches prior survival-grid format)
MODELS={'qwen3_14b':('Qwen3-14B','explorations_qwen3_14b'),'qwen35_27b':('Qwen3.5-27B','explorations_qwen35_27b')}
WL={'specbench':'SpecBench','bfcl_v4':'BFCLv4','swebench_verified':'SWE-Bench Verified'}
MP=24

def get_ktau(mk,wl,path):
    cf=CACHE/f'{mk}_{wl}.npz'
    if cf.exists():
        z=np.load(cf); return z['K'],z['TAU']
    K=[];TAU=[]
    for line in open(path):
        r=json.loads(line)
        if r['method']!='extension:4.0:0.0': continue
        src=r.get('tree_source') or []; acc=r.get('tree_is_accepted') or []
        if not src: continue
        L=int(r.get('accepted',0))
        k=sum(1 for i in range(len(acc)) if acc[i] and i<len(src) and src[i]=='eagle')
        if k>8: continue
        K.append(k); TAU.append(L-k)
    K=np.array(K); TAU=np.array(TAU)
    np.savez(cf,K=K,TAU=TAU); return K,TAU

def suffix_grid(K,TAU,title,fname):
    ts=np.arange(MP+1)
    fig,axes=plt.subplots(3,3,figsize=(16,12),sharex=True,sharey=True)
    for d in range(9):
        ax=axes[d//3][d%3]; tau=TAU[K==d]; nk=len(tau)
        if nk<30:
            ax.text(0.5,0.5,f'd={d}: n={nk}',ha='center',va='center',transform=ax.transAxes)
            ax.set_xlim(0,MP); ax.set_ylim(0,1.02); continue
        A=np.array([(tau>=t).sum()/nk for t in range(MP+1)])
        ax.fill_between(ts,0,A,color=C_EXTENSION,alpha=0.13)         # area only, no box
        ax.plot(ts,A,color=C_EXTENSION,lw=2.6,marker='D',ms=3)
        ax.set_title(f'eagle3 accepted to depth d={d}   (N={nk:,}, mean τ={tau.mean():.2f})',fontsize=10)
        ax.set_xlim(0,MP); ax.set_ylim(0,1.02); ax.grid(alpha=0.3)
        if d%3==0: ax.set_ylabel('suffix survival  A(τ | anchor=d)')
        if d//3==2: ax.set_xlabel('τ  (suffix tokens beyond anchor)')
    fig.suptitle(f'{title} — SUFFIX-decoding conditional survival per anchor depth '
                 f'(eagle3 accepted exactly to depth d; not multiplied by eagle3 reach prob)',
                 y=1.005,fontsize=12)
    plt.tight_layout(); plt.savefig(OUT/fname,dpi=110,bbox_inches='tight'); plt.close(fig); print('saved',fname,flush=True)

t0=time.time()
for mk,(mt,sub) in MODELS.items():
    for wl,wll in WL.items():
        p=R/sub/'anchor_depth'/f'per_step_{wl}.jsonl'
        if not p.exists(): print('MISSING',p); continue
        K,TAU=get_ktau(mk,wl,p); print(f'{mk}/{wl}: N={len(K):,} @{time.time()-t0:.0f}s',flush=True)
        suffix_grid(K,TAU,f'{mt} / {wll} (N={len(K):,})',f'{mk}_{wl}_suffix_cond_survival.png')
print('ALL DONE',flush=True)
