import json, numpy as np, matplotlib, time
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from collections import defaultdict
from pathlib import Path
R=Path('/home/muchwater/advance-spec/simulation/results')
OUT=Path('/home/muchwater/advance-spec/simulation/notebooks/figures/anchor_depth')
SUFFIX_COLOR='#ff7f0e'; EAGLE3_COLOR='#1f77b4'
MODELS={'qwen3_14b':('Qwen3-14B','explorations_qwen3_14b'),'qwen35_27b':('Qwen3.5-27B','explorations_qwen35_27b')}
WL={'specbench':'SpecBench','bfcl_v4':'BFCLv4','swebench_verified':'SWE-Bench Verified'}
MP=24
EDGES=np.arange(0,1.0001,0.1); CENT=(EDGES[:-1]+EDGES[1:])/2; W=(EDGES[1]-EDGES[0])*0.92; MINB=20

def extract(path):
    # per step: k (anchor depth), tau=L-k, path_pt @ anchor, sfx_score @ anchor
    K=[];TAU=[];PPT=[];SC=[]
    for line in open(path):
        r=json.loads(line)
        if r['method']!='extension:4.0:0.0': continue
        src=r.get('tree_source') or []; acc=r.get('tree_is_accepted') or []
        an=r.get('tree_anchor_node_id') or []; pp=r.get('tree_path_prob') or []
        scp=r.get('tree_suffix_cum_prob') or []
        if not src: continue
        L=int(r.get('accepted',0))
        eacc=[i for i in range(len(acc)) if acc[i] and i<len(src) and src[i]=='eagle']
        k=len(eacc)
        if k>8: continue
        anode=eacc[-1] if k>=1 else -1
        path_pt=(pp[anode] if (k>=1 and anode<len(pp) and pp[anode] is not None) else 1.0)
        cand=[scp[i] for i in range(len(src)) if src[i]=='suffix' and i<len(an) and an[i]==anode and i<len(scp) and scp[i] is not None]
        sfx=max(cand) if cand else np.nan
        K.append(k); TAU.append(L-k); PPT.append(path_pt); SC.append(sfx)
    return np.array(K),np.array(TAU,dtype=float),np.array(PPT,dtype=float),np.array(SC,dtype=float)

def int_ticks(hi,t=6):
    s=max(int(np.ceil(hi/t)),1); return np.arange(0,int(hi)+1,s)

def bars(K,TAU,metric,mlabel,color,title,fname):
    per={}; ymax=1.0
    for d in range(9):
        m=(K==d)&np.isfinite(metric); v=metric[m]; t=TAU[m]
        ys=np.full(len(CENT),np.nan); ns=np.zeros(len(CENT),int)
        for i in range(len(CENT)):
            bm=(v>=EDGES[i])&((v<=EDGES[i+1]) if i==len(CENT)-1 else (v<EDGES[i+1]))
            ns[i]=bm.sum()
            if bm.sum()>=MINB: ys[i]=t[bm].mean()
        per[d]=(ys,ns,int(m.sum()))
        f=ys[np.isfinite(ys)]
        if len(f): ymax=max(ymax,float(f.max()))
    ymax=float(int(np.ceil(ymax*1.05)))
    fig,axes=plt.subplots(3,3,figsize=(16,12),sharex=True,sharey=True)
    for d in range(9):
        ax=axes[d//3][d%3]; ys,ns,nk=per[d]
        ax.bar(CENT,np.nan_to_num(ys),width=W,color=color,edgecolor='white',linewidth=0.4)
        nz=np.nonzero(ns>0)[0]; xhi=float(EDGES[nz[-1]+1]) if len(nz) else 1.0
        ax.set_xlim(0,xhi); ax.set_ylim(0,ymax)
        ax.set_yticks(int_ticks(ymax)); ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.set_xticks(np.arange(0,xhi+1e-9,max(xhi/5,0.1)))
        ax.grid(True,alpha=0.3); ax.set_title(f'anchor depth d={d}',fontsize=12,fontweight='bold')
        ax.text(0.02,0.96,f'N = {nk:,}',transform=ax.transAxes,fontsize=9,va='top',ha='left',
                bbox=dict(boxstyle='round,pad=0.3',facecolor='white',alpha=0.85,edgecolor='gray'))
        if d%3==0: ax.set_ylabel('mean extension survival  E[τ]',fontsize=11)
        if d//3==2: ax.set_xlabel(f'{mlabel} @ anchor',fontsize=11)
    fig.suptitle(f'{title} — extension survival E[τ] vs {mlabel}, per anchor depth',fontsize=13,fontweight='bold',y=1.005)
    plt.tight_layout(); plt.savefig(OUT/fname,dpi=110,bbox_inches='tight'); plt.close(fig); print('  saved',fname)

def suffix_cond(K,TAU,title,fname):
    ts=np.arange(MP+1)
    fig,axes=plt.subplots(3,3,figsize=(16,12),sharex=True,sharey=True)
    for d in range(9):
        ax=axes[d//3][d%3]; m=(K==d); tau=TAU[m]; nk=int(m.sum())
        if nk<30:
            ax.text(0.5,0.5,f'd={d}: n={nk}',ha='center',va='center',transform=ax.transAxes)
            ax.set_xlim(0,MP); ax.set_ylim(0,1.02); continue
        A=np.array([(tau>=t).sum()/nk for t in range(MP+1)])
        ax.fill_between(ts,0,A,color=SUFFIX_COLOR,alpha=0.13)
        ax.plot(ts,A,color=SUFFIX_COLOR,lw=2.4,marker='o',ms=3)
        ax.text(0.96,0.92,f'anchor=d: N={nk:,}\nmean τ={tau.mean():.2f}',transform=ax.transAxes,ha='right',va='top',
                fontsize=10,fontweight='bold',bbox=dict(boxstyle='round',fc='white',ec=SUFFIX_COLOR,alpha=0.85))
        ax.set_title(f'eagle3 accepted to depth d={d}',fontsize=11,fontweight='bold')
        ax.set_xlim(0,MP); ax.set_ylim(0,1.02); ax.grid(alpha=0.3)
        if d%3==0: ax.set_ylabel('suffix survival  A(τ | anchor=d)')
        if d//3==2: ax.set_xlabel('τ  (suffix tokens beyond anchor)')
    fig.suptitle(f'{title} — SUFFIX-decoding conditional survival per anchor depth\n'
                 f'(condition: eagle3 accepted exactly to depth d; y=P(suffix tail reaches τ | anchor=d), '
                 f'NOT multiplied by eagle3 reach prob)',y=1.005,fontsize=12)
    plt.tight_layout(); plt.savefig(OUT/fname,dpi=110,bbox_inches='tight'); plt.close(fig); print('  saved',fname)

t0=time.time()
for mk,(mt,sub) in MODELS.items():
    for wl,wll in WL.items():
        path=R/sub/'anchor_depth'/f'per_step_{wl}.jsonl'
        if not path.exists(): print('MISSING',path); continue
        K,TAU,PPT,SC=extract(path); N=len(K)
        title=f'{mt} / {wll} (N={N:,})'
        print(f'{mk}/{wl}: N={N:,} @{time.time()-t0:.0f}s')
        bars(K,TAU,SC,'suffix score (suffix_cum_prob)',SUFFIX_COLOR,title,f'{mk}_{wl}_survival_vs_score.png')
        bars(K,TAU,PPT,'path probability (path_draft_p_t)',EAGLE3_COLOR,title,f'{mk}_{wl}_survival_vs_pathprob.png')
        suffix_cond(K,TAU,title,f'{mk}_{wl}_suffix_cond_survival.png')
print('ALL DONE')
