import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
R=Path('/home/muchwater/advance-spec/simulation/results')
OUT=Path('/home/muchwater/advance-spec/simulation/notebooks/figures/anchor_depth')
BLUE,ORANGE,PURPLE='#1f77b4','#ff7f0e','#7b1fa2'
MP=24; ts=np.arange(MP+1)
def surv(a):
    a=np.asarray(a); n=len(a)
    return np.array([(a>=p).sum()/n for p in range(MP+1)]) if n else np.full(MP+1,np.nan)
def MAT(A): return float(A[1:].sum())

def load(sub,wl):
    p=R/sub/'anchor_depth'/f'per_step_{wl}.jsonl'
    rec=defaultdict(dict)
    for line in open(p):
        r=json.loads(line); m=r['method']; key=(r['request_id'],r.get('call_idx',0),r['step_id'])
        if m=='extension:4.0:0.0':
            src=r.get('tree_source') or []; acc=r.get('tree_is_accepted') or []
            if not src: continue
            k=sum(1 for i in range(len(acc)) if acc[i] and i<len(src) and src[i]=='eagle')
            rec[key]['k']=k; rec[key]['ext']=int(r.get('accepted',0))
        elif m=='single:eagle3': rec[key]['eg']=int(r.get('accepted',0))
        elif m=='single:suffix:4.0:0.0': rec[key]['sf']=int(r.get('accepted',0))
    G=defaultdict(lambda:{'eg':[],'sf':[],'ext':[]})
    for v in rec.values():
        if 'k' not in v or 'eg' not in v or 'sf' not in v: continue
        if v['k']<=8:
            G[v['k']]['eg'].append(v['eg']); G[v['k']]['sf'].append(v['sf']); G[v['k']]['ext'].append(v['ext'])
    return {k:{m:np.asarray(a) for m,a in g.items()} for k,g in G.items()}

def v1(G,bb,title,fname):
    fig,axes=plt.subplots(3,3,figsize=(16,12),sharex=True,sharey=True)
    for d in range(9):
        ax=axes[d//3][d%3]; g=G.get(d); nk=len(g['ext']) if g else 0
        if nk<30:
            ax.text(0.5,0.5,f'd={d}: n={nk}',ha='center',va='center',transform=ax.transAxes)
            ax.set_xlim(0,MP); ax.set_ylim(0,1.02); continue
        A=surv(g['ext']-d); mt=MAT(A)
        ax.fill_between(ts,0,A,color=PURPLE,alpha=0.13)
        ax.plot(ts,A,color=PURPLE,lw=2.6,marker='D',ms=3)
        ax.text(0.96,0.92,f'N={nk:,}\narea(mean τ)={mt:.2f}',transform=ax.transAxes,ha='right',va='top',
                fontsize=10,fontweight='bold',bbox=dict(boxstyle='round',fc='white',ec=PURPLE,alpha=0.85))
        ax.set_title(f'eagle3 accepted to depth d={d}',fontsize=11,fontweight='bold')
        ax.set_xlim(0,MP); ax.set_ylim(0,1.02); ax.grid(alpha=0.3)
        if d%3==0: ax.set_ylabel('suffix survival  A(τ | anchor=d)')
        if d//3==2: ax.set_xlabel('τ  (suffix tokens beyond anchor)')
    fig.suptitle(f'{title} — SUFFIX-decoding conditional survival per anchor depth (extension only)',y=1.005,fontsize=12)
    plt.tight_layout(); plt.savefig(OUT/fname,dpi=110,bbox_inches='tight'); plt.close(fig); print('saved',fname,flush=True)

def v2(G,bb,title,fname):
    # EAGLE3-only and Suffix-only are root-start single methods -> MARGINAL
    # survival over ALL steps (identical in every panel). Only Extension is
    # conditioned on anchor depth d (assumed accepted up to the anchor).
    eg_all=np.concatenate([g['eg'] for g in G.values()])
    sf_all=np.concatenate([g['sf'] for g in G.values()])
    Ae=surv(eg_all); As=surv(sf_all); me,ms=MAT(Ae),MAT(As)
    fig,axes=plt.subplots(3,3,figsize=(16,12),sharex=True,sharey=True)
    for d in range(9):
        ax=axes[d//3][d%3]; g=G.get(d); nk=len(g['ext']) if g else 0
        if nk<30:
            ax.text(0.5,0.5,f'd={d}: n={nk}',ha='center',va='center',transform=ax.transAxes)
            ax.set_xlim(0,MP); ax.set_ylim(0,1.02); continue
        Ax=surv(g['ext']-d); mx=MAT(Ax)        # extension = SUFFIX TAIL only (exclude eagle3 nodes), x=τ
        ax.fill_between(ts,0,Ax,color=PURPLE,alpha=0.13)
        ax.plot(ts,Ae,color=BLUE,lw=2.0,marker='o',ms=3,label=f'{bb} only (all)')
        ax.plot(ts,As,color=ORANGE,lw=2.0,marker='s',ms=3,label='Suffix only (all)')
        ax.plot(ts,Ax,color=PURPLE,lw=2.6,marker='D',ms=3,label='Extension suffix-tail (anchor=d)')
        ax.text(0.96,0.92,f'Ext τ area={mx:.2f}\n {bb}(all)={me:.2f}\n Suffix(all)={ms:.2f}',transform=ax.transAxes,
                ha='right',va='top',fontsize=10,fontweight='bold',bbox=dict(boxstyle='round',fc='white',ec=PURPLE,alpha=0.85))
        ax.set_title(f'eagle3 accepted to depth d={d}  (N={nk:,})',fontsize=11,fontweight='bold')
        ax.set_xlim(0,MP); ax.set_ylim(0,1.02); ax.grid(alpha=0.3)
        if d%3==0: ax.set_ylabel('survival')
        if d//3==2: ax.set_xlabel('position  (Extension: τ beyond anchor; singles: from root)')
        if d==0: ax.legend(loc='upper right',fontsize=8)
    fig.suptitle(f'{title} — Extension SUFFIX-TAIL (eagle3 nodes excluded, anchor=d) vs '
                 f'{bb}-only / Suffix-only (MARGINAL reference)',y=1.005,fontsize=12)
    plt.tight_layout(); plt.savefig(OUT/fname,dpi=110,bbox_inches='tight'); plt.close(fig); print('saved',fname,flush=True)

G=load('explorations_qwen3_14b','bfcl_v4')
N=sum(len(g['ext']) for g in G.values())
title=f'Qwen3-14B / BFCLv4 (N={N:,})'
v1(G,'EAGLE3',title,'qwen3_14b_bfcl_v4_suffix_cond_survival.png')
v2(G,'EAGLE3',title,'qwen3_14b_bfcl_v4_suffix_cond_with_singles.png')
print("DONE",flush=True)
