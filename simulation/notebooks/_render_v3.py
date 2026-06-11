import json, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
R=Path('/home/muchwater/advance-spec/simulation/results')
OUT=Path('/home/muchwater/advance-spec/simulation/notebooks/figures/anchor_depth')
ORANGE,PURPLE,BLUE='#ff7f0e','#7b1fa2','#1f77b4'
MP=24; ts=np.arange(MP+1)
# model key from argv (default qwen3_14b)
MK=sys.argv[1] if len(sys.argv)>1 else 'qwen3_14b'
CFG={'qwen3_14b':('explorations_qwen3_14b','Qwen3-14B','EAGLE3'),
     'qwen35_27b':('explorations_qwen35_27b','Qwen3.5-27B','MTP')}
sub,mt,bb=CFG[MK]
DUMP=R/sub/'anchor_depth'/'per_step_stem_bfcl_v4.jsonl'
SUF='single:suffix:4.0:0.0'
STEM=[f'extension_stem:{d}:4.0:0.0' for d in range(9)]
WANT=set([SUF]+STEM)
def surv(a):
    a=np.asarray(a); n=len(a)
    return np.array([(a>=p).sum()/n for p in range(MP+1)]) if n else np.full(MP+1,np.nan)
def MAT(A): return float(A[1:].sum())

rec=defaultdict(dict)
for line in open(DUMP):
    r=json.loads(line); m=r['method']
    if m in WANT:
        rec[(r['request_id'],r.get('call_idx',0),r['step_id'])][m]=int(r.get('accepted',0))
acc={m:[] for m in WANT}
for v in rec.values():
    if all(m in v for m in WANT):
        for m in WANT: acc[m].append(v[m])
acc={m:np.asarray(a) for m,a in acc.items()}
N=len(acc[SUF]); print(f'{MK}: N(joined)={N:,}',flush=True)

# INVARIANT: extension_stem:0 must equal single:suffix per step (no backbone -> pure suffix)
eq=int((acc[STEM[0]]==acc[SUF]).sum())
print(f'{MK}: INVARIANT stem:0==suffix : {eq}/{N} ({100*eq/max(N,1):.2f}%)  '
      f'mean stem0={acc[STEM[0]].mean():.3f} suffix={acc[SUF].mean():.3f}',flush=True)

As=surv(acc[SUF]); ms=MAT(As)
fig,axes=plt.subplots(3,3,figsize=(16,12),sharex=True,sharey=True)
for d in range(9):
    ax=axes[d//3][d%3]; Ax=surv(acc[STEM[d]]); mx=MAT(Ax)
    ax.plot(ts,As,color=ORANGE,lw=1.8,marker='s',ms=2.5,label=f'Suffix-only (all)  area={ms:.2f}')
    # Extension curve split by region: p<=d is the EAGLE3 backbone (blue),
    # p>d is the grafted suffix stem (purple). Both are ACTUAL survival over
    # all steps (the backbone region follows real eagle3 reach, NOT 1.0).
    if d > 0:
        ax.fill_between(ts[:d+1],0,Ax[:d+1],color=BLUE,alpha=0.16)
        ax.plot(ts[:d+1],Ax[:d+1],color=BLUE,lw=2.8,marker='o',ms=3.5,label=f'{bb} backbone (p≤{d})')
    ax.fill_between(ts[d:],0,Ax[d:],color=PURPLE,alpha=0.13)
    ax.plot(ts[d:],Ax[d:],color=PURPLE,lw=2.6,marker='D',ms=3,label='suffix stem (p>d)')
    ax.axvline(d,color='gray',ls=':',lw=1.2)
    ax.text(0.96,0.92,f'Ext area={mx:.2f}\n Suffix area={ms:.2f}\n diff {mx-ms:+.2f}',transform=ax.transAxes,
            ha='right',va='top',fontsize=10,fontweight='bold',bbox=dict(boxstyle='round',fc='white',ec=PURPLE,alpha=0.85))
    ttl=f'd={d}: {d} {bb} tokens then suffix stem'+('   (= suffix-only)' if d==0 else '')
    ax.set_title(ttl,fontsize=10,fontweight='bold')
    ax.set_xlim(0,MP); ax.set_ylim(0,1.02); ax.grid(alpha=0.3)
    if d%3==0: ax.set_ylabel('survival A(p)  (all steps)')
    if d//3==2: ax.set_xlabel('position p')
    if d==0: ax.legend(loc='upper right',fontsize=8)
fig.suptitle(f'{mt} / BFCLv4 (N={N:,}) — Extension = d {bb} backbone tokens then SUFFIX STEM only '
             f'(no backbone beyond d), over ALL steps. Suffix-only reference (orange). d=0 ≡ suffix-only.',
             y=1.005,fontsize=12)
plt.tight_layout(); fp=OUT/f'{MK}_bfcl_v4_v3_stem_survival.png'
plt.savefig(fp,dpi=120,bbox_inches='tight'); print('saved',fp,flush=True); print('DONE',flush=True)
