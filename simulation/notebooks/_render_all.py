import json, numpy as np, matplotlib, time
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
R=Path('/home/muchwater/advance-spec/simulation/results')
OUT=Path('/home/muchwater/advance-spec/simulation/notebooks/figures/anchor_depth')
BLUE,ORANGE,PURPLE,GREY='#1f77b4','#ff7f0e','#7b1fa2','#999999'
MP=24; xs=np.arange(MP+1); MIN=30
MB,MS,MX='single:eagle3','single:suffix:4.0:0.0','extension:4.0:0.0'
METHODS=[MB,MS,MX]+[f'extension_gd:{d}:4.0:0.0' for d in range(9)]+[f'extension_cumd:{d}:4.0:0.0' for d in range(9)]
MODELS={'qwen3_14b':('Qwen3-14B','EAGLE3'),'qwen35_27b':('Qwen3.5-27B','MTP')}
WL={'specbench':'SpecBench','bfcl_v4':'BFCLv4','swebench_verified':'SWE-Bench Verified'}
DUMPS={}
for mk,sub in [('qwen3_14b','explorations_qwen3_14b'),('qwen35_27b','explorations_qwen35_27b')]:
    for wl in WL:
        DUMPS[(mk,wl)]=R/sub/'anchor_depth'/f'per_step_alldepth_{mk}_{wl}.jsonl'
def surv(a): a=np.asarray(a); return np.array([(a>=p).sum()/len(a) for p in range(MP+1)])
def MAT(A): return float(A[1:].sum())
def cond(a):
    a=np.asarray(a); cnt=np.array([(a>=p).sum() for p in range(MP+1)],float)
    c=np.full(MP+1,np.nan); c[0]=1.0
    for p in range(1,MP+1):
        if cnt[p-1]>=MIN: c[p]=cnt[p]/cnt[p-1]
    return c
def load(path):
    rec=defaultdict(dict)
    for line in open(path):
        r=json.loads(line); m=r['method']
        if m in METHODS: rec[(r['request_id'],r.get('call_idx',0),r['step_id'])][m]=int(r.get('accepted',0))
    out={m:[] for m in METHODS}
    for v in rec.values():
        if all(m in v for m in METHODS):
            for m in METHODS: out[m].append(v[m])
    return {m:np.asarray(a) for m,a in out.items()}, len(out[MB])

def surv_grid(d,N,bb,title,prefix,label_fn,fname):
    Ae,As,Ax=surv(d[MB]),surv(d[MS]),surv(d[MX]); me,ms,mx=MAT(Ae),MAT(As),MAT(Ax)
    fig,axes=plt.subplots(3,3,figsize=(17,13),sharex=True,sharey=True)
    for gd in range(9):
        ax=axes[gd//3][gd%3]; Ag=surv(d[f'{prefix}:{gd}:4.0:0.0']); mg=MAT(Ag)
        ax.fill_between(xs,0,Ag,color=PURPLE,alpha=0.13,zorder=0)
        ax.plot(xs,Ax,color=GREY,lw=1.2,ls='--',label=f'full ext (area {mx:.2f})')
        ax.plot(xs,Ae,color=BLUE,lw=1.8,marker='o',ms=2.5,label=f'{bb} (area {me:.2f})')
        ax.plot(xs,As,color=ORANGE,lw=1.8,marker='s',ms=2.5,label=f'Suffix (area {ms:.2f})')
        ax.plot(xs,Ag,color=PURPLE,lw=2.6,marker='D',ms=2.5,label=f'{label_fn(gd)} (area {mg:.2f})')
        beats='beats BOTH' if (mg>me and mg>ms) else (f'>{bb} only' if mg>me else '-')
        ax.text(0.97,0.55,f'area={mg:.2f}\n vs{bb} +{mg-me:.2f}\n vsSfx {mg-ms:+.2f}',transform=ax.transAxes,
                ha='right',va='top',fontsize=10,fontweight='bold',bbox=dict(boxstyle='round',fc='white',ec=PURPLE,alpha=0.85))
        ax.set_title(f'{label_fn(gd)}   [{beats}]',fontsize=10); ax.set_xlim(0,MP); ax.set_ylim(0,1.02); ax.grid(alpha=0.3)
        if gd%3==0: ax.set_ylabel('survival A(p)')
        if gd//3==2: ax.set_xlabel('position p')
        if gd==0: ax.legend(loc='upper right',fontsize=7.5)
    fig.suptitle(f'{title} (N={N:,}) — {prefix.split("_")[-1]} survival; shaded=area(=MAT); '
                 f'{bb}={me:.2f}, Suffix={ms:.2f}, full ext={mx:.2f}',y=1.005,fontsize=12)
    plt.tight_layout(); plt.savefig(OUT/fname,dpi=110,bbox_inches='tight'); plt.close(fig)

def cond_grid(d,N,bb,title,prefix,label_fn,fname):
    Ce,Cs,Cx=cond(d[MB]),cond(d[MS]),cond(d[MX]); me,ms=MAT(surv(d[MB])),MAT(surv(d[MS]))
    fig,axes=plt.subplots(3,3,figsize=(17,13),sharex=True,sharey=True)
    for gd in range(9):
        ax=axes[gd//3][gd%3]; g=d[f'{prefix}:{gd}:4.0:0.0']; Cg=cond(g); mg=MAT(surv(g))
        ax.plot(xs,Cx,color=GREY,lw=1.2,ls='--',label='full ext')
        ax.plot(xs,Ce,color=BLUE,lw=1.8,marker='o',ms=2.5,label=bb)
        ax.plot(xs,Cs,color=ORANGE,lw=1.8,marker='s',ms=2.5,label='Suffix')
        ax.plot(xs,Cg,color=PURPLE,lw=2.6,marker='D',ms=2.5,label=label_fn(gd))
        ax.text(0.97,0.30,f'MAT={mg:.2f}\n(survival area)\n vs{bb} +{mg-me:.2f}\n vsSfx {mg-ms:+.2f}',transform=ax.transAxes,
                ha='right',va='top',fontsize=9.5,fontweight='bold',bbox=dict(boxstyle='round',fc='white',ec=PURPLE,alpha=0.85))
        ax.set_title(label_fn(gd),fontsize=10); ax.set_xlim(0,MP); ax.set_ylim(0,1.02); ax.grid(alpha=0.3)
        if gd%3==0: ax.set_ylabel('conditional accept a_p')
        if gd//3==2: ax.set_xlabel('position p')
        if gd==0: ax.legend(loc='lower left',fontsize=8)
    fig.suptitle(f'{title} (N={N:,}) — {prefix.split("_")[-1]} conditional a_p=A(p)/A(p-1) [supp>={MIN}]; '
                 f'MAT label=survival area (the metric, not conditional-curve area)',y=1.005,fontsize=11)
    plt.tight_layout(); plt.savefig(OUT/fname,dpi=110,bbox_inches='tight'); plt.close(fig)

GDL=lambda x:('root hybrid (graft@0)' if x==0 else f'graft@d={x}')
CUL=lambda x:('root hybrid (cum<=0)' if x==0 else f'cumulative <=depth {x}')
t0=time.time()
for (mk,wl),path in DUMPS.items():
    if not path.exists(): print('MISSING',path); continue
    bb=MODELS[mk][1]; title=f'{MODELS[mk][0]} / {WL[wl]}'
    d,N=load(path); print(f'{mk}/{wl}: N={N:,} loaded @{time.time()-t0:.0f}s',flush=True)
    surv_grid(d,N,bb,title,'extension_gd',GDL,f'{mk}_{wl}_graft_indep.png')
    cond_grid(d,N,bb,title,'extension_gd',GDL,f'{mk}_{wl}_graft_indep_cond.png')
    surv_grid(d,N,bb,title,'extension_cumd',CUL,f'{mk}_{wl}_cumulative.png')
    cond_grid(d,N,bb,title,'extension_cumd',CUL,f'{mk}_{wl}_cumulative_cond.png')
    print(f'  saved 4 figs for {mk}/{wl} @{time.time()-t0:.0f}s',flush=True)
print('ALL RENDER DONE',flush=True)
