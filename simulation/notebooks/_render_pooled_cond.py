import os, json, numpy as np, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
BASE=Path(os.environ.get('BASE','/home/muchwater/advance-spec'))
R=BASE/'simulation'/'results'
OUT=BASE/'simulation'/'notebooks'/'figures'/'per_method'; OUT.mkdir(parents=True,exist_ok=True)
MAXP=16; MIN=50
A14=R/'explorations_qwen3_14b'/'anchor_depth'; A27=R/'explorations_qwen35_27b'/'anchor_depth'
WLS=['bfcl_v4','specbench','swebench_verified']
DM_FILE={'bfcl_v4':'per_step_dm_bfcl.jsonl','specbench':'per_step_dm_specbench.jsonl',
         'swebench_verified':'per_step_dm_swebench_verified.jsonl'}

# Four methods (pooled across all three workloads). Each entry:
#   label, color, list of (path, jsonl-method) to pool over the 3 workloads.
SPECS=[
 ('EAGLE3 (Qwen3-14B)',     '#1f77b4', [(A14/f'per_step_{wl}.jsonl',      'single:eagle3')         for wl in WLS]),
 ('Suffix (Qwen3-14B)',     '#ff7f0e', [(A14/f'per_step_{wl}.jsonl',      'single:suffix:4.0:0.0') for wl in WLS]),
 ('Draft Model (Qwen3-14B)','#2ca02c', [(A14/DM_FILE[wl],                 'single:draft_model')    for wl in WLS]),
 ('MTP (Qwen3.5-27B)',      '#d62728', [(A27/f'per_step_{wl}.jsonl',      'single:eagle3')         for wl in WLS]),
]

# Build pooled accepted-length histograms. Each per_step file is read once and
# every method-of-interest present in it is tallied in the same pass.
files={}  # path -> set of methods we need from it
for _,_,parts in SPECS:
    for path,m in parts: files.setdefault(str(path),set()).add(m)

hist={}  # (path,method) -> np.array histogram over accepted length (0..MAXP, last bin clips)
for path,methods in files.items():
    if not Path(path).exists():
        print('MISSING FILE',path); continue
    h={m:np.zeros(MAXP+1,np.int64) for m in methods}
    n=0
    for line in open(path):
        try: r=json.loads(line)
        except: continue
        m=r.get('method')
        if m in h:
            a=int(r.get('accepted',0))
            if a<0: a=0
            if a>MAXP: a=MAXP
            h[m][a]+=1; n+=1
    for m in methods: hist[(path,m)]=h[m]
    print(f'read {Path(path).name}: {n:,} matched rows  ({", ".join(sorted(methods))})')

def pooled_counts(parts):
    # cnt[p] = #samples with accepted-length >= p, pooled across workloads
    hh=np.zeros(MAXP+1,np.int64)
    for path,m in parts:
        h=hist.get((str(path),m))
        if h is not None: hh+=h
    N=int(hh.sum())
    cnt=np.array([hh[p:].sum() for p in range(MAXP+1)],float)  # survival counts
    return cnt,N

def conditional(cnt):
    c=np.full(MAXP+1,np.nan); c[0]=1.0
    for p in range(1,MAXP+1):
        if cnt[p-1]>=MIN: c[p]=cnt[p]/cnt[p-1]
    return c

def survival(cnt):
    N=cnt[0] if cnt[0]>0 else 1.0
    return cnt/N

xs=np.arange(MAXP+1)
for metric,mname,ylab in [(conditional,'conditional','conditional accept rate  $a_p=A(p)/A(p-1)$'),
                          (survival,'survival','survival rate  $A(p)=P(L\\geq p)$')]:
    plt.figure(figsize=(9,6)); pmx=0
    for lbl,col,parts in SPECS:
        cnt,N=pooled_counts(parts)
        if N==0: print('NO DATA',lbl); continue
        # last depth with >=MIN samples surviving (so curve isn't drawn on tiny tails)
        pm=max([p for p in range(MAXP+1) if cnt[p]>=MIN], default=0)
        pmx=max(pmx,pm)
        Y=metric(cnt)
        plt.plot(xs[:pm+1],Y[:pm+1],color=col,lw=2.6,marker='o',ms=4,label=f'{lbl}  (N={N:,})')
    plt.xlabel('depth $p$'); plt.ylabel(ylab)
    plt.title(f'Pooled over BFCLv4 + SpecBench + SWE-Bench Verified — per-method {mname} by depth')
    plt.xlim(0,pmx); plt.ylim(0,1.02); plt.grid(alpha=0.3)
    plt.xticks(range(0,pmx+1,1 if pmx<=10 else 2)); plt.legend(fontsize=9)
    plt.tight_layout()
    out=OUT/f'per_method_pooled3wl_{mname}.png'
    plt.savefig(out,dpi=130,bbox_inches='tight'); plt.close()
    print('saved',out)
print('DONE')
