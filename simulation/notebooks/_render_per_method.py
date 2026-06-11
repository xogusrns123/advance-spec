import json, numpy as np, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
R=Path('/home/muchwater/advance-spec/simulation/results')
OUT=Path('/home/muchwater/advance-spec/simulation/notebooks/figures/per_method'); OUT.mkdir(parents=True,exist_ok=True)
MAXP=16; MIN=50
A14=R/'explorations_qwen3_14b'/'anchor_depth'; A27=R/'explorations_qwen35_27b'/'anchor_depth'
WL_TITLE={'bfcl_v4':'BFCLv4','specbench':'SpecBench','swebench_verified':'SWE-Bench Verified'}
DM_FILE={'bfcl_v4':'per_step_dm_bfcl.jsonl','specbench':'per_step_dm_specbench.jsonl',
         'swebench_verified':'per_step_dm_swebench_verified.jsonl'}
def per_method_specs(wl):
    pf=f'per_step_{wl}.jsonl'; dm=DM_FILE[wl]
    return [
     ('eagle3_qwen3_14b',   'EAGLE3 (Qwen3-14B)',      A14/pf, 'single:eagle3',        '#1f77b4'),
     ('suffix_qwen3_14b',   'Suffix (Qwen3-14B)',      A14/pf, 'single:suffix:4.0:0.0','#ff7f0e'),
     ('suffix_qwen35_27b',  'Suffix (Qwen3.5-27B)',    A27/pf, 'single:suffix:4.0:0.0','#ff7f0e'),
     ('draftmodel_qwen3_14b','Draft Model (Qwen3-14B, linear chain)', A14/dm,'single:draft_model','#2ca02c'),
     ('mtp_qwen35_27b',     'MTP (Qwen3.5-27B)',       A27/pf, 'single:eagle3',        '#d62728'),
    ]
def combined_specs(wl):   # Suffix = 14B only
    pf=f'per_step_{wl}.jsonl'; dm=DM_FILE[wl]
    return [
     ('EAGLE3 (Qwen3-14B)',                 A14/pf, 'single:eagle3',        '#1f77b4'),
     ('Suffix (Qwen3-14B)',                 A14/pf, 'single:suffix:4.0:0.0','#ff7f0e'),
     ('Draft Model (Qwen3-14B, linear chain)', A14/dm,'single:draft_model', '#2ca02c'),
     ('MTP (Qwen3.5-27B)',                  A27/pf, 'single:eagle3',        '#d62728'),
    ]
def load_L(path,method):
    if not Path(path).exists(): return None
    L=[]
    for line in open(path):
        try: r=json.loads(line)
        except: continue
        if r.get('method')==method: L.append(int(r.get('accepted',0)))
    return np.asarray(L) if L else None
def survival(L):
    n=len(L); return np.array([(L>=p).sum()/n for p in range(MAXP+1)])
def conditional(L):
    cnt=np.array([(L>=p).sum() for p in range(MAXP+1)],float)
    c=np.full(MAXP+1,np.nan); c[0]=1.0
    for p in range(1,MAXP+1):
        if cnt[p-1]>=MIN: c[p]=cnt[p]/cnt[p-1]
    return c
xs=np.arange(MAXP+1)
xt=lambda pm: 1 if pm<=10 else 2

WLS=sys.argv[1:] or ['bfcl_v4','specbench','swebench_verified']
for wl in WLS:
    title=WL_TITLE[wl]
    # per-method separate
    for slug,label,path,m,col in per_method_specs(wl):
        L=load_L(path,m)
        if L is None: print('MISSING',wl,label,path,m); continue
        pmax=int(min(L.max(),MAXP))
        for metric,mname,ylab in [(survival,'survival',f'survival rate  $A(p)=P(L\\geq p)$'),
                                  (conditional,'conditional',f'conditional accept rate  $a_p=A(p)/A(p-1)$')]:
            Y=metric(L)
            plt.figure(figsize=(8,5.5))
            plt.plot(xs[:pmax+1],Y[:pmax+1],color=col,lw=2.6,marker='o',ms=5)
            plt.xlabel('depth $p$'); plt.ylabel(ylab); plt.title(f'{title} — {label}: {mname} by depth')
            plt.xlim(0,pmax); plt.ylim(0,1.02); plt.grid(alpha=0.3); plt.xticks(range(0,pmax+1,xt(pmax)))
            plt.tight_layout(); plt.savefig(OUT/f'per_method_{wl}_{slug}_{mname}.png',dpi=130,bbox_inches='tight'); plt.close()
        print(f'  {wl}/{slug}: n={len(L):,} pmax={pmax}')
    # combined
    for metric,mname,ylab in [(survival,'survival','survival rate  $A(p)=P(L\\geq p)$'),
                              (conditional,'conditional','conditional accept rate  $a_p=A(p)/A(p-1)$')]:
        plt.figure(figsize=(9,6)); pmx=0
        for lbl,path,m,col in combined_specs(wl):
            L=load_L(path,m)
            if L is None: continue
            pm=int(min(L.max(),MAXP)); pmx=max(pmx,pm); Y=metric(L)
            plt.plot(xs[:pm+1],Y[:pm+1],color=col,lw=2.6,marker='o',ms=4,label=lbl)
        plt.xlabel('depth $p$'); plt.ylabel(ylab)
        plt.title(f'{title} — per-method {mname} by depth (Suffix=14B; Draft=linear chain)')
        plt.xlim(0,pmx); plt.ylim(0,1.02); plt.grid(alpha=0.3); plt.xticks(range(0,pmx+1,2)); plt.legend(fontsize=9)
        plt.tight_layout(); plt.savefig(OUT/f'per_method_{wl}_combined_{mname}.png',dpi=130,bbox_inches='tight'); plt.close()
    print(f'  {wl}: combined saved')
print('DONE')
