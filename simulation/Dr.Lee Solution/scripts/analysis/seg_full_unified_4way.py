# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json, os, re, sys
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
for p in ("/workspace/simulation/Dr.Lee Solution/scripts","/workspace",
          "/workspace/vendor/ddtree","/workspace/vendor/ddtree/model",
          "/workspace/simulation/scripts/experiments"):
    sys.path.insert(0, p)
from transformers import AutoTokenizer
from measure_k_fusion import ArcticSuffix
from capture_perpos import suffix_sweep
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
R="/workspace/simulation/results/"
WL={
 "bfcl":    (R+"bfcl_v4_full_traj/qwen35_27b_dflash/gt_tokens.jsonl","bfcl"),
 "swebench":(R+"swebench_full_traj/qwen35_27b_dflash/gt_tokens.jsonl","swebench"),
 "spider":  (R+"spider_dbt_full_traj/qwen35_27b_dflash/gt_tokens.jsonl","spider"),
}
NUMSPEC=32; DROP=8192

def offsets(ids):
    ps=[tok.decode([t],skip_special_tokens=False) for t in ids]; offs=[]; c=0
    for p in ps: offs.append((c,c+len(p))); c+=len(p)
    return offs,"".join(ps)

def tag(full, offs, kind):
    n=len(offs); cats=["final"]*n; body0=0
    et=full.find("</think>")
    if et!=-1:
        end=et+len("</think>"); body0=end
        for i in range(n):
            if offs[i][1]<=end: cats[i]="think"
    tc=None
    if kind=="bfcl":
        m=re.search(r"\[\s*[A-Za-z_]\w*\s*\(", full[body0:])
        if m:
            a0=body0+m.start(); a1=full.rfind("]"); a1=(a1+1) if a1>a0 else len(full); tc=(a0,a1)
    elif kind=="swebench":
        ms=list(re.finditer(r"```mswea_bash_command.*?```", full, re.DOTALL))
        if ms: tc=(ms[0].start(), ms[-1].end())
    elif kind=="spider":
        m=re.search(r"Action:\s*[A-Za-z]\w*\(", full[body0:])
        if m:
            a0=body0+m.start(); a1=full.find("<|im_end|>",a0); a1=a1 if a1!=-1 else len(full); tc=(a0,a1)
    for i in range(n):
        if cats[i]=="think": continue
        s,e=offs[i]
        if tc and s>=tc[0] and e<=tc[1]: cats[i]="tool_call"
        elif tc and e<=tc[0]: cats[i]="preamble"
        else: cats[i]="final"
    return cats

for wl,(fn,kind) in WL.items():
    gt=[json.loads(l) for l in open(fn) if l.strip()]
    conv=[]; cur=-1; prev=None
    for r in gt:
        pt=len(r["input_ids"])
        if prev is None or pt<prev*0.8: cur+=1
        conv.append(cur); prev=pt
    warm=[]; ev=[]
    for r,c in zip(gt,conv):
        if len(r.get("output_ids") or [])>=DROP: continue
        (warm if c%2==0 else ev).append(r)
    suf_w=ArcticSuffix(); suf_w.fit([r["output_ids"] for r in warm if r.get("output_ids")])
    suf_c=ArcticSuffix()
    agg={}; nth=0
    for r in ev:
        gtk=r["output_ids"]; pid=r["input_ids"]
        if not gtk: continue
        _,mlw=suffix_sweep(suf_w,pid,gtk,NUMSPEC)
        _,mlc=suffix_sweep(suf_c,pid,gtk,NUMSPEC)
        offs,full=offsets(gtk)
        if "</think>" in full: nth+=1
        cats=tag(full,offs,kind)
        for p in range(len(gtk)):
            cat=cats[p] if p<len(cats) else "final"
            a=agg.setdefault(cat,[0,0,0]); a[0]+=1
            a[1]+=(mlw[p] if p<len(mlw) else 0); a[2]+=(mlc[p] if p<len(mlc) else 0)
    tot=sum(a[0] for a in agg.values())
    print(f"\n===== {wl} ({kind})  rows={len(gt)} convs={cur+1} warm={len(warm)} eval={len(ev)}  think_traces={nth}/{len(ev)} =====")
    print(f"  {'segment':<10}{'n':>9}{'%pos':>7}{'Suffix(w)':>11}{'Suffix(c)':>11}")
    for cat in ["think","preamble","tool_call","final"]:
        if cat in agg:
            n,sw,sc=agg[cat]; print(f"  {cat:<10}{n:>9}{100*n/tot:>6.1f}%{sw/n:>11.3f}{sc/n:>11.3f}")
    n=tot; sw=sum(a[1] for a in agg.values()); sc=sum(a[2] for a in agg.values())
    print(f"  {'ALL':<10}{n:>9}{100.0:>6.1f}%{sw/n:>11.3f}{sc/n:>11.3f}")
