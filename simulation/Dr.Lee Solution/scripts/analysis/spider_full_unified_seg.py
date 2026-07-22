# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json, os, re, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
for p in ("/workspace/simulation/Dr.Lee Solution/scripts","/workspace",
          "/workspace/vendor/ddtree","/workspace/vendor/ddtree/model",
          "/workspace/simulation/scripts/experiments"):
    sys.path.insert(0, p)
from transformers import AutoTokenizer
from measure_k_fusion import ArcticSuffix
from capture_perpos import suffix_sweep
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")

GT="/workspace/simulation/results/spider_dbt_full_traj/qwen35_27b_dflash/gt_tokens.jsonl"
gt=[json.loads(l) for l in open(GT) if l.strip()]

# conv split: reset = large prompt-length drop (robust; all thresholds agree -> 71 convs)
conv_of=[]; cur=-1; prev=None
for r in gt:
    pt=len(r["input_ids"])
    if prev is None or pt < prev*0.8: cur+=1
    conv_of.append(cur); prev=pt
nconv=cur+1

NUMSPEC=32; DROP=8192
warm=[]; ev=[]
for r,c in zip(gt,conv_of):
    if len(r.get("output_ids") or [])>=DROP: continue     # runaway guard
    (warm if c%2==0 else ev).append(r)
warm_traces=[r["output_ids"] for r in warm if r.get("output_ids")]
suf_w=ArcticSuffix(); suf_w.fit(warm_traces)               # UNIFIED tree (all segments)
suf_c=ArcticSuffix()                                       # cold (self-context only)
print(f"[full] rows={len(gt)} convs={nconv} warm_rows={len(warm)} eval_rows={len(ev)}")

def offsets(ids):
    ps=[tok.decode([t],skip_special_tokens=False) for t in ids]; offs=[]; c=0
    for p in ps: offs.append((c,c+len(p))); c+=len(p)
    return offs,"".join(ps)

def tag_spider(full, offs):
    # 4-way: think / preamble (text before tool_call) / tool_call / final (text w/ no tool_call)
    n=len(offs); cats=["final"]*n
    et=full.find("</think>"); body0=0
    if et!=-1:
        end=et+len("</think>"); body0=end
        for i in range(n):
            if offs[i][1]<=end: cats[i]="think"
    m=re.search(r"Action:\s*[A-Za-z]\w*\(", full[body0:])
    if m:
        a0=body0+m.start(); a1=full.find("<|im_end|>",a0); a1=a1 if a1!=-1 else len(full)
        for i in range(n):
            s,e=offs[i]
            if cats[i]=="think": continue
            if s>=a0 and e<=a1: cats[i]="tool_call"
            elif e<=a0: cats[i]="preamble"
            else: cats[i]="final"
    return cats

agg={}; nth=0
for r in ev:
    gtk=r["output_ids"]; pid=r["input_ids"]
    if not gtk: continue
    _,mlw=suffix_sweep(suf_w,pid,gtk,NUMSPEC)
    _,mlc=suffix_sweep(suf_c,pid,gtk,NUMSPEC)
    offs,full=offsets(gtk)
    if "</think>" in full: nth+=1
    cats=tag_spider(full,offs)
    for p in range(len(gtk)):
        cat=cats[p] if p<len(cats) else "final"
        a=agg.setdefault(cat,[0,0,0]); a[0]+=1
        a[1]+=(mlw[p] if p<len(mlw) else 0); a[2]+=(mlc[p] if p<len(mlc) else 0)
tot=sum(a[0] for a in agg.values())
print(f"eval traces with </think>={nth}/{len(ev)}")
print(f"\n=== thinking-ON spider FULL ({len(gt)} rows / {nconv} convs)  UNIFIED suffix tree, per-segment, CPU ===")
print(f"  {'segment':<10}{'n':>9}{'%pos':>7}{'Suffix(w)':>11}{'Suffix(c)':>11}")
for cat in ["think","preamble","tool_call","final"]:
    if cat in agg:
        n,sw,sc=agg[cat]; print(f"  {cat:<10}{n:>9}{100*n/tot:>6.1f}%{sw/n:>11.3f}{sc/n:>11.3f}")
n=tot; sw=sum(a[1] for a in agg.values()); sc=sum(a[2] for a in agg.values())
print(f"  {'ALL':<10}{n:>9}{100.0:>6.1f}%{sw/n:>11.3f}{sc/n:>11.3f}")
