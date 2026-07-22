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
B = "/workspace/simulation/Dr.Lee Solution/"
WL = {
 "bfcl":     ("perpos_bfcl_full/bfcl_v4_full", "bfcl", ["think","action","text"]),
 "specbench":("perpos_specbench_full/specbench", "specbench", ["think","text"]),
 "swebench": ("perpos_swebench/swebench_quick", "swebench", ["action","text"]),
 "spider":   ("perpos_spider/spider", "spider", ["action","text"]),
}

def offsets(ids):
    ps=[tok.decode([t],skip_special_tokens=False) for t in ids]; offs=[]; c=0
    for p in ps: offs.append((c,c+len(p))); c+=len(p)
    return offs,"".join(ps)

def tag(full, offs, kind):
    n=len(offs); cats=["text"]*n; body0=0
    et=full.find("</think>")
    if et!=-1:
        end=et+len("</think>")
        for i in range(n):
            if offs[i][1]<=end: cats[i]="think"
        body0=end
    spans=[]
    if kind=="bfcl":
        m=re.search(r"\[\s*[A-Za-z_]\w*\s*\(", full[body0:])
        if m:
            a0=body0+m.start(); a1=full.rfind("]"); a1=(a1+1) if a1>=a0 else len(full)
            spans.append((a0,a1))
    elif kind=="swebench":
        for m in re.finditer(r"```mswea_bash_command.*?```", full, re.DOTALL):
            spans.append((m.start(),m.end()))
    elif kind=="spider":
        m=re.search(r"Action:\s*Bash\(", full)
        if m:
            a1=full.find("<|im_end|>",m.start()); a1=a1 if a1!=-1 else len(full)
            spans.append((m.start(),a1))
    for (a0,a1) in spans:
        for i in range(n):
            s,e=offs[i]
            if s>=a0 and e<=a1 and cats[i]!="think": cats[i]="action"
    return cats

for wl,(stem,kind,segs) in WL.items():
    tr=json.load(open(B+"results/"+stem+".traces.json"))
    ns=tr.get("num_spec",32)
    def tagged(ids):
        offs,full=offsets(ids); return tag(full,offs,kind)
    warm=[(ids, tagged(ids)) for ids in tr["warm_traces"]]
    evl =[(t["output_ids"], t["rid"], tagged(t["output_ids"])) for t in tr["eval_traces"]]
    cats_by_rid={rid:cats for (_,rid,cats) in evl}
    # ---- (A) isolated per-segment tree ----
    A={}
    for S in segs:
        warm_S=[[t for t,c in zip(ids,cats) if c==S] for (ids,cats) in warm]
        warm_S=[s for s in warm_S if s]
        suf=ArcticSuffix(); suf.fit(warm_S)
        tot=cnt=0
        for (ids,rid,cats) in evl:
            seq=[t for t,c in zip(ids,cats) if c==S]
            if not seq: continue
            _,mls=suffix_sweep(suf,[],seq,ns)
            tot+=sum(mls); cnt+=len(mls)
        A[S]=(tot/cnt if cnt else 0.0, cnt)
    # ---- (B) shared tree: bucket suffix_match_warm from .jsonl ----
    Bd={}
    for line in open(B+"results/"+stem+".jsonl"):
        line=line.strip()
        if not line: continue
        r=json.loads(line); rid=r["rid"]; gi=r["pos"]-1
        cats=cats_by_rid.get(rid)
        if cats is None or not (0<=gi<len(cats)): continue
        c=cats[gi]; d=Bd.setdefault(c,[0,0]); d[0]+=1; d[1]+=r.get("suffix_match_warm",0)
    print(f"\n===== {wl} ({kind}) num_spec={ns} =====")
    print(f"  {'seg':<8}{'nA':>8}{'(A) isolated':>14}   {'nB':>8}{'(B) shared':>12}   {'A/B':>6}")
    for S in segs:
        a,na=A[S]; nb,sb=Bd.get(S,[0,0]); b=(sb/nb if nb else 0.0)
        print(f"  {S:<8}{na:>8}{a:>14.3f}   {nb:>8}{b:>12.3f}   {(a/b if b else 0):>6.2f}")
