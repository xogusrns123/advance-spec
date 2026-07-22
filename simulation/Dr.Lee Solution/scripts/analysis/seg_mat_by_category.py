# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json, os, re
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
B = "/workspace/simulation/Dr.Lee Solution/"

WL = {
 "bfcl":     ("perpos_bfcl_full/bfcl_v4_full", "bfcl"),
 "specbench":("perpos_specbench_full/specbench", "specbench"),
 "swebench": ("perpos_swebench/swebench_quick", "swebench"),
 "spider":   ("perpos_spider/spider", "spider"),
}

def piece_offsets(ids):
    """exact char offsets by per-token decode (byte-BPE concatenation == full decode)."""
    pieces = [tok.decode([t], skip_special_tokens=False) for t in ids]
    offs, c = [], 0
    for p in pieces:
        offs.append((c, c+len(p))); c += len(p)
    return offs, "".join(pieces)

def tag(full, offs, kind):
    n = len(offs); cats = ["text"]*n
    # ---- think region (bfcl/specbench): up to and incl </think>
    et = full.find("</think>")
    body0 = 0
    if et != -1:
        end = et + len("</think>")
        for i in range(n):
            if offs[i][1] <= end: cats[i] = "think"
        body0 = end
    # ---- action region
    spans = []
    if kind == "bfcl":
        m = re.search(r"\[\s*[A-Za-z_]\w*\s*\(", full[body0:])
        if m:
            a0 = body0 + m.start(); a1 = full.rfind("]")
            a1 = (a1+1) if a1 >= a0 else len(full)
            spans.append((a0, a1))
    elif kind == "swebench":
        for m in re.finditer(r"```mswea_bash_command.*?```", full, re.DOTALL):
            spans.append((m.start(), m.end()))
    elif kind == "spider":
        m = re.search(r"Action:\s*Bash\(", full)
        if m:
            a1 = full.find("<|im_end|>", m.start()); a1 = a1 if a1 != -1 else len(full)
            spans.append((m.start(), a1))
    # specbench: no action
    for (a0, a1) in spans:
        for i in range(n):
            s, e = offs[i]
            if s >= a0 and e <= a1 and cats[i] != "think":
                cats[i] = "action"
    return cats

def accept_len(match):
    c = 0
    for m in match:
        if m == 1: c += 1
        else: break
    return c

for wl,(stem,kind) in WL.items():
    tr = json.load(open(B+"results/"+stem+".traces.json"))
    cats_by_rid = {}
    for t in tr["eval_traces"]:
        offs, full = piece_offsets(t["output_ids"])
        cats_by_rid[t["rid"]] = tag(full, offs, kind)
    # aggregate per segment
    agg = {}   # cat -> [n, sum_dflash, sum_suffix_warm, sum_suffix_cold]
    miss = 0
    for line in open(B+"results/"+stem+".jsonl"):
        line=line.strip()
        if not line: continue
        r = json.loads(line)
        rid = r["rid"]; gi = r["pos"] - 1
        cats = cats_by_rid.get(rid)
        if cats is None or not (0 <= gi < len(cats)):
            miss += 1; continue
        cat = cats[gi]
        a = agg.setdefault(cat, [0,0,0,0])
        a[0]+=1
        a[1]+=accept_len(r["dflash_match"])
        a[2]+=r.get("suffix_match_warm",0)
        a[3]+=r.get("suffix_match_cold",0)
    tot = sum(a[0] for a in agg.values())
    print(f"\n===== {wl}  (kind={kind}, {len(cats_by_rid)} traces, {tot} positions, miss={miss}) =====")
    print(f"  {'segment':<10} {'n':>7} {'%pos':>6}   {'DFlash':>7} {'Suf(w)':>7} {'Suf(c)':>7}   (mean accepted draft len)")
    order = ["think","action","text"]
    for cat in order + [c for c in agg if c not in order]:
        if cat not in agg: continue
        n,sd,sw,sc = agg[cat]
        print(f"  {cat:<10} {n:>7} {100*n/tot:>5.1f}%   {sd/n:>7.3f} {sw/n:>7.3f} {sc/n:>7.3f}")
    n = tot; sd=sum(a[1] for a in agg.values()); sw=sum(a[2] for a in agg.values()); sc=sum(a[3] for a in agg.values())
    print(f"  {'ALL':<10} {n:>7} {100.0:>5.1f}%   {sd/n:>7.3f} {sw/n:>7.3f} {sc/n:>7.3f}")
