#!/usr/bin/env python3
"""Per-SEGMENT ownership decomposition of either4 = suffix_only + dflash_only + both,
and the unified complementary-coverage C = sqrt(suffix_only * dflash_only), to locate
WHERE in an agentic output the two proposers are complementary.

Segment taxonomy (4): think / preamble / tool_call / final  (say "tool_call", not
"action"). Aligns the arm-independent interp-study curves (s = suffix copy depth,
a = DFlash leading match; position p<->output token p) to per-token segment labels
tagged on the decoded text. Runs inside sglang-bench (needs the 27B tokenizer).

  docker exec sglang-bench bash -lc \
    "cd '/workspace/simulation/Dr.Lee Solution' && python3 scripts/seg_ownership.py"
"""
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import gzip, json, os, re
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from collections import defaultdict
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
TH = 4

# interp-study curve source (rid + length aligned; verified on host)
WL = {
    "bfcl":      ("perpos_bfcl_full/bfcl_v4_full", "bfcl"),
    "specbench": ("perpos_specbench_full/specbench", "specbench"),
    "swebench":  ("perpos_swebench_alleval/swebench_4way", "swebench"),
    "spider":    ("perpos_spider_alleval/spider_4way", "spider"),
}
SEGS = ["think", "preamble", "tool_call", "final"]


def load_curves(p):
    o = {}
    with gzip.open(p, "rt") as f:
        for l in f:
            r = json.loads(l)
            o[r["rid"]] = r
    return o


def piece_offsets(ids):
    pieces = [tok.decode([t], skip_special_tokens=False) for t in ids]
    offs, c = [], 0
    for p in pieces:
        offs.append((c, c + len(p)))
        c += len(p)
    return offs, "".join(pieces)


def tool_spans(full, body0, kind):
    spans = []
    if kind == "bfcl":
        for m in re.finditer(r"\[\s*[A-Za-z_]\w*\s*\(", full[body0:]):
            a0 = body0 + m.start()
            # close at matching-ish ']' after this call start
            a1 = full.find("]", a0)
            a1 = (a1 + 1) if a1 != -1 else len(full)
            spans.append((a0, a1))
    elif kind == "swebench":
        for m in re.finditer(r"```mswea_bash_command.*?```", full, re.DOTALL):
            spans.append((m.start(), m.end()))
    elif kind == "spider":
        for m in re.finditer(r"Action:\s*Bash\(", full):
            a1 = full.find("<|im_end|>", m.start())
            a1 = a1 if a1 != -1 else len(full)
            spans.append((m.start(), a1))
    # specbench: none
    return spans


def tag(full, offs, kind):
    n = len(offs)
    cats = ["preamble"] * n
    et = full.find("</think>")
    body0 = 0
    if et != -1:
        end = et + len("</think>")
        for i in range(n):
            if offs[i][1] <= end:
                cats[i] = "think"
        body0 = end
    spans = tool_spans(full, body0, kind)
    first_tool = min((s for s, _ in spans), default=None)
    for i in range(n):
        if cats[i] == "think":
            continue
        s, e = offs[i]
        in_tool = any(a0 <= s and e <= a1 for a0, a1 in spans)
        if in_tool:
            cats[i] = "tool_call"
        elif first_tool is None:
            cats[i] = "final"          # no tool at all -> body is the answer
        elif s >= first_tool:
            cats[i] = "final"
        else:
            cats[i] = "preamble"
    return cats


def main():
    B = os.environ.get("DRLEE", "/workspace/simulation/Dr.Lee Solution")
    idir = f"{B}/results/interp_validation"
    out_rows = []
    for wl, (stem, kind) in WL.items():
        cur = load_curves(f"{idir}/curves_{wl}.jsonl.gz")
        tr = json.load(open(f"{B}/results/{stem}.traces.json"))
        ev = {t["rid"]: t for t in tr["eval_traces"]}
        agg = {sg: defaultdict(float) for sg in SEGS}
        for rid, cu in cur.items():
            t = ev.get(rid)
            if not t:
                continue
            oid = t["output_ids"]
            offs, full = piece_offsets(oid)
            cats = tag(full, offs, kind)
            s, a = cu["s"], cu["a"]
            # per-position argmax(a,s) winner label (tie inherits prev decisive)
            lab, prev = [], 0
            for ai, si in zip(a, s):
                w = 1 if ai > si else (-1 if si > ai else prev)
                lab.append(w)
                if w:
                    prev = w
            for i in range(len(s)):
                if a[i] < 0:
                    continue
                seg = cats[i + 1] if i + 1 < len(cats) else cats[-1]
                d = agg[seg]
                S, A = s[i] >= TH, a[i] >= TH
                d["L"] += 1
                if S or A: d["ei"] += 1
                if S and not A: d["so"] += 1
                if A and not S: d["do"] += 1
                if S and A: d["bo"] += 1
                # boundary switch: winner flips between i and i+1 (attribute to seg of i)
                if i + 1 < len(s) and lab[i] and lab[i + 1] and lab[i + 1] != lab[i]:
                    d["flip"] += 1
        tot = sum(agg[sg]["L"] for sg in SEGS) or 1.0
        for sg in SEGS:
            d = agg[sg]
            L = d["L"] or 1.0
            so, do = d["so"] / L, d["do"] / L
            out_rows.append(dict(
                wl=wl, seg=sg, share=d["L"] / tot, ntok=int(d["L"]),
                either4=d["ei"] / L, suffix_only=so, dflash_only=do,
                both=d["bo"] / L, C=(so * do) ** 0.5,
                switch100=100.0 * d["flip"] / L))

    # print table
    hdr = (f"{'workload':<10}{'segment':<10}{'tok%':>7}{'either4':>9}"
           f"{'suf_only':>10}{'dfl_only':>10}{'both':>7}{'C=sqrt':>9}{'switch/100':>11}")
    print(hdr); print("-" * len(hdr))
    for wl in WL:
        for r in [x for x in out_rows if x["wl"] == wl]:
            print(f"{r['wl']:<10}{r['seg']:<10}{100*r['share']:>6.1f}%"
                  f"{r['either4']:>9.3f}{r['suffix_only']:>10.3f}"
                  f"{r['dflash_only']:>10.3f}{r['both']:>7.3f}{r['C']:>9.3f}"
                  f"{r['switch100']:>11.2f}")
        print()
    outp = f"{idir}/seg_ownership.json"
    try:
        json.dump(out_rows, open(outp, "w"), indent=1)
        print("saved ->", outp)
    except Exception as e:
        print("(no save:", e, ")")


if __name__ == "__main__":
    main()
