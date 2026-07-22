#!/usr/bin/env python3
"""Decode the collected multislot traces into human-readable markdown, per k.
Shows the shared skeleton once, then each warm/eval sample's slot values + the
actual target-greedy generated code. CPU only (tokenizer). Run in docker:
  python3 scripts/dump_readable.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json
from pathlib import Path
from transformers import AutoTokenizer

import os
BASE = Path("/workspace/simulation/Dr.Lee Solution")
REC_DIR = os.environ.get("RECORD_DIR", "results/perpos")   # e.g. results/perpos_inf
OUT = BASE / "readable_outputs"
OUT.mkdir(exist_ok=True)
VARY = {0: [], 1: ["group_field"], 2: ["group_field", "value_field"],
        4: ["filter_val", "group_field", "value_field", "agg"],
        8: ["fname", "filter_field", "filter_val", "group_field",
            "value_field", "agg", "default", "ndigits"]}


def clean(txt: str) -> str:
    t = txt.strip()
    for fence in ("```python", "```py", "```"):
        if t.startswith(fence):
            t = t[len(fence):]
            break
    if t.endswith("```"):
        t = t[:-3]
    return t.strip("\n")


def skeleton(prompt: str, slots: dict) -> str:
    # replace each slot value with <SLOT_NAME> (longest-first to avoid partial hits)
    for key in sorted(slots, key=lambda k: -len(str(slots[k]))):
        prompt = prompt.replace(str(slots[key]), f"<{key}>")
    return prompt


def sample_block(idx, role, slots, vary, code):
    varv = {k: slots[k] for k in vary} if vary else {}
    vartxt = ", ".join(f"**{k}**=`{v}`" for k, v in varv.items()) or "(no varying slot — exact repeat)"
    fixed = {k: v for k, v in slots.items() if k not in vary}
    lines = [f"### {role} #{idx} — novel slots: {vartxt}"]
    if fixed:
        lines.append(f"<sub>fixed: {', '.join(f'{k}={v}' for k, v in fixed.items())}</sub>\n")
    lines.append("```python\n" + code + "\n```\n")
    return "\n".join(lines)


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    for k in [0, 1, 2, 4, 8]:
        rows = [json.loads(l) for l in open(BASE / f"scripts/bench_prompts_multislot_k{k}.jsonl")]
        warm = [r for r in rows if r["role"] == "warm"]
        ev = [r for r in rows if r["role"] == "eval"]
        tr = json.load(open(BASE / f"{REC_DIR}/multislot_k{k}.traces.json"))
        warm_out = tr["warm_traces"]
        eval_tr = {t["rid"]: t for t in tr["eval_traces"]}
        vary = VARY[k]

        md = [f"# multislot k={k} — 실제 생성 출력 (target-greedy)\n"]
        md.append(f"- **varying(novel) slots ({len(vary)}개)**: "
                  f"{', '.join(vary) if vary else '없음 → 모든 행 동일 (exact repeat)'}")
        md.append(f"- warm {len(warm_out)}개 / eval {len(eval_tr)}개, "
                  f"각 prompt는 아래 skeleton에 slot 값만 채운 것\n")
        md.append("## 공유 skeleton (slot을 `<이름>`으로 표기)\n")
        md.append("```\n" + skeleton(warm[0]["prompt"], warm[0]["slots"]) + "\n```\n")

        md.append("## EVAL 출력 (테스트 대상 — novel slot 값)\n")
        for rid in sorted(eval_tr):
            code = clean(tok.decode(eval_tr[rid]["output_ids"], skip_special_tokens=True))
            md.append(sample_block(rid, "eval", ev[rid]["slots"], vary, code))

        md.append("## WARM 출력 (suffix tree 워밍용)\n")
        for i, out_ids in enumerate(warm_out):
            code = clean(tok.decode(out_ids, skip_special_tokens=True))
            md.append(sample_block(i, "warm", warm[i]["slots"], vary, code))

        p = OUT / f"multislot_k{k}.md"
        p.write_text("\n".join(md))
        print(f"wrote {p}  ({len(eval_tr)} eval + {len(warm_out)} warm)")


if __name__ == "__main__":
    main()
