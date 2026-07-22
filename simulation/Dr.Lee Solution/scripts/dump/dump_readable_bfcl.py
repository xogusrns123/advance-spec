#!/usr/bin/env python3
"""Decode the BFCL v4 capture into readable markdown (prompt head+tail + full
target output). BFCL prompts are long (tool schemas + web results + dialogue),
so the prompt is truncated to head 600 + tail 900 chars. CPU only.
  python3 scripts/dump_readable_bfcl.py
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

BASE = Path("/workspace/simulation/Dr.Lee Solution")
OUT = BASE / "readable_outputs" / "bfcl"
OUT.mkdir(parents=True, exist_ok=True)


def show_prompt(tok, ids):
    t = tok.decode(ids, skip_special_tokens=True)
    if len(t) <= 1700:
        return t
    return (t[:600] + f"\n\n…[중략 — {len(t) - 1500} chars / tool schema + web results]…\n\n"
            + t[-900:])


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    tr = json.load(open(BASE / "results/perpos_bfcl/bfcl_v4.traces.json"))
    ev = tr["eval_traces"]
    md = ["# BFCL v4 — 실제 궤적 (prompt + target 생성)\n"]
    md.append("- ⚠️ 이 캡처는 **thinking-ON** (40/40 출력이 `<think>…</think>` 추론 포함). "
              "multislot/SpecBench(제가 thinking-off로 생성)와 이 축이 다릅니다.")
    md.append(f"- {len(ev)} eval samples · agentic web_search · 27B pinned trajectory "
              "(chain_hybrid gt_tokens)")
    md.append("- prompt은 3k~10k 토큰이라 **앞 600자 + 뒤 900자만** 표시 (중간 tool schema/web 결과 생략)\n")
    for i, e in enumerate(ev):
        p = show_prompt(tok, e["prompt_ids"]).replace("```", "'''")
        o = tok.decode(e["output_ids"], skip_special_tokens=True).strip().replace("```", "'''")
        md.append(f"## bfcl_v4 #{i}  (prompt {len(e['prompt_ids'])} tok → output {len(e['output_ids'])} tok)\n")
        md.append("**Prompt (head+tail):**\n```\n" + p + "\n```\n")
        md.append("**Output (target-greedy):**\n```\n" + o + "\n```\n")
    p = OUT / "bfcl_v4.md"
    p.write_text("\n".join(md))
    print(f"wrote {p}  ({len(ev)} samples)")


if __name__ == "__main__":
    main()
