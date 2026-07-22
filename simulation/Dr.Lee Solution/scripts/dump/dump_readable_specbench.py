#!/usr/bin/env python3
"""Decode the SpecBench capture into human-readable markdown, per subtask.
Shows each eval sample's prompt (truncated) + the actual target-greedy output.
CPU only (tokenizer). Run in docker:
  python3 scripts/dump_readable_specbench.py
"""
from __future__ import annotations
# --- scripts-root + own-dir on sys.path (relocated into subfolder) ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
# --- end path shim ---
import json
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer

BASE = Path("/workspace/simulation/Dr.Lee Solution")
OUT = BASE / "readable_outputs" / "specbench"
OUT.mkdir(parents=True, exist_ok=True)
PROMPT_CHARS = 600     # truncate long prompts (summarization/rag carry articles)


def clean_prompt(txt: str) -> str:
    # keep the user turn; drop the trailing assistant tag noise
    t = txt.replace("\n\n", "\n").strip()
    if len(t) > PROMPT_CHARS:
        t = t[:PROMPT_CHARS] + " …[truncated]"
    return t


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    tr = json.load(open(BASE / "results/perpos/specbench.traces.json"))
    by_task = defaultdict(list)
    for e in tr["eval_traces"]:
        by_task[e["task"]].append(e)

    for task in sorted(by_task):
        samples = by_task[task]
        md = [f"# SpecBench — {task}  (실제 생성 출력, target-greedy, enable_thinking=False)\n"]
        md.append(f"- {len(samples)} eval samples · MAXTOK=96 (일부 긴 출력은 96토큰에서 잘릴 수 있음)\n")
        for i, e in enumerate(samples):
            prompt = clean_prompt(tok.decode(e["prompt_ids"], skip_special_tokens=True))
            out = tok.decode(e["output_ids"], skip_special_tokens=True).strip()
            trunc = " ⚠️(96토큰 절단)" if len(e["output_ids"]) >= 96 else ""
            md.append(f"## {task} #{i}{trunc}\n")
            md.append("**Prompt:**\n")
            md.append("```\n" + prompt + "\n```\n")
            md.append(f"**Output ({len(e['output_ids'])} tokens):**\n")
            md.append("```\n" + out + "\n```\n")
        p = OUT / f"{task}.md"
        p.write_text("\n".join(md))
        print(f"wrote {p}  ({len(samples)} samples)")


if __name__ == "__main__":
    main()
