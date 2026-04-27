#!/usr/bin/env python3
"""Create stratified LongBench LCC subset in specbench format."""
import json
import os

N_PER_LANG = 30
buckets = {"python": [], "java": [], "csharp": []}
with open("/tmp/data/lcc.jsonl") as f:
    for line in f:
        e = json.loads(line)
        lang = e.get("language")
        if lang in buckets and len(buckets[lang]) < N_PER_LANG:
            buckets[lang].append(e)

os.makedirs("/workspace/data/longbench_lcc", exist_ok=True)
out = "/workspace/data/longbench_lcc/dataset.jsonl"
total = 0
with open(out, "w") as f:
    for lang, items in buckets.items():
        for e in items:
            ctx = e.get("context", "")
            prompt = (
                f"Please complete the following {lang.capitalize()} code. "
                f"Output ONLY the next line of code, no explanation, no "
                f"comments, no code fences.\n\n"
                f"```{lang}\n{ctx}\n```"
            )
            entry = {
                "question_id": str(e["_id"]),
                "category": f"longbench/lcc/{lang}",
                "turns": [prompt],
            }
            f.write(json.dumps(entry) + "\n")
            total += 1
print(f"Wrote {total} LCC examples to {out}")
print("Per language:", {k: len(v) for k, v in buckets.items()})
