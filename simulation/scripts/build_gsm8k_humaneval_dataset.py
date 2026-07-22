#!/usr/bin/env python3
"""Build a combined GSM8K (math) + HumanEval (code) single-turn dataset in the SpecBench format
(question_id / category / subtask / turns) so it runs through the proven specbench_agent +
record->replay(oracle) chain-hybrid pipeline (no agentic loop, suffix-win IS computable — unlike
the SWE-bench mini-swe-agent replay).

  gsm8k     : math reasoning (HF openai/gsm8k? -> 'gsm8k' config 'main', split test)
  humaneval : code completion (HF 'openai_humaneval', split test, 164 problems)

Interleaved round-robin by subtask (like SpecBench) so any early-stopped prefix stays balanced.
Output: data/gsm8k_humaneval/dataset_interleaved.jsonl

Run IN docker (HF/datasets + network): docker exec sglang-bench python3 \
  /workspace/simulation/scripts/build_gsm8k_humaneval_dataset.py --n-per 164
"""
import argparse, json, os


def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    rows = []
    for i, r in enumerate(ds):
        if i >= n: break
        q = r["question"].strip()
        rows.append({"subtask": "gsm8k", "category": "gsm8k",
                     "turns": [f"{q}\n\nSolve this step by step and give the final numeric answer."]})
    return rows


def load_humaneval(n):
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    rows = []
    for i, r in enumerate(ds):
        if i >= n: break
        prompt = r["prompt"]
        rows.append({"subtask": "humaneval", "category": "humaneval", "he_task_id": r["task_id"],
                     "turns": ["Complete the following Python function. Return the full function "
                               "implementation in a code block.\n\n```python\n" + prompt + "\n```"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=164, help="max problems per subtask")
    ap.add_argument("--out-dir", default="data/gsm8k_humaneval")
    args = ap.parse_args()
    g = load_gsm8k(args.n_per)
    h = load_humaneval(args.n_per)
    print(f"gsm8k={len(g)} humaneval={len(h)}")
    # round-robin interleave by subtask
    inter = []
    for i in range(max(len(g), len(h))):
        if i < len(g): inter.append(g[i])
        if i < len(h): inter.append(h[i])
    for k, r in enumerate(inter):
        r["question_id"] = str(k)
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, "dataset_interleaved.jsonl")
    with open(p, "w") as f:
        for r in inter:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(inter)} rows -> {p}")
    # also non-interleaved copy for reference
    with open(os.path.join(args.out_dir, "dataset.jsonl"), "w") as f:
        for r in (g + h):
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
