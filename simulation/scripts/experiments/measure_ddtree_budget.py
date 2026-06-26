"""DDTree node-budget sweep on real serving (author's algorithm, our workload).

Reproduces the DDTree figure (Acc. Length tau & ms/tok vs Node Budget) using the
official DDTree implementation (vendor/ddtree, arXiv 2604.12989) on our target +
the z-lab DFlash drafter, over the same BFCLv4 web_search workload as the other
budget-tradeoff figures.

Unlike raw DFlash (a single linear block, acceptance DECREASES past the trained
block), DDTree builds a draft TREE from the drafter's per-position distributions
via a best-first heap under a node budget — nested in budget, so acceptance is
non-decreasing.

Emits the same JSON schema as measure_budget_sweep.py so plot_budget_tradeoff.py
works unchanged:
    simulation/results/budget_tradeoff/<model_slug>/DDTree.json

Run INSIDE sglang-bench on Blackwell GPU0 (sdpa draft — no flash-attn / sm_120
kernels needed; also avoids the transformers-5.6 flash s_aux=None bug):
    docker exec -u root sglang-bench bash -lc 'cd /workspace && \
      CUDA_VISIBLE_DEVICES=0 python3 \
      simulation/scripts/experiments/measure_ddtree_budget.py \
      --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_VENDOR = "/workspace/vendor/ddtree"
sys.path.insert(0, _VENDOR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from model import DFlashDraftModel  # noqa: E402  (vendor/ddtree/model)
from ddtree import ddtree_generate, maybe_enable_cpp_compact  # noqa: E402

REPO = Path("/workspace")
DATASET = REPO / "data/bfcl_agent/dataset_stratified_interleaved.jsonl"


def slug(model: str) -> str:
    return model.split("/")[-1].lower().replace(".", "").replace("-", "_")


def load_web_search_prompts(tok, n_tasks: int) -> list[str]:
    """First n_tasks BFCLv4 web_search tasks -> chat-templated prompt strings."""
    prompts: list[str] = []
    with open(DATASET) as f:
        for line in f:
            e = json.loads(line)
            if "web_search" not in (e.get("category") or ""):
                continue
            msgs = e["question"][0]  # list of {role, content} for turn 1
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            prompts.append(text)
            if len(prompts) >= n_tasks:
                break
    if not prompts:
        raise SystemExit("no web_search tasks found in dataset")
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--budgets", default="16,32,64,128,256,512,1024")
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default="simulation/results/budget_tradeoff")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    dev = "cuda"
    maybe_enable_cpp_compact(True)  # builds an inline C++ KV-compactor; falls back to Python

    print(f"loading target {args.target} (sdpa, bf16) ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    print(f"loading draft {args.draft} (sdpa, bf16) ...", flush=True)
    draft = DFlashDraftModel.from_pretrained(
        args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.target)
    block_size = draft.block_size
    print(f"block_size={block_size} mask_token_id={draft.mask_token_id}", flush=True)

    prompts = load_web_search_prompts(tok, args.n_tasks)
    print(f"{len(prompts)} web_search prompts", flush=True)

    # Warmup (JIT/caches), result discarded.
    _ = ddtree_generate(
        model=draft, target=target,
        input_ids=tok.encode(prompts[0], return_tensors="pt").to(dev),
        mask_token_id=draft.mask_token_id, max_new_tokens=16,
        block_size=block_size, tree_budget=64,
        stop_token_ids=[tok.eos_token_id], temperature=0.0)

    out_dir = (REPO / args.out) / slug(args.target)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "DDTree.json"
    doc = {"model": args.target, "method": "DDTree", "cell": f"{slug(args.target)}_ddtree",
           "draft": args.draft, "block_size": block_size, "n_tasks": len(prompts),
           "budgets": budgets, "results": []}

    results = []
    for B in budgets:
        all_acc: list[int] = []
        tot_time = 0.0
        tot_tok = 0
        for text in prompts:
            ids = tok.encode(text, return_tensors="pt").to(dev)
            out = ddtree_generate(
                model=draft, target=target, input_ids=ids,
                mask_token_id=draft.mask_token_id, max_new_tokens=args.max_new_tokens,
                block_size=block_size, tree_budget=B,
                stop_token_ids=[tok.eos_token_id], temperature=0.0)
            all_acc.extend(int(a) for a in out.acceptance_lengths)
            tot_time += float(out.time_per_output_token) * int(out.num_output_tokens)
            tot_tok += int(out.num_output_tokens)
        committed = float(np.mean(all_acc)) if all_acc else 0.0  # incl. root (paper convention)
        mat = committed - 1.0                                    # accepted draft tokens (our tau, like other figures)
        per_tok = (tot_time / max(tot_tok, 1)) * 1000.0
        row = {"budget": B, "accept_length_mean": round(mat, 4),
               "committed_per_round": round(committed, 4),
               "per_token_ms": round(per_tok, 3),
               "n_rounds": len(all_acc), "n_output_tokens": tot_tok}
        results.append(row)
        doc["results"] = results
        out_path.write_text(json.dumps(doc, indent=2))
        print(f"  B={B}: tau(mat)={mat:.3f} committed={committed:.3f} "
              f"per_tok={per_tok:.3f}ms rounds={len(all_acc)}", flush=True)

    print(f"-> {out_path}\nDDTREE_SWEEP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
