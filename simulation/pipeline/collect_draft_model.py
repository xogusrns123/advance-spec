"""Collect per-step draft-model proposals (Stage 3b).

For each decoding step in agent_results_eagle3.json, call a small draft LM
(e.g. Qwen/Qwen3-0.6B) on the per-step context and record its flat
autoregressive draft chain. Stage 4 (collect_union_trie) consumes this file
to merge draft-model proposals into the union trie.

Two backends are supported:
  --server-url <url>   → SGLang server (preferred, uses prefix caching)
  (default)            → HuggingFace model loaded in-process

Output schema (one JSONL record per step):
    {"request_id": bfcl_id, "call_idx": int, "step_idx": int,
     "token_ids": [...], "parents": [-1, 0, 1, ...]}

Usage:
    # SGLang backend
    python3 -m simulation.pipeline.collect_draft_model \\
        --agent-results results/.../agent_results_eagle3.json \\
        --output simulation/results/.../draft_model_drafts.jsonl \\
        --model Qwen/Qwen3-0.6B \\
        --server-url http://localhost:31000 \\
        --max-draft-tokens 16

    # Sharded parallel (see run_parallel_draft_model.sh)
    ... --shard 0/4
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from simulation.evaluation.run_oracle_sim import (
    extract_requests,
    load_exclude_ids,
)


def _iter_steps(req: dict) -> Iterator[Tuple[int, int, List[int]]]:
    """Yield (call_idx, step_idx, context_token_ids) for every step that
    needs a draft. Context matches the definition used by Stage 3a/Stage 4:
    ``prompt + tokens[0:step_idx]``.

    Steps where the remaining future has ≤1 token are skipped (nothing to
    verify) — same guard used in collect_suffix_drafts.
    """
    prompt_ids_list = req.get("per_call_prompt_ids")
    for call_idx in range(len(req["per_call_tokens"])):
        tokens = req["per_call_tokens"][call_idx]
        n = len(tokens)
        if n == 0:
            continue
        prompt = (list(prompt_ids_list[call_idx])
                  if prompt_ids_list and call_idx < len(prompt_ids_list)
                  else [])
        decoded: List[int] = []
        for pos in range(n):
            if n - pos <= 1:
                decoded.append(tokens[pos])
                continue
            context = prompt + decoded if prompt else list(decoded)
            yield call_idx, pos, context
            decoded.append(tokens[pos])


def _flat_chain(draft_tokens: List[int]) -> Tuple[List[int], List[int]]:
    n = len(draft_tokens)
    parents = [-1] + list(range(n - 1))
    return list(draft_tokens), parents[:n]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _generate_sglang(
    requests: list[dict],
    server_url: str,
    max_tokens: int,
    out_fp,
) -> int:
    import requests as http_requests

    total_steps = 0
    for req in requests:
        total_steps += sum(
            max(0, len(call_tokens) - 1)
            for call_tokens in req["per_call_tokens"]
        )

    written = 0
    t0 = time.time()
    for req in requests:
        rid = req["bfcl_id"]
        for call_idx, step_idx, context in _iter_steps(req):
            try:
                resp = http_requests.post(
                    f"{server_url}/generate",
                    json={
                        "input_ids": context,
                        "sampling_params": {
                            "max_new_tokens": max_tokens,
                            "temperature": 0,
                        },
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                draft_tokens = resp.json().get("output_ids", []) or []
            except Exception as e:
                print(f"WARN: draft LM request failed "
                      f"({rid}, call {call_idx}, step {step_idx}): {e}",
                      file=sys.stderr)
                draft_tokens = []

            tids, pids = _flat_chain(draft_tokens)
            out_fp.write(json.dumps({
                "request_id": rid,
                "call_idx": call_idx,
                "step_idx": step_idx,
                "token_ids": tids,
                "parents": pids,
            }) + "\n")
            written += 1
            if written % 200 == 0:
                out_fp.flush()
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                eta = (total_steps - written) / rate if rate > 0 else 0
                print(f"  [{written}/{total_steps}] {rate:.1f} steps/s, "
                      f"ETA {eta:.0f}s", file=sys.stderr)
    out_fp.flush()
    return written


def _generate_hf(
    requests: list[dict],
    model_name: str,
    device: str,
    max_tokens: int,
    out_fp,
    checkpoint_every: int,
) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading draft model: {model_name} on {device}", file=sys.stderr)
    _ = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    model.eval()

    written = 0
    t0 = time.time()
    try:
        for ri, req in enumerate(requests):
            rid = req["bfcl_id"]
            for call_idx, step_idx, context in _iter_steps(req):
                if not context:
                    tids, pids = [], []
                else:
                    input_ids = torch.tensor(
                        [context], dtype=torch.long, device=device)
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=max_tokens,
                            do_sample=False,
                            use_cache=True,
                        )
                    draft_tokens = outputs[0][len(context):].tolist()
                    tids, pids = _flat_chain(draft_tokens)

                out_fp.write(json.dumps({
                    "request_id": rid,
                    "call_idx": call_idx,
                    "step_idx": step_idx,
                    "token_ids": tids,
                    "parents": pids,
                }) + "\n")
                written += 1

            if checkpoint_every > 0 and (
                    (ri + 1) % checkpoint_every == 0 or ri == len(requests) - 1):
                out_fp.flush()
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                print(f"  [{ri+1}/{len(requests)}] {written} drafts, "
                      f"{rate:.1f} steps/s", file=sys.stderr)
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return written


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------

def _shard_requests(requests: list[dict], shard_id: int,
                    num_shards: int) -> list[dict]:
    """Greedy bin-packing by step count for balanced load across shards."""
    step_counts = [
        (i, sum(max(0, len(t) - 1) for t in req["per_call_tokens"]))
        for i, req in enumerate(requests)
    ]
    step_counts.sort(key=lambda x: -x[1])
    loads = [0] * num_shards
    assignment = {}
    for idx, count in step_counts:
        target = min(range(num_shards), key=lambda s: loads[s])
        assignment[idx] = target
        loads[target] += count
    mine = [requests[i] for i in range(len(requests))
            if assignment[i] == shard_id]
    print(f"Shard {shard_id}/{num_shards}: "
          f"{len(mine)} requests (load balance: {loads})", file=sys.stderr)
    return mine


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-results", required=True,
                        help="Path to agent_results_eagle3.json")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--model", required=True,
                        help="Draft LM name (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--max-draft-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--server-url", default=None,
                        help="SGLang server URL; if set, uses SGLang backend")
    parser.add_argument("--shard", default=None, help="e.g. '0/4'")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="HF backend: flush + GC every N requests")
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--target-model", default=None,
                        help="Target model name for BFCL prompt tokenizer")
    parser.add_argument("--dataset", default=None,
                        help="BFCL/SpecBench dataset.jsonl")
    parser.add_argument("--responses", default=None,
                        help="BFCL agent_results_responses.json")
    args = parser.parse_args()

    exclude_ids = load_exclude_ids(args.exclude) if args.exclude else set()

    print(f"Loading: {args.agent_results}", file=sys.stderr)
    with open(args.agent_results) as f:
        data = json.load(f)

    tokenizer = None
    if args.target_model:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.target_model}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    bfcl_dataset: Optional[dict] = None
    resp_by_id: Optional[dict] = None
    specbench_dataset: Optional[dict] = None
    if args.dataset and args.responses:
        try:
            PROJECT_ROOT = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(PROJECT_ROOT))
            from simulation.agents.bfcl_agent import preprocess_bfcl_requests
            entries = []
            with open(args.dataset) as f:
                for line in f:
                    entries.append(json.loads(line))
            preprocess_bfcl_requests(entries)
            bfcl_dataset = {e["bfcl_id"]: e for e in entries}
            with open(args.responses) as f:
                resp_data = json.load(f)
            resp_by_id = {r["bfcl_id"]: r for r in resp_data}
        except Exception as e:
            print(f"WARN: BFCL prompt reconstruction failed: {e}",
                  file=sys.stderr)
    elif args.dataset:
        try:
            specbench_dataset = {}
            with open(args.dataset) as f:
                for line in f:
                    entry = json.loads(line)
                    specbench_dataset[entry["question_id"]] = entry
        except Exception as e:
            print(f"WARN: SpecBench dataset load failed: {e}",
                  file=sys.stderr)

    requests_ = extract_requests(
        data, exclude_ids, None,
        tokenizer, bfcl_dataset, resp_by_id, specbench_dataset)
    print(f"Requests: {len(requests_)}", file=sys.stderr)

    if args.shard:
        shard_id, num_shards = map(int, args.shard.split("/"))
        requests_ = _shard_requests(requests_, shard_id, num_shards)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out_fp:
        if args.server_url:
            written = _generate_sglang(
                requests_, args.server_url, args.max_draft_tokens, out_fp)
        else:
            written = _generate_hf(
                requests_, args.model, args.device, args.max_draft_tokens,
                out_fp, args.checkpoint_every)

    print(f"Output: {args.output} ({written} drafts)", file=sys.stderr)


if __name__ == "__main__":
    main()
