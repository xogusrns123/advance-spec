"""Collect draft model proposals and rebuild union trie.

Runs a small draft model (e.g. Qwen3-0.6B) on each step's context
and collects N autoregressive draft tokens as a flat chain proposer.
Rebuilds the union trie to include draft_model nodes so that
downstream p_t collection covers all proposers.

Can run on multiple GPUs in parallel (one model per GPU).

Usage:
    python3 scripts/collect_draft_model.py \
        --union-trie-data results/.../union_trie_data.jsonl \
        --output results/.../union_trie_data_with_dm.jsonl \
        --model Qwen/Qwen3-0.6B \
        --max-draft-tokens 16
"""

from __future__ import annotations

import argparse
import json
import gc
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List

from simulation.pipeline.collect_union_trie import build_union_trie


def _rebuild_union_trie(rec: dict) -> None:
    """Rebuild union_trie and source_map to include draft_model nodes."""
    per_proposer = rec.get("per_proposer", {})
    proposer_trees = {}
    for name, data in per_proposer.items():
        tids = data.get("token_ids", [])
        pids = data.get("parents", [])
        if tids:
            proposer_trees[name] = (tids, pids)

    if not proposer_trees:
        return

    flat_tokens, flat_parents, source_map = build_union_trie(proposer_trees)
    rec["union_trie"] = {
        "token_ids": flat_tokens,
        "parents": flat_parents,
    }
    rec["source_map"] = source_map


def collect_draft_chains_hf(
    records: List[dict],
    model_name: str,
    device: str = "cuda",
    max_tokens: int = 16,
) -> None:
    """Generate draft chains using HuggingFace model. Modifies records in-place."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading draft model: {model_name} on {device}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    model.eval()

    n = len(records)
    t0 = time.time()

    # Group by request for KV cache reuse
    prev_key = None
    past_kv = None
    cached_len = 0

    for i, rec in enumerate(records):
        context_ids = rec.get("context_token_ids", [])
        if not context_ids:
            continue

        curr_key = (rec.get("request_id"), rec.get("call_idx"))

        # Generate draft tokens autoregressively
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
            )

        # Extract generated tokens (excluding prompt)
        draft_tokens = outputs[0][len(context_ids):].tolist()

        # Build flat chain tree: linear parent structure
        # parents = [-1, 0, 1, 2, ...] (each node's parent is the previous)
        n_draft = len(draft_tokens)
        parents = [-1] + list(range(n_draft - 1))

        # Add to per_proposer
        if "per_proposer" not in rec:
            rec["per_proposer"] = {}
        rec["per_proposer"]["draft_model"] = {
            "token_ids": draft_tokens,
            "parents": parents[:n_draft],
            "size": n_draft,
        }
        # Rebuild union trie to include draft_model nodes
        _rebuild_union_trie(rec)

        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {rate:.1f} steps/s, ETA {eta:.0f}s",
                  file=sys.stderr)

    del model
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def collect_draft_chains_sglang(
    records: List[dict],
    server_url: str,
    max_tokens: int = 16,
) -> None:
    """Generate draft chains via SGLang server. Much faster with prefix caching."""
    import requests as http_requests

    n = len(records)
    t0 = time.time()

    for i, rec in enumerate(records):
        context_ids = rec.get("context_token_ids", [])
        if not context_ids:
            continue

        try:
            resp = http_requests.post(
                f"{server_url}/generate",
                json={
                    "input_ids": context_ids,
                    "sampling_params": {
                        "max_new_tokens": max_tokens,
                        "temperature": 0,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            # output_ids contains only generated tokens (no prompt)
            draft_tokens = data.get("output_ids", [])
        except Exception as e:
            draft_tokens = []

        n_draft = len(draft_tokens)
        parents = [-1] + list(range(n_draft - 1))

        if "per_proposer" not in rec:
            rec["per_proposer"] = {}
        rec["per_proposer"]["draft_model"] = {
            "token_ids": draft_tokens,
            "parents": parents[:n_draft],
            "size": n_draft,
        }
        _rebuild_union_trie(rec)

        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {rate:.1f} steps/s, ETA {eta:.0f}s",
                  file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--union-trie-data", required=True,
                        help="Input JSONL (from collect_union_trie)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL with draft_model added to per_proposer")
    parser.add_argument("--model", required=True,
                        help="Draft model name (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--max-draft-tokens", type=int, default=16,
                        help="Max tokens to generate per step")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--server-url", default=None,
                        help="SGLang server URL (e.g. http://localhost:30010). "
                             "If set, uses SGLang instead of HuggingFace.")
    parser.add_argument("--shard", default=None,
                        help="Process shard M/N, e.g. '0/4'")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Save checkpoint every N requests")
    args = parser.parse_args()

    # Load records
    print(f"Loading: {args.union_trie_data}", file=sys.stderr)
    records = []
    with open(args.union_trie_data) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Shard: greedy bin-packing by step count for balanced distribution
    if args.shard:
        shard_id, num_shards = map(int, args.shard.split("/"))
        from collections import Counter
        req_counts = Counter(r.get("request_id", "") for r in records)
        # Sort requests largest-first, assign to smallest shard
        sorted_reqs = sorted(req_counts.items(), key=lambda x: -x[1])
        shard_loads = [0] * num_shards
        shard_assignment = {}  # req_id → shard_id
        for req_id, count in sorted_reqs:
            target = min(range(num_shards), key=lambda s: shard_loads[s])
            shard_assignment[req_id] = target
            shard_loads[target] += count
        my_req_ids = set(rid for rid, sid in shard_assignment.items()
                         if sid == shard_id)
        records = [r for r in records if r.get("request_id", "") in my_req_ids]
        print(f"Shard {shard_id}/{num_shards}: {len(records)} records "
              f"(balanced load: {shard_loads})", file=sys.stderr)

    print(f"Records: {len(records)}", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.server_url:
        # SGLang server mode (fast, uses prefix caching)
        collect_draft_chains_sglang(records, args.server_url, args.max_draft_tokens)
        with open(output_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    elif args.checkpoint_every > 0:
        # HuggingFace with per-request checkpointing
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading draft model: {args.model} on {args.device}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, trust_remote_code=True,
        ).to(args.device)
        model.eval()

        by_req = OrderedDict()
        for rec in records:
            by_req.setdefault(rec.get("request_id", ""), []).append(rec)

        t0 = time.time()
        processed = 0

        with open(output_path, "w") as f:
            for ri, (rid, req_records) in enumerate(by_req.items()):
                for rec in req_records:
                    context_ids = rec.get("context_token_ids", [])
                    if not context_ids:
                        continue

                    input_ids = torch.tensor([context_ids], dtype=torch.long,
                                             device=args.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids, max_new_tokens=args.max_draft_tokens,
                            do_sample=False, use_cache=True,
                        )

                    draft_tokens = outputs[0][len(context_ids):].tolist()
                    n_draft = len(draft_tokens)
                    parents = [-1] + list(range(n_draft - 1))

                    if "per_proposer" not in rec:
                        rec["per_proposer"] = {}
                    rec["per_proposer"]["draft_model"] = {
                        "token_ids": draft_tokens,
                        "parents": parents[:n_draft],
                        "size": n_draft,
                    }
                    _rebuild_union_trie(rec)

                for rec in req_records:
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                processed += len(req_records)

                if (ri + 1) % args.checkpoint_every == 0 or ri == len(by_req) - 1:
                    elapsed = time.time() - t0
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"  [{ri+1}/{len(by_req)}] {processed} steps, "
                          f"{rate:.1f} steps/s", file=sys.stderr)
                    gc.collect()
                    torch.cuda.empty_cache()

        del model
    else:
        collect_draft_chains_hf(records, args.model, args.device, args.max_draft_tokens)
        with open(output_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
