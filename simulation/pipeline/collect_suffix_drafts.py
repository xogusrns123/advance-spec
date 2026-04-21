"""Collect per-step suffix-decoding drafts (Stage 3a).

For each decoding step in agent_results_eagle3.json, run
arctic_inference.SuffixDecodingCache.speculate on the per-step context and
emit a JSONL record. Stage 4 (collect_union_trie) consumes this file to
merge suffix proposals into the union trie.

The cache accumulates state as we iterate sequentially through requests —
every completed step's token feeds back into the cache so later steps
benefit from observed patterns. This matches the in-place behaviour the
logic used to have inside collect_union_trie.py.

Output schema (one JSONL record per step where suffix produced a draft):
    {"request_id": bfcl_id, "call_idx": int, "step_idx": int,
     "token_ids": [...], "parents": [...], "score": float}

Usage:
    python3 -m simulation.pipeline.collect_suffix_drafts \\
        --agent-results results/.../agent_results_eagle3.json \\
        --output simulation/results/.../suffix_drafts.jsonl \\
        --model zai-org/GLM-4.7-Flash
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from simulation.evaluation.run_oracle_sim import (
    extract_requests,
    load_exclude_ids,
)


def _warmup_cache(cache, train_requests: list[dict]) -> None:
    """Feed completed trajectories from train split into the cache."""
    warmup_id = 0
    for req in train_requests:
        for call_idx in range(len(req["per_call_tokens"])):
            tokens = req["per_call_tokens"][call_idx]
            if not tokens:
                continue
            prompt_ids_list = req.get("per_call_prompt_ids")
            if prompt_ids_list and call_idx < len(prompt_ids_list):
                prompt = np.array(prompt_ids_list[call_idx], dtype=np.int32)
            else:
                prompt = np.array([], dtype=np.int32)
            cache.start_request(warmup_id, prompt)
            cache.add_active_response(warmup_id, tokens)
            cache.stop_request(warmup_id)
            warmup_id += 1


def collect_suffix_drafts(
    requests: list[dict],
    suffix_cache,
    max_spec_tokens: int = 256,
    max_spec_factor: float = 4.0,
    min_token_prob: float = 0.0,
) -> list[dict]:
    """Generate per-step suffix drafts, returning JSONL-ready records."""
    records = []
    req_id = 0

    for ri, req in enumerate(requests):
        bfcl_id = req["bfcl_id"]
        prompt_ids_list = req.get("per_call_prompt_ids")

        for call_idx in range(len(req["per_call_tokens"])):
            tokens = req["per_call_tokens"][call_idx]
            N = len(tokens)
            if N == 0:
                continue

            if prompt_ids_list and call_idx < len(prompt_ids_list):
                prompt = np.array(prompt_ids_list[call_idx], dtype=np.int32)
            else:
                prompt = np.array([], dtype=np.int32)
            suffix_cache.start_request(req_id, prompt)
            decoded: list[int] = []

            for pos in range(N):
                future = tokens[pos:]
                if len(future) <= 1:
                    decoded.append(tokens[pos])
                    suffix_cache.add_active_response(req_id, [tokens[pos]])
                    continue

                if len(prompt) > 0:
                    suffix_context = (
                        np.concatenate(
                            [prompt, np.array(decoded, dtype=np.int32)])
                        if decoded else prompt.copy())
                else:
                    suffix_context = np.array(decoded, dtype=np.int32)

                draft = suffix_cache.speculate(
                    req_id, suffix_context,
                    max_spec_tokens=max_spec_tokens,
                    max_spec_factor=max_spec_factor,
                    min_token_prob=min_token_prob,
                    use_tree_spec=True,
                )

                if draft.token_ids:
                    records.append({
                        "request_id": bfcl_id,
                        "call_idx": call_idx,
                        "step_idx": pos,
                        "token_ids": list(draft.token_ids),
                        "parents": list(draft.parents),
                        "score": float(getattr(draft, "score", 0.0)),
                    })

                decoded.append(tokens[pos])
                suffix_cache.add_active_response(req_id, [tokens[pos]])

            suffix_cache.stop_request(req_id)
            req_id += 1

        if (ri + 1) % 10 == 0:
            print(f"  Processed {ri + 1}/{len(requests)} requests, "
                  f"{len(records)} drafts", file=sys.stderr)

    return records


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-results", required=True,
                        help="Path to agent_results_eagle3.json")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path for suffix drafts")
    parser.add_argument("--exclude", default=None,
                        help="Path to file with one bfcl_id per line to skip")
    parser.add_argument("--model", default=None,
                        help="Target model name for tokenizer (BFCL prompt reconstruction)")
    parser.add_argument("--dataset", default=None,
                        help="Path to dataset.jsonl (BFCL or SpecBench)")
    parser.add_argument("--responses", default=None,
                        help="Path to agent_results_responses.json (BFCL only)")
    parser.add_argument("--train-ratio", type=float, default=0.0,
                        help="Fraction of requests per category used to warm "
                             "the suffix cache (default: 0)")
    parser.add_argument("--max-spec-tokens", type=int, default=256)
    parser.add_argument("--max-spec-factor", type=float, default=4.0)
    parser.add_argument("--min-token-prob", type=float, default=0.0)
    args = parser.parse_args()

    from arctic_inference.suffix_decoding import SuffixDecodingCache

    exclude_ids = load_exclude_ids(args.exclude) if args.exclude else set()

    print(f"Loading: {args.agent_results}", file=sys.stderr)
    with open(args.agent_results) as f:
        data = json.load(f)

    tokenizer = None
    if args.model:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.model}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

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

    all_requests = extract_requests(
        data, exclude_ids, None,
        tokenizer, bfcl_dataset, resp_by_id, specbench_dataset)
    print(f"Requests: {len(all_requests)}", file=sys.stderr)

    if args.train_ratio > 0:
        by_cat: dict = defaultdict(list)
        for req in all_requests:
            by_cat[req["category"]].append(req)
        train_requests: list[dict] = []
        test_requests: list[dict] = []
        for cat in sorted(by_cat):
            reqs = by_cat[cat]
            n_train = int(len(reqs) * args.train_ratio)
            train_requests.extend(reqs[:n_train])
            test_requests.extend(reqs[n_train:])
    else:
        train_requests = []
        test_requests = all_requests

    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
    if train_requests:
        print(f"Warming suffix cache with {len(train_requests)} requests...",
              file=sys.stderr)
        _warmup_cache(cache, train_requests)

    t0 = time.time()
    print(f"Speculating for {len(test_requests)} requests...", file=sys.stderr)
    records = collect_suffix_drafts(
        test_requests, cache,
        max_spec_tokens=args.max_spec_tokens,
        max_spec_factor=args.max_spec_factor,
        min_token_prob=args.min_token_prob,
    )
    elapsed = time.time() - t0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    total_nodes = sum(len(r["token_ids"]) for r in records)
    print(f"\nDone in {elapsed:.1f}s", file=sys.stderr)
    print(f"Drafts: {len(records)} "
          f"({total_nodes:,} total nodes, "
          f"avg {total_nodes / max(len(records), 1):.1f}/draft)",
          file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
