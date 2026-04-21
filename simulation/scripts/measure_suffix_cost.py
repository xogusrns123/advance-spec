"""Measure single-call latency of ArcticInference SuffixDecodingCache.speculate().

Per workload: warm the cache with one prompt, then time a single
``speculate(..., use_tree_spec=True)`` call on a separate prompt. CPU only,
no server, runs in under a second.

Usage:
    python3 simulation/scripts/measure_suffix_cost.py \\
        --workloads specbench,bfcl_v4 \\
        --model Qwen/Qwen3-8B \\
        --output results/latency/suffix_cost.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _workload_prompts import load_workload_prompts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads", default="specbench,bfcl_v4,swebench",
                        help="Comma-separated workload names")
    parser.add_argument("--model", required=True,
                        help="Model name for tokenizer (matches target model)")
    parser.add_argument("--max-spec-tokens", type=int, default=256)
    parser.add_argument("--max-spec-factor", type=float, default=4.0)
    parser.add_argument("--min-token-prob", type=float, default=0.0)
    parser.add_argument("--output", required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    from arctic_inference.suffix_decoding import SuffixDecodingCache
    from transformers import AutoTokenizer

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]

    print(f"Loading tokenizer: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100)

    results = []
    for w in workloads:
        prompts = load_workload_prompts(w, n_samples=2)
        if len(prompts) < 2:
            print(f"SKIP {w}: need 2 prompts, got {len(prompts)}", file=sys.stderr)
            continue
        warmup_prompt, measure_prompt = prompts[0], prompts[1]

        # Warm global suffix tree with the warmup prompt's content
        warmup_ids = np.array(
            tokenizer.encode(warmup_prompt["messages"][0]["content"]),
            dtype=np.int32)
        w_rid = f"warmup_{w}"
        cache.start_request(w_rid, warmup_ids)
        cache.add_active_response(w_rid, warmup_ids.tolist())
        cache.stop_request(w_rid)

        # Measure: single speculate() call on the second prompt
        measure_ids = np.array(
            tokenizer.encode(measure_prompt["messages"][0]["content"]),
            dtype=np.int32)
        m_rid = f"measure_{w}"
        cache.start_request(m_rid, measure_ids)
        # one warmup speculate (result discarded) to ensure first-call JIT costs
        # don't dominate the measured call
        try:
            cache.speculate(
                m_rid, measure_ids,
                max_spec_tokens=args.max_spec_tokens,
                max_spec_factor=args.max_spec_factor,
                min_token_prob=args.min_token_prob,
                use_tree_spec=True)
        except Exception as e:
            print(f"WARN {w}: warmup speculate failed: {e}", file=sys.stderr)
        # timed call
        try:
            t0 = time.perf_counter()
            draft = cache.speculate(
                m_rid, measure_ids,
                max_spec_tokens=args.max_spec_tokens,
                max_spec_factor=args.max_spec_factor,
                min_token_prob=args.min_token_prob,
                use_tree_spec=True)
            t1 = time.perf_counter()
            speculate_ms = (t1 - t0) * 1000
            draft_size = len(draft.token_ids)
        except Exception as e:
            print(f"ERROR {w}: measure speculate failed: {e}", file=sys.stderr)
            speculate_ms = None
            draft_size = 0
        cache.stop_request(m_rid)

        entry = {
            "workload": w,
            "prompt_len": int(len(measure_ids)),
            "draft_size": int(draft_size),
            "speculate_ms": (round(speculate_ms, 4)
                             if speculate_ms is not None else None),
            "id": measure_prompt["id"],
            "category": measure_prompt["category"],
        }
        results.append(entry)
        print(f"  {w:10s}  prompt_len={entry['prompt_len']:>5d}  "
              f"draft_size={entry['draft_size']:>3d}  "
              f"speculate={entry['speculate_ms']} ms",
              file=sys.stderr)

    output = {
        "params": {
            "max_spec_tokens": args.max_spec_tokens,
            "max_spec_factor": args.max_spec_factor,
            "min_token_prob": args.min_token_prob,
            "use_tree_spec": True,
        },
        "model_tokenizer": args.model,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
