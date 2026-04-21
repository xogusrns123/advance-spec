"""
Collect EAGLE-3 draft tokens and target model ground truth.

Sends prompts to a running SGLang server with EAGLE-3 enabled,
and logs per-step draft tokens + accepted tokens for analysis.

Usage:
    python -m simulation.analysis.collect_eagle3_drafts \
        --server-url http://localhost:30000 \
        --dataset humaneval \
        --output-dir simulation/results/eagle3_drafts
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests


@dataclass
class DecodingStep:
    """Record of a single speculative decoding step."""

    step_idx: int
    draft_tokens: list[int]       # tokens proposed by EAGLE-3
    draft_probs: list[float]      # EAGLE-3 probabilities for each draft
    accepted_tokens: list[int]    # tokens accepted by target model
    num_accepted: int             # how many draft tokens were accepted
    target_token: int             # the correct next token from target model
    draft_tree_paths: list[list[int]] = field(default_factory=list)


@dataclass
class GenerationRecord:
    """Full record of a single generation."""

    prompt_id: str
    prompt: str
    generated_text: str
    generated_tokens: list[int]
    steps: list[DecodingStep]
    total_tokens: int = 0
    total_accepted: int = 0
    wall_time_s: float = 0.0


def load_dataset(dataset_name: str, max_samples: int = 100) -> list[dict]:
    """Load evaluation dataset prompts."""
    if dataset_name == "humaneval":
        return _load_humaneval(max_samples)
    elif dataset_name == "mt_bench":
        return _load_mt_bench(max_samples)
    elif dataset_name == "docqa":
        return _load_docqa(max_samples)
    elif dataset_name == "agentic_sql":
        return _load_agentic_sql(max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_humaneval(max_samples: int) -> list[dict]:
    """Load HumanEval prompts. Requires datasets library."""
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("openai/openai_humaneval", split="test")
        prompts = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            prompts.append({
                "id": item["task_id"],
                "prompt": item["prompt"],
            })
        return prompts
    except ImportError:
        print("Warning: 'datasets' not installed. Using dummy prompts.")
        return [{"id": f"HE_{i}", "prompt": f"def solution_{i}():\n    "} for i in range(min(5, max_samples))]


def _load_mt_bench(max_samples: int) -> list[dict]:
    """Load MT-Bench prompts."""
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("HuggingFaceH4/mt_bench_prompts", split="train")
        prompts = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            prompts.append({
                "id": f"MT_{i}",
                "prompt": item["prompt"][0],  # first turn
            })
        return prompts
    except ImportError:
        print("Warning: 'datasets' not installed. Using dummy prompts.")
        return [{"id": f"MT_{i}", "prompt": f"Explain concept {i} in detail."} for i in range(min(5, max_samples))]


def _load_docqa(max_samples: int) -> list[dict]:
    """Load document QA prompts (MultiFieldQA)."""
    return [{"id": f"DOC_{i}", "prompt": f"Question {i} about the document."} for i in range(min(5, max_samples))]


def _load_agentic_sql(max_samples: int) -> list[dict]:
    """Load agentic SQL generation prompts."""
    sql_templates = [
        "Write a SQL query to find the top 10 customers by total order amount.",
        "Write a SQL query to calculate monthly revenue for the last 12 months.",
        "Write a SQL query to find products that have never been ordered.",
        "Write a SQL query to identify duplicate records in the users table.",
        "Write a SQL query to compute a 7-day moving average of daily sales.",
    ]
    return [{"id": f"SQL_{i}", "prompt": sql_templates[i % len(sql_templates)]} for i in range(min(len(sql_templates), max_samples))]


def generate_with_logging(
    server_url: str,
    prompt: str,
    prompt_id: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> GenerationRecord:
    """
    Send a request to SGLang server and collect generation details.

    Note: To get per-step draft/accept info, SGLang needs to be patched
    to expose speculative decoding internals. This function currently
    collects what's available via the standard API, and logs token-level
    output for offline analysis.
    """
    start = time.time()

    # Standard SGLang generate endpoint
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    resp = requests.post(f"{server_url}/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()

    wall_time = time.time() - start

    generated_text = data.get("text", "")
    # Extract token-level info from logprobs if available
    meta = data.get("meta_info", {})
    output_token_ids = meta.get("output_token_ids", [])
    output_logprobs = meta.get("output_token_logprobs", [])

    record = GenerationRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        generated_text=generated_text,
        generated_tokens=output_token_ids,
        steps=[],  # Populated by patched SGLang or offline reconstruction
        total_tokens=len(output_token_ids),
        wall_time_s=wall_time,
    )

    return record


def reconstruct_steps_offline(
    record: GenerationRecord,
    draft_log_path: str | None = None,
) -> GenerationRecord:
    """
    If SGLang is patched to dump per-step draft info to a file,
    load and attach it to the record.

    Expected log format (JSONL, one line per step):
    {"step": 0, "draft_tokens": [...], "draft_probs": [...],
     "accepted_tokens": [...], "target_token": 42, "tree_paths": [[...]]}
    """
    if draft_log_path is None or not os.path.exists(draft_log_path):
        return record

    steps = []
    with open(draft_log_path) as f:
        for line in f:
            d = json.loads(line)
            steps.append(DecodingStep(
                step_idx=d["step"],
                draft_tokens=d["draft_tokens"],
                draft_probs=d.get("draft_probs", []),
                accepted_tokens=d["accepted_tokens"],
                num_accepted=len(d["accepted_tokens"]),
                target_token=d["target_token"],
                draft_tree_paths=d.get("tree_paths", []),
            ))

    record.steps = steps
    record.total_accepted = sum(s.num_accepted for s in steps)
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Collect EAGLE-3 draft tokens from SGLang server"
    )
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument("--dataset", default="humaneval",
                        choices=["humaneval", "mt_bench", "docqa", "agentic_sql"])
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default="simulation/results/eagle3_drafts")
    parser.add_argument("--draft-log-dir", default=None,
                        help="Directory with per-request draft logs from patched SGLang")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_dataset(args.dataset, args.max_samples)
    print(f"Loaded {len(prompts)} prompts from {args.dataset}")

    records = []
    for i, prompt_data in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Generating: {prompt_data['id']}")

        record = generate_with_logging(
            args.server_url,
            prompt_data["prompt"],
            prompt_data["id"],
            max_tokens=args.max_tokens,
        )

        # Attach draft logs if available
        if args.draft_log_dir:
            log_path = os.path.join(args.draft_log_dir, f"{prompt_data['id']}.jsonl")
            record = reconstruct_steps_offline(record, log_path)

        records.append(record)

        # Save incrementally
        out_path = output_dir / f"{prompt_data['id']}.json"
        with open(out_path, "w") as f:
            json.dump(asdict(record), f, indent=2)

    # Save summary
    summary = {
        "dataset": args.dataset,
        "num_samples": len(records),
        "avg_tokens": sum(r.total_tokens for r in records) / max(len(records), 1),
        "avg_accepted": sum(r.total_accepted for r in records) / max(len(records), 1),
        "avg_wall_time": sum(r.wall_time_s for r in records) / max(len(records), 1),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results saved to {output_dir}")
    print(f"  Avg tokens: {summary['avg_tokens']:.1f}")
    print(f"  Avg accepted: {summary['avg_accepted']:.1f}")
    print(f"  Avg wall time: {summary['avg_wall_time']:.3f}s")


if __name__ == "__main__":
    main()
