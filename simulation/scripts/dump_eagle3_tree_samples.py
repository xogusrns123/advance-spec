"""Extract a few EAGLE3 draft-tree samples for visualization.

For the first N (request, step) records, save:
  - eagle3 tree topology (token_ids, parents)
  - path_draft_p_t (cumulative path-product) → per-step probability
    derivable as cum[i] / cum[parent[i]]
  - ground_truth_future (first ``max_gt`` tokens, for accept-walk highlight)
  - detokenized BPE strings for each tree node + each gt token

Output JSON is small (few KB per sample); the visualization notebook
reads it without touching the multi-GB capture again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.pipeline.assemble_records import (  # noqa: E402
    assemble_records_from_artifacts,
)


def _decode_tokens(tokenizer, ids: List[int]) -> List[str]:
    """Map ids → BPE strings (NOT a merged decode, so each node is one token)."""
    try:
        return [str(tokenizer.convert_ids_to_tokens(int(i))) for i in ids]
    except Exception:
        return [f"<id={int(i)}>" for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-trajectory", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--responses", default=None)
    ap.add_argument("--model", required=True,
                    help="HF model name for tokenizer (e.g., Qwen/Qwen3-14B)")
    ap.add_argument("--reslice-steps", type=int, default=8)
    ap.add_argument("--reslice-topk", type=int, default=8)
    ap.add_argument("--capture-steps", type=int, default=8)
    ap.add_argument("--capture-topk", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=5,
                    help="How many (req, step) records to dump.")
    ap.add_argument("--max-gt", type=int, default=16,
                    help="Truncate ground_truth_future to this many tokens.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    eagle3_reslice = (args.capture_steps, args.capture_topk,
                      args.reslice_steps, args.reslice_topk)

    print(f"[dump] loading capture: {args.agent_trajectory}", file=sys.stderr)
    records = assemble_records_from_artifacts(
        agent_trajectory_path=args.agent_trajectory,
        suffix_drafts_path=None, draft_model_drafts_path=None,
        mtp_agent_trajectory_path=None, exclude_path=None,
        model=args.model, dataset_path=args.dataset,
        responses_path=args.responses, eagle3_reslice=eagle3_reslice,
    )
    print(f"[dump] {len(records)} records loaded", file=sys.stderr)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    samples: List[Dict[str, Any]] = []
    for rec in records:
        if len(samples) >= args.n_samples:
            break
        gt = rec.get("ground_truth_future") or []
        base = (rec.get("per_proposer") or {}).get("eagle3")
        if not gt or not base or not base.get("token_ids"):
            continue
        token_ids = list(base["token_ids"])
        parents = list(base["parents"])
        pdpt = base.get("path_draft_p_t")
        if pdpt is None or len(pdpt) != len(token_ids):
            continue
        cum = [float(x) for x in pdpt]
        gt_short = list(gt[:args.max_gt])
        samples.append({
            "request_id": rec.get("request_id"),
            "call_idx": rec.get("call_idx", 0),
            "step_idx": rec.get("step_idx"),
            "token_ids": [int(t) for t in token_ids],
            "parents": [int(p) for p in parents],
            "path_draft_p_t": cum,
            "ground_truth_future": [int(t) for t in gt_short],
            "token_strs": _decode_tokens(tokenizer, token_ids),
            "gt_strs": _decode_tokens(tokenizer, gt_short),
            "n_nodes": len(token_ids),
        })

    out = {
        "metadata": {
            "input_source": args.agent_trajectory,
            "model": args.model,
            "reslice": {"S": args.capture_steps, "K": args.capture_topk,
                        "s": args.reslice_steps, "k": args.reslice_topk},
            "n_samples": len(samples),
            "max_gt": args.max_gt,
            "_doc": (
                "Each sample carries one resliced EAGLE3 tree (BFS-ordered "
                "token_ids/parents), the cumulative path-product "
                "path_draft_p_t (per-node single-step probability is "
                "cum[i] / cum[parent[i]], root divided by 1), the next "
                "max_gt ground-truth tokens, and BPE-string forms via "
                "AutoTokenizer.convert_ids_to_tokens. The visualization "
                "notebook builds the layout and highlights the greedy-walk "
                "accepted path."),
        },
        "samples": samples,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[dump] wrote {len(samples)} samples → {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
