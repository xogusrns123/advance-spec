"""Collect target model probabilities p_t for union trie nodes.

Reads union_trie_data.jsonl (from collect_union_trie.py) and computes
p_t(v|parent(v)) for each node via HuggingFace model forward pass
with tree attention.

Usage:
    python -m hybrid_spec_decoding.analysis.collect_target_probs \
        --union-trie-data results/.../union_trie_data.jsonl \
        --output results/.../union_trie_data_with_pt.jsonl \
        --model zai-org/GLM-4.7-Flash

    # Oracle-only mode (no GPU needed):
    python -m hybrid_spec_decoding.analysis.collect_target_probs \
        --union-trie-data results/.../union_trie_data.jsonl \
        --output results/.../union_trie_data_with_pt.jsonl \
        --oracle-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Tree attention mask construction
# ---------------------------------------------------------------------------

def build_tree_attention_mask(
    context_len: int,
    trie_token_ids: List[int],
    trie_parents: List[int],
) -> List[List[bool]]:
    """Build full attention mask for context + trie tokens.

    Returns mask of shape [total_len, total_len] where
    total_len = context_len + num_trie_nodes.

    - Context tokens: standard causal mask.
    - Trie tokens: attend to all context + ancestors in trie.
    """
    n_trie = len(trie_token_ids)
    total = context_len + n_trie

    mask = [[False] * total for _ in range(total)]

    # Context: causal mask
    for i in range(context_len):
        for j in range(i + 1):
            mask[i][j] = True

    # Trie nodes
    for i in range(n_trie):
        row = context_len + i
        # Attend to all context tokens
        for j in range(context_len):
            mask[row][j] = True
        # Attend to self
        mask[row][row] = True
        # Attend to ancestors in trie
        parent = trie_parents[i]
        while parent != -1:
            mask[row][context_len + parent] = True
            parent = trie_parents[parent]

    return mask


def build_position_ids(
    context_len: int,
    trie_parents: List[int],
) -> List[int]:
    """Build position IDs for context + trie tokens.

    Context: [0, 1, ..., context_len-1]
    Trie nodes: context_len + depth - 1 (tree-aware positioning)
    """
    # Compute depth of each trie node
    n_trie = len(trie_parents)
    depths = [0] * n_trie
    for i in range(n_trie):
        if trie_parents[i] == -1:
            depths[i] = 1
        else:
            depths[i] = depths[trie_parents[i]] + 1

    context_positions = list(range(context_len))
    trie_positions = [context_len + d - 1 for d in depths]

    return context_positions + trie_positions


def extract_parent_logits(
    all_logits: "torch.Tensor",
    context_len: int,
    trie_parents: List[int],
) -> "torch.Tensor":
    """Extract logits at each trie node's parent position.

    For root children (parent=-1): use logits at context_len - 1
    For deeper nodes (parent=j): use logits at context_len + j

    Returns tensor of shape [n_trie, vocab_size].
    """
    n_trie = len(trie_parents)
    parent_indices = []
    for i in range(n_trie):
        if trie_parents[i] == -1:
            parent_indices.append(context_len - 1)
        else:
            parent_indices.append(context_len + trie_parents[i])

    return all_logits[parent_indices]  # [n_trie, vocab_size]


# ---------------------------------------------------------------------------
# p_t computation
# ---------------------------------------------------------------------------

def compute_p_t_from_parent_logits(
    parent_logits: "torch.Tensor",
    trie_token_ids: List[int],
) -> List[float]:
    """Compute p_t(v|parent(v)) = softmax(parent_logits)[v.token_id].

    Parameters
    ----------
    parent_logits : torch.Tensor, shape [n_trie, vocab_size]
    trie_token_ids : list[int], length n_trie

    Returns
    -------
    p_t : list[float]
    """
    probs = F.softmax(parent_logits.float(), dim=-1)
    p_t = []
    for i, tid in enumerate(trie_token_ids):
        p_t.append(probs[i, tid].item())
    return p_t


def _build_mask_tensor(
    context_len: int,
    trie_tids: List[int],
    trie_parents: List[int],
    dtype: "torch.dtype",
) -> "torch.Tensor":
    """Build 4D attention mask tensor efficiently using torch ops."""
    n_trie = len(trie_tids)
    total = context_len + n_trie
    min_val = torch.finfo(dtype).min

    # Start with all masked
    mask = torch.full((1, 1, total, total), min_val, dtype=dtype)

    # Context: causal mask (lower triangular)
    if context_len > 0:
        mask[0, 0, :context_len, :context_len] = torch.where(
            torch.tril(torch.ones(context_len, context_len, dtype=torch.bool)),
            torch.tensor(0.0, dtype=dtype),
            torch.tensor(min_val, dtype=dtype),
        )

    # Trie nodes
    for i in range(n_trie):
        row = context_len + i
        # Attend to all context
        mask[0, 0, row, :context_len] = 0.0
        # Attend to self
        mask[0, 0, row, row] = 0.0
        # Attend to ancestors
        parent = trie_parents[i]
        while parent != -1:
            mask[0, 0, row, context_len + parent] = 0.0
            parent = trie_parents[parent]

    return mask


def _trim_past_kv(past_key_values, keep_len: int):
    """Trim KV cache to keep only first keep_len positions.

    Works with various past_key_values formats (tuple of tuples,
    DynamicCache, etc.).
    """
    if past_key_values is None:
        return None

    try:
        # Try DynamicCache API first (transformers >= 4.36)
        if hasattr(past_key_values, "crop"):
            past_key_values.crop(keep_len)
            return past_key_values

        # Try key_cache/value_cache attributes (DynamicCache)
        if hasattr(past_key_values, "key_cache"):
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, :keep_len, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, :keep_len, :]
            return past_key_values

        # Legacy tuple format
        new_past = []
        for layer_kv in past_key_values:
            trimmed = tuple(t[:, :, :keep_len, :] for t in layer_kv)
            new_past.append(trimmed)
        return tuple(new_past)
    except Exception:
        return None


def collect_p_t_for_records(
    records: List[dict],
    model,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 1,
) -> None:
    """Compute real p_t for all records using model forward passes.

    Uses KV cache to avoid recomputing context prefix for consecutive
    steps within the same request/call.  Modifies records in-place.
    """
    model.eval()
    n = len(records)
    t0 = time.time()
    model_dtype = next(model.parameters()).dtype

    # Group records by (request_id, call_idx) for KV cache reuse
    prev_key = None
    past_kv = None
    cached_ctx_len = 0

    for i, rec in enumerate(records):
        context_ids = rec.get("context_token_ids", [])
        trie_tids = rec["union_trie"]["token_ids"]
        trie_parents = rec["union_trie"]["parents"]

        if not context_ids or not trie_tids:
            rec["p_t"] = [0.0] * len(trie_tids)
            continue

        context_len = len(context_ids)
        n_trie = len(trie_tids)
        curr_key = (rec.get("request_id"), rec.get("call_idx"))

        # Check if we can reuse KV cache from previous step
        if (prev_key == curr_key and past_kv is not None
                and cached_ctx_len < context_len):
            # Incremental: only forward the new context tokens + trie
            new_ctx = context_ids[cached_ctx_len:]
            input_ids = new_ctx + trie_tids
            new_ctx_len = len(new_ctx)
            total_new = len(input_ids)

            # Position IDs: continue from cached_ctx_len
            pos_ctx = list(range(cached_ctx_len, context_len))
            # Trie nodes: depth-based positions
            depths = [0] * n_trie
            for j in range(n_trie):
                if trie_parents[j] == -1:
                    depths[j] = 1
                else:
                    depths[j] = depths[trie_parents[j]] + 1
            pos_trie = [context_len + d - 1 for d in depths]
            pos_ids = pos_ctx + pos_trie

            # Attention mask: (total_new) attends to (cached + total_new)
            full_len = cached_ctx_len + total_new
            mask = torch.full((1, 1, total_new, full_len), torch.finfo(model_dtype).min,
                              dtype=model_dtype)
            # New context tokens: causal over cached + new context
            for j in range(new_ctx_len):
                mask[0, 0, j, :cached_ctx_len + j + 1] = 0.0
            # Trie nodes: attend to all context (cached + new) + ancestors
            for j in range(n_trie):
                row = new_ctx_len + j
                mask[0, 0, row, :cached_ctx_len + new_ctx_len] = 0.0  # all context
                mask[0, 0, row, new_ctx_len + j] = 0.0  # self
                parent = trie_parents[j]
                while parent != -1:
                    mask[0, 0, row, new_ctx_len + parent] = 0.0
                    parent = trie_parents[parent]

            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            pos_tensor = torch.tensor([pos_ids], dtype=torch.long)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_tensor,
                    attention_mask=mask,
                    position_ids=pos_tensor,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                all_logits = outputs.logits[0].float().cpu()  # [total_new, vocab]

            # Extract p_t: parent logits are at positions relative to new output
            # For trie node i, parent is at:
            #   root child (parent=-1): last new context token = new_ctx_len - 1
            #   deeper (parent=j): new_ctx_len + j
            p_t = []
            for j in range(n_trie):
                if trie_parents[j] == -1:
                    parent_pos = new_ctx_len - 1
                else:
                    parent_pos = new_ctx_len + trie_parents[j]
                probs = F.softmax(all_logits[parent_pos], dim=-1)
                p_t.append(probs[trie_tids[j]].item())
            rec["p_t"] = p_t

            # Update cache: trim to context only (discard trie KV entries)
            past_kv = _trim_past_kv(outputs.past_key_values, context_len)
            cached_ctx_len = context_len if past_kv is not None else 0

        else:
            # Full forward: no cache or different request
            past_kv = None
            cached_ctx_len = 0

            input_ids = context_ids + trie_tids
            total_len = len(input_ids)

            mask = _build_mask_tensor(context_len, trie_tids, trie_parents, model_dtype)
            pos_ids = build_position_ids(context_len, trie_parents)

            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            pos_tensor = torch.tensor([pos_ids], dtype=torch.long)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_tensor,
                    attention_mask=mask,
                    position_ids=pos_tensor,
                    use_cache=True,
                )
                logits = outputs.logits[0].float().cpu()

            parent_logits = extract_parent_logits(logits, context_len, trie_parents)
            p_t = compute_p_t_from_parent_logits(parent_logits, trie_tids)
            rec["p_t"] = p_t

            # Cache context KV for next step
            past_kv = _trim_past_kv(outputs.past_key_values, context_len)
            cached_ctx_len = context_len if past_kv is not None else 0

        prev_key = curr_key

        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {rate:.1f} steps/s, ETA {eta:.0f}s",
                  file=sys.stderr)


# ---------------------------------------------------------------------------
# Server-based p_t collection
# ---------------------------------------------------------------------------

def collect_p_t_via_server(
    records: List[dict],
    server_url: str,
) -> None:
    """Collect p_t by sending requests to verify_server.py."""
    import requests as http_requests

    n = len(records)
    t0 = time.time()
    url = f"{server_url.rstrip('/')}/verify_tree"

    for i, rec in enumerate(records):
        context_ids = rec.get("context_token_ids", [])
        trie_tids = rec["union_trie"]["token_ids"]
        trie_parents = rec["union_trie"]["parents"]

        if not context_ids or not trie_tids:
            rec["p_t"] = [0.0] * len(trie_tids)
            continue

        payload = {
            "context_token_ids": context_ids,
            "tree_token_ids": trie_tids,
            "tree_parents": trie_parents,
            "request_id": rec.get("request_id"),
            "call_idx": rec.get("call_idx"),
        }
        resp = http_requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        rec["p_t"] = resp.json()["p_t"]

        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {rate:.1f} steps/s, ETA {eta:.0f}s",
                  file=sys.stderr)


# ---------------------------------------------------------------------------
# Oracle p_t (ground truth baseline)
# ---------------------------------------------------------------------------

def enrich_with_ground_truth_p_t(records: List[dict]) -> List[dict]:
    """Add oracle p_t (1.0 for correct, 0.0 for incorrect) as a baseline."""
    for rec in records:
        token_ids = rec["union_trie"]["token_ids"]
        parents = rec["union_trie"]["parents"]
        gt = rec["ground_truth_future"]

        n = len(token_ids)
        p_t_oracle = [0.0] * n

        children: Dict[int, List[int]] = {-1: []}
        for i in range(n):
            children.setdefault(parents[i], []).append(i)
            children.setdefault(i, [])

        node = -1
        for gt_token in gt:
            for child_idx in children.get(node, []):
                if token_ids[child_idx] == gt_token:
                    p_t_oracle[child_idx] = 1.0
                    node = child_idx
                    break
            else:
                break

        rec["p_t_oracle"] = p_t_oracle

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--union-trie-data", required=True,
                        help="Input JSONL from collect_union_trie.py")
    parser.add_argument("--output", required=True,
                        help="Output JSONL with p_t values added")
    parser.add_argument("--oracle-only", action="store_true",
                        help="Only add ground-truth oracle p_t (no model needed)")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model name or path (local mode)")
    parser.add_argument("--verify-server-url", default=None,
                        help="URL of verify_server.py (e.g. http://localhost:8100)")
    parser.add_argument("--device", default="cuda",
                        help="Device for model (default: cuda)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N records (for testing)")
    args = parser.parse_args()

    # Load records
    print(f"Loading: {args.union_trie_data}", file=sys.stderr)
    records = []
    with open(args.union_trie_data) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} step records", file=sys.stderr)

    # Add oracle p_t (always)
    enrich_with_ground_truth_p_t(records)
    print(f"Added oracle p_t (ground truth)", file=sys.stderr)

    if not args.oracle_only:
        if not args.model and not args.verify_server_url:
            parser.error("--model or --verify-server-url required unless --oracle-only")

        # Check context_token_ids availability
        has_context = any("context_token_ids" in r for r in records)
        if not has_context:
            parser.error(
                "Records missing context_token_ids. "
                "Re-run collect_union_trie.py to generate updated data."
            )

        t0 = time.time()

        if args.verify_server_url:
            # Server mode: send requests to verify_server.py
            collect_p_t_via_server(records, args.verify_server_url)
        else:
            # Local mode: load model directly
            if not HAS_TORCH:
                parser.error("PyTorch required for local model-based p_t collection")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading model: {args.model}", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(args.model,
                                                       trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.float16,
                trust_remote_code=True, device_map="auto",
            )
            print(f"Model loaded on {args.device}", file=sys.stderr)
            collect_p_t_for_records(records, model, tokenizer, device=args.device)

        print(f"p_t collection: {time.time() - t0:.1f}s", file=sys.stderr)

        # Sanity check: compare with oracle
        n_checked = 0
        n_agree = 0
        for rec in records:
            if "p_t" not in rec or "p_t_oracle" not in rec:
                continue
            for pi, (pt_real, pt_oracle) in enumerate(
                    zip(rec["p_t"], rec["p_t_oracle"])):
                n_checked += 1
                if pt_oracle == 1.0 and pt_real > 0.01:
                    n_agree += 1
                elif pt_oracle == 0.0:
                    n_agree += 1  # don't penalise wrong-path nodes
        if n_checked > 0:
            print(f"Sanity: {n_agree}/{n_checked} nodes agree "
                  f"({100*n_agree/n_checked:.1f}%)", file=sys.stderr)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Records: {len(records)}", file=sys.stderr)


if __name__ == "__main__":
    main()
