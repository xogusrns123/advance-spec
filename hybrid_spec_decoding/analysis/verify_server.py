"""Lightweight tree verification server for union trie p_t collection.

Loads target model via HuggingFace and serves a /verify_tree endpoint
that accepts context + tree structure, runs tree attention forward,
and returns per-node p_t.

Uses KV cache for consecutive steps within the same request to avoid
redundant context computation.

Usage:
    python -m hybrid_spec_decoding.analysis.verify_server \
        --model Qwen/Qwen3-8B --port 8100

    # Then from collect_target_probs.py:
    python -m hybrid_spec_decoding.analysis.collect_target_probs \
        --union-trie-data union_trie_data.jsonl \
        --verify-server-url http://localhost:8100 \
        --output union_trie_data_with_pt.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from hybrid_spec_decoding.analysis.collect_target_probs import (
    _build_mask_tensor,
    _trim_past_kv,
    build_position_ids,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Tree Verify Server")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class VerifyTreeRequest(BaseModel):
    context_token_ids: List[int]
    tree_token_ids: List[int]
    tree_parents: List[int]
    request_id: Optional[str] = None
    call_idx: Optional[int] = None
    step_idx: Optional[int] = None


class VerifyTreeResponse(BaseModel):
    p_t: List[float]


class VerifyBatchRequest(BaseModel):
    records: List[VerifyTreeRequest]


class VerifyBatchResponse(BaseModel):
    results: List[VerifyTreeResponse]


# ---------------------------------------------------------------------------
# Model state (module-level, initialized at startup)
# ---------------------------------------------------------------------------

_state: Dict = {
    "model": None,
    "model_dtype": None,
    "device": "cuda",
    # KV cache state
    "prev_key": None,
    "past_kv": None,
    "cached_ctx_len": 0,
}


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def verify_single(
    context_ids: List[int],
    trie_tids: List[int],
    trie_parents: List[int],
    request_key: Optional[Tuple] = None,
) -> List[float]:
    """Run tree attention forward and return per-node p_t."""
    model = _state["model"]
    model_dtype = _state["model_dtype"]

    if not context_ids or not trie_tids:
        return [0.0] * len(trie_tids)

    context_len = len(context_ids)
    n_trie = len(trie_tids)

    # Check KV cache reuse
    prev_key = _state["prev_key"]
    past_kv = _state["past_kv"]
    cached_ctx_len = _state["cached_ctx_len"]

    if (request_key and prev_key == request_key
            and past_kv is not None and cached_ctx_len < context_len):
        # Incremental: forward only new context tokens + trie
        new_ctx = context_ids[cached_ctx_len:]
        input_ids = new_ctx + trie_tids
        new_ctx_len = len(new_ctx)

        # Position IDs
        pos_ctx = list(range(cached_ctx_len, context_len))
        depths = [0] * n_trie
        for j in range(n_trie):
            depths[j] = 1 if trie_parents[j] == -1 else depths[trie_parents[j]] + 1
        pos_trie = [context_len + d - 1 for d in depths]
        pos_ids = pos_ctx + pos_trie

        # Attention mask
        total_new = len(input_ids)
        full_len = cached_ctx_len + total_new
        mask = torch.full((1, 1, total_new, full_len),
                          torch.finfo(model_dtype).min, dtype=model_dtype)
        for j in range(new_ctx_len):
            mask[0, 0, j, :cached_ctx_len + j + 1] = 0.0
        for j in range(n_trie):
            row = new_ctx_len + j
            mask[0, 0, row, :cached_ctx_len + new_ctx_len] = 0.0
            mask[0, 0, row, new_ctx_len + j] = 0.0
            parent = trie_parents[j]
            while parent != -1:
                mask[0, 0, row, new_ctx_len + parent] = 0.0
                parent = trie_parents[parent]

        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor([input_ids], dtype=torch.long),
                attention_mask=mask,
                position_ids=torch.tensor([pos_ids], dtype=torch.long),
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = outputs.logits[0].float().cpu()

        # Extract p_t
        p_t = []
        for j in range(n_trie):
            parent_pos = (new_ctx_len - 1) if trie_parents[j] == -1 else (new_ctx_len + trie_parents[j])
            probs = F.softmax(logits[parent_pos], dim=-1)
            p_t.append(probs[trie_tids[j]].item())

        # Update cache
        _state["past_kv"] = _trim_past_kv(outputs.past_key_values, context_len)
        _state["cached_ctx_len"] = context_len if _state["past_kv"] is not None else 0

    else:
        # Full forward
        _state["past_kv"] = None
        _state["cached_ctx_len"] = 0

        input_ids = context_ids + trie_tids
        mask = _build_mask_tensor(context_len, trie_tids, trie_parents, model_dtype)
        pos_ids = build_position_ids(context_len, trie_parents)

        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor([input_ids], dtype=torch.long),
                attention_mask=mask,
                position_ids=torch.tensor([pos_ids], dtype=torch.long),
                use_cache=True,
            )
            logits = outputs.logits[0].float().cpu()

        # Extract p_t from parent positions
        p_t = []
        for j in range(n_trie):
            parent_pos = (context_len - 1) if trie_parents[j] == -1 else (context_len + trie_parents[j])
            probs = F.softmax(logits[parent_pos], dim=-1)
            p_t.append(probs[trie_tids[j]].item())

        # Cache
        _state["past_kv"] = _trim_past_kv(outputs.past_key_values, context_len)
        _state["cached_ctx_len"] = context_len if _state["past_kv"] is not None else 0

    _state["prev_key"] = request_key
    return p_t


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/verify_tree", response_model=VerifyTreeResponse)
def verify_tree(req: VerifyTreeRequest):
    key = (req.request_id, req.call_idx) if req.request_id else None
    p_t = verify_single(req.context_token_ids, req.tree_token_ids,
                         req.tree_parents, request_key=key)
    return VerifyTreeResponse(p_t=p_t)


@app.post("/verify_batch", response_model=VerifyBatchResponse)
def verify_batch(req: VerifyBatchRequest):
    results = []
    for r in req.records:
        key = (r.request_id, r.call_idx) if r.request_id else None
        p_t = verify_single(r.context_token_ids, r.tree_token_ids,
                             r.tree_parents, request_key=key)
        results.append(VerifyTreeResponse(p_t=p_t))
    return VerifyBatchResponse(results=results)


class BenchmarkRequest(BaseModel):
    context_token_ids: List[int]
    tree_token_ids: List[int]
    tree_parents: List[int]
    budgets: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    n_trials: int = 30


class BenchmarkResponse(BaseModel):
    vanilla_step_ms: float
    verify_latencies: Dict[str, float]  # budget → ms


@app.post("/benchmark_verify", response_model=BenchmarkResponse)
def benchmark_verify(req: BenchmarkRequest):
    """Measure verify latency at different budget sizes.

    Uses full forward (context + trie) for each budget size.
    Vanilla = context + 1 token (B=0).
    Verify(B) = context + B trie nodes with tree attention.
    """
    import statistics

    model = _state["model"]
    model_dtype = _state["model_dtype"]
    ctx = req.context_token_ids
    full_tids = req.tree_token_ids
    full_pids = req.tree_parents
    context_len = len(ctx)

    def _measure(input_ids, mask, pos_ids, n_warmup=3, n_trials=None):
        if n_trials is None:
            n_trials = req.n_trials
        inp = torch.tensor([input_ids], dtype=torch.long)
        pos = torch.tensor([pos_ids], dtype=torch.long)
        for _ in range(n_warmup):
            with torch.no_grad():
                model(input_ids=inp, attention_mask=mask,
                      position_ids=pos, use_cache=False)
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(input_ids=inp, attention_mask=mask,
                      position_ids=pos, use_cache=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return statistics.median(times)

    # Vanilla: context + 1 token (causal)
    vanilla_ids = ctx + [full_tids[0] if full_tids else 0]
    vanilla_len = len(vanilla_ids)
    vanilla_mask = _build_mask_tensor(context_len, [vanilla_ids[-1]], [-1], model_dtype)
    vanilla_pos = list(range(context_len)) + [context_len]
    vanilla_ms = _measure(vanilla_ids, vanilla_mask, vanilla_pos)

    # Verify at each budget
    results = {}
    for B in req.budgets:
        tids = full_tids[:B]
        pids = full_pids[:B]
        pids = [p if p < B else -1 for p in pids]

        mask = _build_mask_tensor(context_len, tids, pids, model_dtype)
        pos = build_position_ids(context_len, pids)
        input_ids = ctx + tids

        ms = _measure(input_ids, mask, pos)
        results[str(B)] = ms

    return BenchmarkResponse(vanilla_step_ms=vanilla_ms, verify_latencies=results)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16,
        trust_remote_code=True, device_map="auto")
    model.eval()

    _state["model"] = model
    _state["model_dtype"] = next(model.parameters()).dtype
    _state["device"] = args.device

    print(f"Model loaded. Starting server on {args.host}:{args.port}",
          file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
