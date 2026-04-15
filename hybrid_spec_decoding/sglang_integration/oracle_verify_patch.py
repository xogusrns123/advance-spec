"""
Oracle Verify Patch: Replace suffix_worker speculation with pre-built union tries.

When SGLANG_ORACLE_VERIFY_TRIES=<path> is set, the suffix_worker reads
union tries from a JSONL file instead of using suffix cache speculation.
Each step's union trie is verified through the target model's tree
attention forward pass, and the resulting logits/p_t are logged.

Combined with oracle_patch.py's verify_tree_greedy_func patch (accept_length=0),
this ensures 1-token advance per step while capturing full tree logits.

Usage:
    export SGLANG_ORACLE_VANILLA=1
    export SGLANG_ORACLE_VERIFY_TRIES=results/.../union_trie_data.jsonl
    python3 -m hybrid_spec_decoding.sglang_integration.install_hook \
        --model-path Qwen/Qwen3-8B --tp-size 1 \
        --speculative-algorithm SUFFIX ...
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from hybrid_spec_decoding.sglang_integration.suffix_worker import SuffixWorker

logger = logging.getLogger(__name__)

VERIFY_TRIES_PATH = os.environ.get("SGLANG_ORACLE_VERIFY_TRIES", "")


def is_verify_tries_enabled() -> bool:
    return bool(VERIFY_TRIES_PATH)


class UnionTrieFeeder:
    """Feeds pre-built union tries sequentially.

    Simply pops records in JSONL order. Works correctly when requests
    are processed sequentially (num_workers=1), since decode step order
    matches the record order in the file. All TP ranks pop in lockstep
    because they process the same batches simultaneously.
    """

    def __init__(self, jsonl_path: str, rid_map_path: str | None = None):
        self._records: List[dict] = []
        self._pos: int = 0

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

        logger.info(f"UnionTrieFeeder: loaded {len(self._records)} records (sequential mode)")

    def get_next_trie(self) -> Optional[dict]:
        """Pop the next union trie record in order."""
        if self._pos >= len(self._records):
            return None
        rec = self._records[self._pos]
        self._pos += 1
        return rec


def patch_suffix_worker_for_verify(suffix_worker: "SuffixWorker") -> None:
    """Patch suffix_worker to use pre-built union tries instead of speculation.

    Replaces _prepare_draft_tokens to read from UnionTrieFeeder.
    Also patches forward_batch_generation to capture and log per-node p_t.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    feeder = UnionTrieFeeder(VERIFY_TRIES_PATH)
    suffix_worker._oracle_trie_feeder = feeder

    original_prepare = suffix_worker._prepare_draft_tokens

    def patched_prepare_draft_tokens(batch):
        bs = batch.batch_size()
        D = suffix_worker.draft_token_num

        all_drafts = np.empty(bs * D, dtype=np.int64)
        all_masks = np.zeros(bs * D * D, dtype=bool)

        for i, req in enumerate(batch.reqs):
            req_id = req.rid
            context = list(req.origin_input_ids) + list(req.output_ids)
            last_token = context[-1] if context else 0

            # Get union trie from feeder
            rec = feeder.get_next_trie(req_id)
            if rec is not None:
                trie = rec.get("union_trie", {})
                token_ids = trie.get("token_ids", [])
                parents = trie.get("parents", [])
            else:
                token_ids = []
                parents = []

            # Truncate to fit draft_token_num - 1 (position 0 is verified_id)
            max_nodes = D - 1
            if len(token_ids) > max_nodes:
                token_ids = token_ids[:max_nodes]
                parents = parents[:max_nodes]

            drafts, mask = suffix_worker._suffix_draft_to_numpy(
                token_ids, parents, last_token
            )

            all_drafts[i * D: (i + 1) * D] = drafts
            all_masks[i * D * D: (i + 1) * D * D] = mask.flatten()

        return all_drafts, all_masks

    suffix_worker._prepare_draft_tokens = patched_prepare_draft_tokens

    # Patch forward to capture logits and compute p_t
    original_forward = suffix_worker.forward_batch_generation
    log_path = Path("/tmp/sglang_oracle_verify_p_t.jsonl")

    def patched_forward(batch):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        # Capture logits from target forward
        original_target_forward = suffix_worker.target_worker.forward_batch_generation
        captured_logits = [None]

        def capturing_forward(model_worker_batch, is_verify=False):
            result = original_target_forward(model_worker_batch, is_verify=is_verify)
            if is_verify and result.logits_output is not None:
                try:
                    logits = result.logits_output.next_token_logits
                    if logits is not None:
                        captured_logits[0] = logits.cpu().clone()
                except Exception:
                    pass
            return result

        suffix_worker.target_worker.forward_batch_generation = capturing_forward
        result = original_forward(batch)
        suffix_worker.target_worker.forward_batch_generation = original_target_forward

        # Log p_t from captured logits
        if captured_logits[0] is not None:
            try:
                _log_verify_p_t(batch, captured_logits[0], feeder, log_path,
                                suffix_worker.draft_token_num)
            except Exception as e:
                logger.debug(f"Verify p_t logging failed: {e}")

        return result

    suffix_worker.forward_batch_generation = patched_forward

    # Patch finalize to clean up feeder state
    original_finalize = suffix_worker._finalize_completed_requests

    def patched_finalize(batch):
        result = original_finalize(batch)
        # Reset feeder for completed requests
        for req in batch.reqs:
            if req.finished():
                feeder.reset_request(req.rid)
        return result

    suffix_worker._finalize_completed_requests = patched_finalize

    logger.info("Patched suffix_worker for union trie verification")


def _log_verify_p_t(batch, logits, feeder, log_path, draft_token_num):
    """Compute and log per-node p_t from verification logits."""
    import torch
    import torch.nn.functional as F

    D = draft_token_num

    for i, req in enumerate(batch.reqs):
        req_id = req.rid

        # Get the union trie that was used (it was consumed by get_next_trie,
        # so we peek at the previous position)
        records = feeder.tries.get(req_id, [])
        pos = feeder._positions.get(req_id, 0) - 1
        if pos < 0 or pos >= len(records):
            continue
        rec = records[pos]
        trie = rec.get("union_trie", {})
        token_ids = trie.get("token_ids", [])
        parents = trie.get("parents", [])
        n = min(len(token_ids), D - 1)  # truncated size

        if n == 0:
            continue

        # Logits layout: [bs * D, vocab_size]
        # Position 0 = verified_id, positions 1..n = trie nodes
        req_offset = i * D

        p_t = []
        for j in range(n):
            tid = token_ids[j]
            if parents[j] == -1:
                # Root child: parent is position 0 (verified_id)
                parent_pos = req_offset + 0
            else:
                # Parent is trie node parents[j], at position parents[j] + 1
                parent_pos = req_offset + parents[j] + 1

            if parent_pos >= len(logits):
                p_t.append(0.0)
                continue

            probs = F.softmax(logits[parent_pos].float(), dim=-1)
            p_t.append(probs[tid].item())

        entry = {
            "request_id": rec.get("request_id", req_id),
            "call_idx": rec.get("call_idx", 0),
            "step_idx": rec.get("step_idx", 0),
            "p_t": p_t,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
