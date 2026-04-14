"""
SuffixWorker: model-free speculative decoding via suffix tree matching.

Follows the NGRAMWorker pattern from SGLang v0.5.10:
- No draft model required (model-free, like NGRAMWorker)
- Uses arctic-inference SuffixDecodingCache for draft token generation
- Reuses NgramVerifyInput for target model verification
- Compatible with --speculative-algorithm SUFFIX

The suffix tree accumulates patterns from completed requests and uses
them to predict future tokens. Requires warming (via prior requests
or explicit corpus) to produce useful drafts.
"""

from __future__ import annotations

# Limit torch.compile workers BEFORE any torch import.
# This module is imported by spawned scheduler processes, so this is the
# earliest point to set it for child processes.
# GLM-4.7-Flash MoE topk uses @torch.compile, spawning min(32, cpu_count)
# subprocesses per TP rank.
import os as _os
if "TORCHINDUCTOR_COMPILE_THREADS" not in _os.environ:
    _os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch

from arctic_inference.suffix_decoding import SuffixDecodingCache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class SuffixWorker:
    """
    Speculative decoding worker using suffix tree pattern matching.

    Mirrors NGRAMWorker's interface so it can be used as a drop-in
    replacement via SpeculativeAlgorithm.create_worker().
    """

    def __init__(
        self,
        server_args: "ServerArgs",
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: "TpModelWorker",
    ):
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        # Suffix tree cache
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=64,
            max_cached_requests=100000,
        )
        # Track active requests for start/stop lifecycle
        self._active_requests: set[str] = set()

        self._init_preallocated_tensors()
        logger.info(
            f"SuffixWorker initialized (draft_token_num={self.draft_token_num})"
        )

    def clear_cache_pool(self):
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=64,
            max_cached_requests=100000,
        )
        self._active_requests.clear()

    # ------------------------------------------------------------------ #
    #  Preallocated tensors (identical to NGRAMWorker)
    # ------------------------------------------------------------------ #

    def _init_preallocated_tensors(self):
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        # Precomputed batch slices
        self.draft_tokens_batch = []
        self.tree_mask_batch = []
        self.retrieve_indexes_batch = []
        self.retrive_next_token_batch = []
        self.retrive_next_sibling_batch = []
        self.positions_batch = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrive_next_token_batch.append(self.retrive_next_token[:bs, :])
            self.retrive_next_sibling_batch.append(self.retrive_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    # ------------------------------------------------------------------ #
    #  Suffix draft → tree mask conversion
    # ------------------------------------------------------------------ #

    def _suffix_draft_to_numpy(
        self,
        token_ids: list[int],
        parents: list[int],
        last_token_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a suffix draft (token_ids + parents) into the flat numpy
        arrays expected by NGRAMWorker's verification pipeline.

        Must match the format produced by NgramCache.batch_get():
        - drafts[0] = last context token (verified_id)
        - drafts[1..n] = draft candidates (chain from suffix match)
        - drafts[n+1..D-1] = 0 (padding)
        - mask: star structure for padding (only see position 0)
        - mask: causal chain for real tokens

        Args:
            token_ids: Draft token IDs from suffix speculation.
            parents: Parent indices (-1 = child of context/root).
            last_token_id: The last verified token.

        Returns:
            (req_drafts, mask) where:
            - req_drafts: int64 array of shape [draft_token_num]
            - mask: bool array of shape [draft_token_num, draft_token_num]
        """
        n = len(token_ids)
        D = self.draft_token_num

        # Position 0 = verified_id (last context token), rest = draft or padding(0)
        drafts = np.zeros(D, dtype=np.int64)
        drafts[0] = last_token_id
        if n > 0:
            fill = min(n, D - 1)  # leave position 0 for verified_id
            for i in range(fill):
                drafts[i + 1] = token_ids[i]

        # Build tree mask matching NGRAM format:
        # - Position 0 (verified_id): sees only itself
        # - Real tokens: causal chain (sees all ancestors including position 0)
        # - Padding tokens: star (sees only position 0 and itself)
        mask = np.zeros((D, D), dtype=bool)

        # Position 0: verified_id, sees only itself
        mask[0][0] = True

        # Real suffix tokens at positions 1..n: chain from position 0
        real_end = min(n, D - 1) + 1  # +1 because offset by verified_id
        for i in range(1, real_end):
            mask[i][i] = True
            # Each real token sees all previous real tokens (causal chain)
            for j in range(i):
                mask[i][j] = True

        # Padding tokens: star structure (see only position 0 and themselves)
        for i in range(real_end, D):
            mask[i][i] = True
            mask[i][0] = True

        return drafts, mask

    def _prepare_draft_tokens(
        self, batch: "ScheduleBatch"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate draft tokens for each request in the batch using
        suffix tree speculation.

        Returns:
            (all_drafts, all_masks) concatenated across the batch:
            - all_drafts: [bs * draft_token_num]
            - all_masks: [bs * draft_token_num * draft_token_num]
        """
        bs = batch.batch_size()
        D = self.draft_token_num

        all_drafts = np.empty(bs * D, dtype=np.int64)
        all_masks = np.zeros(bs * D * D, dtype=bool)

        for i, req in enumerate(batch.reqs):
            req_id = req.rid
            context = list(req.origin_input_ids) + list(req.output_ids)
            last_token = context[-1] if context else 0

            # Lazy lifecycle: start request on first encounter
            if req_id not in self._active_requests:
                self.suffix_cache.start_request(req_id, list(req.origin_input_ids))
                self._active_requests.add(req_id)

            # Speculate (but IGNORE the result — use fallback only)
            try:
                draft = self.suffix_cache.speculate(
                    req_id, context, max_spec_tokens=D
                )
                token_ids = list(draft.token_ids)
                parents = list(draft.parents)
                if token_ids:
                    logger.warning(
                        f"SUFFIX DRAFT req={req_id}: "
                        f"n_tokens={len(token_ids)}, "
                        f"token_ids={token_ids[:8]}, "
                        f"parents={parents[:8]}, "
                        f"score={draft.score:.2f}, match_len={draft.match_len}"
                    )
            except Exception as e:
                logger.debug(f"Suffix speculation failed for {req_id}: {e}")
                token_ids = []
                parents = []

            drafts, mask = self._suffix_draft_to_numpy(
                token_ids, parents, last_token
            )

            all_drafts[i * D : (i + 1) * D] = drafts
            all_masks[i * D * D : (i + 1) * D * D] = mask.flatten()

        return all_drafts, all_masks

    def _prepare_for_speculative_decoding(self, batch: "ScheduleBatch"):
        """Prepare batch for speculative verification (mirrors NGRAMWorker)."""
        from sglang.srt.speculative.ngram_worker import (
            NgramVerifyInput,
            SpeculativeAlgorithm,
            reconstruct_indices_from_tree_mask,
        )

        if batch.forward_mode.is_extend():
            return

        bs = batch.batch_size()

        retrive_index = self.retrieve_indexes_batch[bs]
        retrive_next_token = self.retrive_next_token_batch[bs]
        retrive_next_sibling = self.retrive_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        req_drafts, mask = self._prepare_draft_tokens(batch)
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            bs,
            self.draft_token_num,
        )

        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM  # reuse NGRAM verify path
        batch.forward_mode = batch.forward_mode.TARGET_VERIFY
        batch.spec_info = NgramVerifyInput(
            draft_tokens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            self.draft_token_num,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    # ------------------------------------------------------------------ #
    #  Suffix cache update
    # ------------------------------------------------------------------ #

    def _finalize_completed_requests(self, batch: "ScheduleBatch"):
        """Feed completed request's full response into suffix tree at once.

        Instead of incrementally calling add_active_response each step
        (which can corrupt the cache), we batch-add the entire response
        only when the request finishes. This is safer and matches the
        pattern used in arctic-inference's simulator.
        """
        for req in batch.reqs:
            if not req.finished():
                continue

            req_id = req.rid
            if req_id not in self._active_requests:
                continue

            try:
                if len(req.output_ids) > 0:
                    self.suffix_cache.add_active_response(
                        req_id, list(req.output_ids)
                    )
                self.suffix_cache.stop_request(req_id)
            except Exception as e:
                logger.debug(f"Suffix finalize failed for {req_id}: {e}")

            self._active_requests.discard(req_id)

    # ------------------------------------------------------------------ #
    #  Main forward loop (mirrors NGRAMWorker)
    # ------------------------------------------------------------------ #

    def forward_batch_generation(self, batch: "ScheduleBatch"):
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.ngram_worker import (
            NgramVerifyInput,
            add_output_logprobs_for_spec_v1,
            generate_token_bitmask,
        )

        self._prepare_for_speculative_decoding(batch)
        model_worker_batch = batch.get_model_worker_batch()
        spec_info = model_worker_batch.spec_info
        num_accepted_tokens = 0
        accept_lens = None

        if model_worker_batch.forward_mode.is_target_verify():
            if batch.has_grammar:
                retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
                retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
                draft_tokens_cpu = spec_info.draft_token.view(
                    spec_info.retrive_next_token.shape
                ).cpu()

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )

            verify_input: NgramVerifyInput = model_worker_batch.spec_info
            vocab_mask = None
            if batch.has_grammar:
                vocab_mask = generate_token_bitmask(
                    batch.reqs,
                    verify_input,
                    retrieve_next_token_cpu,
                    retrieve_next_sibling_cpu,
                    draft_tokens_cpu,
                    batch.sampling_info.vocab_size,
                )
                if vocab_mask is not None:
                    assert verify_input.grammar is not None
                    vocab_mask = vocab_mask.to(verify_input.retrive_next_token.device)
                    batch.sampling_info.vocab_mask = None

            logits_output, next_token_ids, num_accepted_tokens = verify_input.verify(
                batch, logits_output, self.page_size, vocab_mask
            )
            accept_lens = verify_input.accept_length
            if batch.return_logprob:
                add_output_logprobs_for_spec_v1(batch, verify_input, logits_output)
            self._finalize_completed_requests(batch)
            batch.forward_mode = ForwardMode.DECODE

        else:
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
        )
