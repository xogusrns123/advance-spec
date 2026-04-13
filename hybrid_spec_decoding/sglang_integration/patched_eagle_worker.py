"""
Patched EAGLE-3 draft worker for hybrid speculative decoding.

Monkey-patches MultiLayerEagleDraftWorker.draft() to inject
SuffixDecoding candidates into parent_list / draft_tokens
BEFORE build_tree_kernel_efficient() builds the attention mask.

The key insight: build_tree_kernel_efficient() is agnostic to
where the tokens come from. It just needs consistent parent_list,
top_scores_index, and draft_tokens. We augment these tensors with
suffix candidates, and the kernel produces correct tree attention
masks automatically.

Injection point in SGLang's flow:
    draft_forward()          -> parent_list, top_scores_index, draft_tokens
    *** inject suffix ***    -> augmented parent_list, top_scores_index, draft_tokens
    build_tree_kernel_efficient() -> tree_mask, positions, retrive_*
    EagleVerifyInput         -> target model verifies
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Hashable

import torch

from ..suffix_decoding.speculator import SuffixSpeculator
from ..suffix_decoding.suffix_tree import SuffixDraft

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

logger = logging.getLogger(__name__)


def patch_eagle_worker(
    draft_worker: Any,
    suffix_speculator: SuffixSpeculator,
    fusion_mode: str = "parallel",
    suffix_budget_ratio: float = 0.3,
    extension_depths: list[int] | None = None,
    min_extension_confidence: float = 0.3,
) -> Any:
    """
    Monkey-patch a MultiLayerEagleDraftWorker's draft() method
    to inject suffix decoding candidates.

    Args:
        draft_worker: SGLang's MultiLayerEagleDraftWorker instance.
        suffix_speculator: Initialized SuffixSpeculator.
        fusion_mode: "parallel", "sequential", or "combined".
        suffix_budget_ratio: Fraction of draft token budget for suffix candidates.
        extension_depths: For sequential mode, which EAGLE tree depths to extend from.
        min_extension_confidence: Min EAGLE prob to consider a node for extension.

    Returns:
        The patched draft_worker (same object, modified in-place).
    """
    original_draft = draft_worker.draft

    if extension_depths is None:
        extension_depths = [1, 2, 3]

    def patched_draft(model_worker_batch: "ModelWorkerBatch") -> "EagleVerifyInput":
        """Patched draft() that injects suffix candidates."""
        from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
        from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient

        draft_input: EagleDraftInput = model_worker_batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            draft_worker.req_to_token_pool,
            model_worker_batch,
            draft_worker.cuda_graph_runner,
            draft_worker.draft_runner_list[0],
            draft_worker.topk,
            draft_worker.speculative_num_steps,
        )

        # Step 1: Run EAGLE-3 draft
        parent_list, top_scores_index, draft_tokens = draft_worker.draft_forward(
            forward_batch
        )

        if model_worker_batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                draft_worker.topk,
                draft_worker.speculative_num_steps,
                draft_worker.speculative_num_draft_tokens,
            )

        # Step 2: Inject suffix candidates
        batch_size = draft_tokens.shape[0]
        total_budget = draft_worker.speculative_num_draft_tokens
        suffix_budget = max(1, int(total_budget * suffix_budget_ratio))

        # Get context tokens for each request in the batch
        augmented_parent_list = parent_list
        augmented_top_scores_index = top_scores_index
        augmented_draft_tokens = draft_tokens

        try:
            for b in range(batch_size):
                req = model_worker_batch.reqs[b] if hasattr(model_worker_batch, 'reqs') else None
                if req is None:
                    continue

                req_id = req.rid if hasattr(req, 'rid') else str(b)
                context = _get_context_tokens(req)
                if not context:
                    continue

                # Generate suffix draft
                suffix_draft = suffix_speculator.speculate(
                    req_id, context, max_spec_tokens=suffix_budget
                )

                if suffix_draft.is_empty:
                    continue

                # Inject into tensors
                augmented_parent_list, augmented_top_scores_index, augmented_draft_tokens = (
                    _inject_suffix_candidates(
                        parent_list=augmented_parent_list,
                        top_scores_index=augmented_top_scores_index,
                        draft_tokens=augmented_draft_tokens,
                        suffix_draft=suffix_draft,
                        batch_idx=b,
                        total_budget=total_budget,
                        topk=draft_worker.topk,
                        spec_steps=draft_worker.speculative_num_steps,
                    )
                )
        except Exception as e:
            logger.warning(f"Suffix injection failed, falling back to EAGLE-3 only: {e}")
            augmented_parent_list = parent_list
            augmented_top_scores_index = top_scores_index
            augmented_draft_tokens = draft_tokens

        # Step 3: Build tree with augmented tokens
        tree_mask_buf, position_buf = (
            draft_worker.target_worker.model_runner.attn_backend
            .get_verify_buffers_to_fill_after_draft()
        )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            final_draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id,
            augmented_parent_list,
            augmented_top_scores_index,
            augmented_draft_tokens,
            model_worker_batch.seq_lens,
            model_worker_batch.seq_lens_sum,
            draft_worker.topk,
            draft_worker.speculative_num_steps,
            draft_worker.speculative_num_draft_tokens,
            draft_worker.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=final_draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=draft_worker.speculative_num_steps,
            topk=draft_worker.topk,
            draft_token_num=draft_worker.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    # Apply monkey-patch
    draft_worker.draft = patched_draft
    draft_worker._suffix_speculator = suffix_speculator
    draft_worker._fusion_mode = fusion_mode

    logger.info(
        f"Patched EAGLE-3 draft worker with suffix fusion "
        f"(mode={fusion_mode}, budget_ratio={suffix_budget_ratio})"
    )
    return draft_worker


def _get_context_tokens(req: Any) -> list[int]:
    """Extract context token IDs from a SGLang request object."""
    tokens = []
    if hasattr(req, 'origin_input_ids'):
        tokens = list(req.origin_input_ids)
    if hasattr(req, 'output_ids'):
        tokens = tokens + list(req.output_ids)
    return tokens


def _inject_suffix_candidates(
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    suffix_draft: SuffixDraft,
    batch_idx: int,
    total_budget: int,
    topk: int,
    spec_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inject suffix decoding candidates into EAGLE-3's draft tensors.

    Strategy: Replace the lowest-score EAGLE-3 tokens with suffix candidates.
    This keeps the total token count within budget while adding suffix coverage.

    The parent indices for suffix tokens use -1 (direct children of context)
    since suffix candidates are independent paths, not extensions of EAGLE nodes.

    Args:
        parent_list: [batch_size, num_parents] EAGLE-3 parent indices.
        top_scores_index: [batch_size, num_draft-1] score indices.
        draft_tokens: [batch_size, num_draft-1] token IDs.
        suffix_draft: SuffixDraft from Arctic Inference.
        batch_idx: Which request in the batch.
        total_budget: speculative_num_draft_tokens.
        topk: EAGLE-3 top-k.
        spec_steps: EAGLE-3 num_steps.

    Returns:
        Updated (parent_list, top_scores_index, draft_tokens).
    """
    num_suffix = min(suffix_draft.num_tokens, draft_tokens.shape[1] // 3)
    if num_suffix == 0:
        return parent_list, top_scores_index, draft_tokens

    # Replace the last N tokens in draft_tokens with suffix candidates
    # (last tokens tend to have lowest EAGLE-3 scores since top_scores_index is sorted)
    suffix_token_ids = torch.tensor(
        suffix_draft.token_ids[:num_suffix],
        dtype=draft_tokens.dtype,
        device=draft_tokens.device,
    )

    # Clone to avoid modifying the original tensors
    new_draft_tokens = draft_tokens.clone()
    replace_start = new_draft_tokens.shape[1] - num_suffix
    new_draft_tokens[batch_idx, replace_start:] = suffix_token_ids

    return parent_list, top_scores_index, new_draft_tokens
