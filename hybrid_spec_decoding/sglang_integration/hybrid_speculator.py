"""
Hybrid Speculator: orchestrates EAGLE-3 + SuffixDecoding on SGLang.

Provides:
1. Patching SGLang's MultiLayerEagleWorkerV2 with suffix fusion
2. Managing SuffixSpeculator lifecycle (start/stop/update per request)
3. Hooking into SGLang's request lifecycle for suffix tree updates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..suffix_decoding.speculator import SuffixSpeculator
from .patched_eagle_worker import patch_eagle_worker

__all__ = ["ExperimentConfig", "HybridConfig", "HybridSpeculator"]

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Reproducible experiment config used by benchmark scripts."""

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.0

    # Tree budget
    max_tree_tokens: int = 64
    num_draft_tokens: int = 64

    # EAGLE-3 specific
    num_steps: int = 5
    eagle_topk: int = 8

    # Suffix specific
    suffix_match_len: int = 16
    max_candidates: int = 10

    # Fusion
    fusion_mode: str = "parallel"
    pruning_topk: int = 10
    suffix_budget_ratio: float = 0.3

    # Reproducibility
    seed: int = 42


@dataclass
class HybridConfig:
    """Configuration for hybrid speculative decoding."""

    # Suffix tree params
    max_tree_depth: int = 64
    max_cached_requests: int = 100000
    max_spec_tokens: int = 16
    max_spec_factor: float = 1.0
    max_spec_offset: float = 0.0
    min_token_prob: float = 0.1
    use_tree_spec: bool = True

    # Fusion params
    fusion_mode: str = "parallel"        # parallel, sequential, combined
    suffix_budget_ratio: float = 0.3     # fraction of draft budget for suffix
    extension_depths: list[int] = field(default_factory=lambda: [1, 2, 3])
    min_extension_confidence: float = 0.3


class HybridSpeculator:
    """
    Manages the lifecycle of hybrid EAGLE-3 + SuffixDecoding.

    Usage:
        config = HybridConfig(fusion_mode="parallel")
        hybrid = HybridSpeculator(config)
        hybrid.patch(sglang_spec_worker)

        # Per-request lifecycle (called from scheduler hooks):
        hybrid.on_request_start(req_id, prompt_tokens)
        hybrid.on_tokens_generated(req_id, new_tokens)
        hybrid.on_request_end(req_id)
    """

    def __init__(self, config: HybridConfig):
        self.config = config
        self.speculator = SuffixSpeculator(
            max_tree_depth=config.max_tree_depth,
            max_cached_requests=config.max_cached_requests,
            max_spec_tokens=config.max_spec_tokens,
            max_spec_factor=config.max_spec_factor,
            max_spec_offset=config.max_spec_offset,
            min_token_prob=config.min_token_prob,
            use_tree_spec=config.use_tree_spec,
        )
        self._patched = False

    def patch(self, spec_worker: Any) -> None:
        """
        Patch SGLang's MultiLayerEagleWorkerV2 with suffix fusion.

        Args:
            spec_worker: The SGLang speculative worker
                (MultiLayerEagleWorkerV2 instance).
        """
        draft_worker = spec_worker.draft_worker
        patch_eagle_worker(
            draft_worker,
            self.speculator,
            fusion_mode=self.config.fusion_mode,
            suffix_budget_ratio=self.config.suffix_budget_ratio,
            extension_depths=self.config.extension_depths,
            min_extension_confidence=self.config.min_extension_confidence,
        )
        self._patched = True
        logger.info("HybridSpeculator: patched SGLang spec worker")

    # ---- Request lifecycle hooks ----

    def on_request_start(self, req_id: str, prompt_tokens: list[int]) -> None:
        """Call when a new request arrives. Builds local suffix tree from prompt."""
        self.speculator.start_request(req_id, prompt_tokens)

    def on_tokens_generated(self, req_id: str, new_tokens: list[int]) -> None:
        """Call after each decoding step. Updates both local and global trees."""
        self.speculator.add_active_response(req_id, new_tokens)

    def on_request_end(self, req_id: str) -> None:
        """Call when request completes. Cleans up local tree."""
        self.speculator.stop_request(req_id)

    # ---- Convenience ----

    @staticmethod
    def get_sglang_launch_cmd(
        target_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        draft_model: str = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        num_steps: int = 5,
        eagle_topk: int = 8,
        num_draft_tokens: int = 64,
    ) -> str:
        """Generate the SGLang server launch command."""
        return (
            f"python3 -m sglang.launch_server "
            f"--model {target_model} "
            f"--speculative-algorithm EAGLE3 "
            f"--speculative-draft-model-path {draft_model} "
            f"--speculative-num-steps {num_steps} "
            f"--speculative-eagle-topk {eagle_topk} "
            f"--speculative-num-draft-tokens {num_draft_tokens}"
        )
