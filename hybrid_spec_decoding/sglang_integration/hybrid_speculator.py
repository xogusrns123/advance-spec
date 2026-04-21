"""
Hybrid Speculator: manages SuffixDecoding lifecycle.

Provides:
1. Suffix tree warming from corpus data
2. Request lifecycle management (start/stop/update)
3. SGLang launch command generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from ..suffix_decoding.speculator import SuffixSpeculator

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
    num_draft_tokens: int = 16

    # Suffix specific
    suffix_match_len: int = 16
    max_candidates: int = 10

    # Reproducibility
    seed: int = 42


@dataclass
class HybridConfig:
    """Configuration for suffix speculative decoding."""

    # Suffix tree params
    max_tree_depth: int = 64
    max_cached_requests: int = 100000
    max_spec_tokens: int = 16
    max_spec_factor: float = 1.0
    max_spec_offset: float = 0.0
    min_token_prob: float = 0.1
    use_tree_spec: bool = True


class HybridSpeculator:
    """
    Manages the lifecycle of SuffixDecoding.

    Usage:
        config = HybridConfig()
        hybrid = HybridSpeculator(config)

        # Per-request lifecycle:
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

    # ---- Warming ----

    def warm_from_corpus(self, token_sequences: Sequence[Sequence[int]]) -> None:
        """
        Pre-populate the global suffix tree with tokenized sequences.

        Call before serving to give the suffix tree initial data.
        Each sequence is treated as a completed request: a minimal prompt
        is registered, the rest is added as response, then the request
        is stopped (promoting data to the global tree).

        Args:
            token_sequences: List of tokenized text sequences.
        """
        for i, seq in enumerate(token_sequences):
            if len(seq) < 2:
                continue
            req_id = f"__warmup_{i}"
            self.speculator.start_request(req_id, seq[:1])
            self.speculator.add_active_response(req_id, seq[1:])
            self.speculator.stop_request(req_id)
        logger.info(f"Warmed suffix tree with {len(token_sequences)} sequences")

    # ---- Convenience ----

    @staticmethod
    def get_sglang_launch_cmd(
        target_model: str = "zai-org/GLM-4.7-Flash",
        tp_size: int = 4,
        num_draft_tokens: int = 16,
        mem_fraction_static: float = 0.8,
        port: int = 30000,
    ) -> str:
        """Generate the SGLang server launch command for suffix decoding."""
        return (
            f"python3 -m simulation.oracle.install_hook -- "
            f"--model-path {target_model} "
            f"--tp-size {tp_size} "
            f"--speculative-algorithm SUFFIX "
            f"--speculative-num-draft-tokens {num_draft_tokens} "
            f"--mem-fraction-static {mem_fraction_static} "
            f"--disable-cuda-graph "
            f"--host 0.0.0.0 --port {port}"
        )
