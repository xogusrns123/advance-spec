"""
SuffixDecoding Speculator -- high-level interface for tree fusion.

Wraps SuffixDecodingCache and provides methods specific to
hybrid EAGLE-3 + SuffixDecoding fusion:
- speculate_for_fusion(): returns draft in a format ready for tree merging
- speculate_from_extended_context(): for sequential extension (Case 2a)
"""

from __future__ import annotations

from typing import Hashable, Sequence

from .suffix_tree import SuffixDecodingCache, SuffixDraft


class SuffixSpeculator:
    """
    High-level speculator for hybrid tree fusion.

    Wraps Arctic Inference's SuffixDecodingCache with convenience methods
    for EAGLE-3 + SuffixDecoding integration.
    """

    def __init__(
        self,
        max_tree_depth: int = 64,
        max_cached_requests: int = 100000,
        max_spec_tokens: int = 16,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = True,
    ):
        self.cache = SuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            max_spec_offset=max_spec_offset,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec,
        )

    # ---- Request lifecycle (delegate to cache) ----

    def start_request(
        self, req_id: Hashable, prompt_token_ids: Sequence[int]
    ) -> None:
        self.cache.start_request(req_id, prompt_token_ids)

    def stop_request(self, req_id: Hashable) -> None:
        self.cache.stop_request(req_id)

    def add_active_response(
        self, req_id: Hashable, token_ids: Sequence[int]
    ) -> None:
        self.cache.add_active_response(req_id, token_ids)

    # ---- Speculation ----

    def speculate(
        self,
        req_id: Hashable,
        context: Sequence[int],
        max_spec_tokens: int | None = None,
    ) -> SuffixDraft:
        """Standard speculation from current context."""
        return self.cache.speculate(
            req_id, context, max_spec_tokens=max_spec_tokens
        )

    def speculate_from_extended_context(
        self,
        req_id: Hashable,
        context: Sequence[int],
        eagle_draft_tokens: Sequence[int],
        max_spec_tokens: int | None = None,
    ) -> SuffixDraft:
        """
        Speculate using context extended by EAGLE-3 draft tokens.

        This is the key operation for sequential extension (Case 2a):
        EAGLE-3 draft tokens create new context that may enable
        suffix matches impossible with the original context alone.
        """
        extended = list(context) + list(eagle_draft_tokens)
        return self.cache.speculate(
            req_id, extended, max_spec_tokens=max_spec_tokens
        )
