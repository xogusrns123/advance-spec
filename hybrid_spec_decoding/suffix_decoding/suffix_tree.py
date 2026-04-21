"""
SuffixDecoding via Arctic Inference.

Directly uses Snowflake's arctic-inference C++ SuffixTree
(path-compressed, count-ordered, O(N) memory) instead of
a Python reimplementation. Provides the same SuffixDecodingCache
with global + per-request dual tree architecture.

Install: uv pip install arctic-inference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Optional, Sequence

from arctic_inference.suffix_decoding import SuffixDecodingCache as _NativeCache
from arctic_inference.suffix_decoding.cache import SuffixDecodingDraft


@dataclass
class SuffixDraft:
    """Draft result from suffix speculation.

    Mirrors arctic_inference's SuffixDecodingDraft but adds helper methods
    for tree fusion integration.
    """

    token_ids: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)   # -1 = child of context
    probs: list[float] = field(default_factory=list)
    score: float = 0.0
    match_len: int = 0

    @staticmethod
    def from_native(draft: SuffixDecodingDraft) -> SuffixDraft:
        return SuffixDraft(
            token_ids=list(draft.token_ids),
            parents=list(draft.parents),
            probs=list(draft.probs),
            score=draft.score,
            match_len=draft.match_len,
        )

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_empty(self) -> bool:
        return len(self.token_ids) == 0


class SuffixDecodingCache:
    """
    Thin wrapper around Arctic Inference's SuffixDecodingCache.

    Manages dual suffix trees (global + per-request local) with
    C++ path-compressed suffix tree and count-based ordering.

    Lifecycle per request:
        start_request(req_id, prompt_tokens)
        add_active_response(req_id, new_tokens)   # called each step
        speculate(req_id, context) -> SuffixDraft
        stop_request(req_id)
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
        self._cache = _NativeCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
        )
        self.max_tree_depth = max_tree_depth
        self.max_spec_tokens = max_spec_tokens
        self.max_spec_factor = max_spec_factor
        self.max_spec_offset = max_spec_offset
        self.min_token_prob = min_token_prob
        self.use_tree_spec = use_tree_spec

    def start_request(
        self,
        req_id: Hashable,
        prompt_token_ids: Sequence[int],
    ) -> None:
        """Initialize local tree for a new request with prompt tokens."""
        self._cache.start_request(req_id, prompt_token_ids)

    def stop_request(self, req_id: Hashable) -> None:
        """Clean up local tree. Global tree retains the response."""
        self._cache.stop_request(req_id)

    def add_active_response(
        self,
        req_id: Hashable,
        token_ids: Sequence[int],
    ) -> None:
        """Append newly generated tokens to both local and global trees."""
        self._cache.add_active_response(req_id, token_ids)

    def speculate(
        self,
        req_id: Hashable,
        context: Sequence[int],
        max_spec_tokens: int | None = None,
        max_spec_factor: float | None = None,
        max_spec_offset: float | None = None,
        min_token_prob: float | None = None,
        use_tree_spec: bool | None = None,
    ) -> SuffixDraft:
        """
        Generate draft candidates by matching context against both trees.

        Returns whichever tree (global or local) produces a higher-score draft.
        """
        native_draft = self._cache.speculate(
            req_id,
            context[-self.max_tree_depth:],
            max_spec_tokens=max_spec_tokens if max_spec_tokens is not None else self.max_spec_tokens,
            max_spec_factor=max_spec_factor if max_spec_factor is not None else self.max_spec_factor,
            max_spec_offset=max_spec_offset if max_spec_offset is not None else self.max_spec_offset,
            min_token_prob=min_token_prob if min_token_prob is not None else self.min_token_prob,
            use_tree_spec=use_tree_spec if use_tree_spec is not None else self.use_tree_spec,
        )
        return SuffixDraft.from_native(native_draft)

    def evict_cached_response(self, req_id: Hashable) -> None:
        """Remove a cached response from the global tree."""
        self._cache.evict_cached_response(req_id)
