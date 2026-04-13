"""
Suffix Decoding proposer -- wraps the existing SuffixSpeculator.

Model-free proposer that matches the current context against a
path-compressed suffix tree (global + per-request local trees via
Arctic Inference).  Produces a DraftTree from the matched candidates.
"""

from __future__ import annotations

from typing import Any, Hashable, Sequence

from ..suffix_decoding.speculator import SuffixSpeculator
from ..suffix_decoding.suffix_tree import SuffixDraft
from ..tree_fusion.tree_utils import DraftTree
from .base import BaseProposer, ProposerOutput


class SuffixProposer(BaseProposer):
    """Proposer backed by SuffixDecoding (Arctic Inference C++ trees)."""

    def __init__(
        self,
        max_tree_depth: int = 64,
        max_cached_requests: int = 100_000,
        max_spec_tokens: int = 16,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = True,
    ):
        self.speculator = SuffixSpeculator(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
            max_spec_tokens=max_spec_tokens,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec,
        )

    @property
    def name(self) -> str:
        return "suffix"

    # ---- request lifecycle (proxy to speculator) ----

    def start_request(self, req_id: Hashable, prompt_ids: Sequence[int]) -> None:
        self.speculator.start_request(req_id, prompt_ids)

    def stop_request(self, req_id: Hashable) -> None:
        self.speculator.stop_request(req_id)

    def add_tokens(self, req_id: Hashable, token_ids: Sequence[int]) -> None:
        self.speculator.add_active_response(req_id, token_ids)

    # ---- core propose ----

    def propose_tree(
        self,
        context_ids: Sequence[int],
        max_tokens: int = 64,
        **kwargs: Any,
    ) -> ProposerOutput:
        req_id: Hashable = kwargs.get("req_id", "__default__")
        draft: SuffixDraft = self.speculator.speculate(
            req_id, context_ids, max_spec_tokens=max_tokens,
        )
        tree = self._draft_to_tree(draft)
        return ProposerOutput(tree=tree)

    @staticmethod
    def _draft_to_tree(draft: SuffixDraft) -> DraftTree:
        """Convert a SuffixDraft (flat token+parent arrays) into a DraftTree."""
        tree = DraftTree()
        if draft.is_empty:
            return tree

        idx_to_node = {-1: tree.root}
        for i, (tid, pid, prob) in enumerate(
            zip(draft.token_ids, draft.parents, draft.probs)
        ):
            parent_node = idx_to_node.get(pid, tree.root)
            node = tree.add_node(
                parent_node, tid, prob=0.0, suffix_score=prob, source="suffix"
            )
            idx_to_node[i] = node
        return tree
