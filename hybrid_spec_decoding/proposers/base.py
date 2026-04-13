"""
Abstract base class for all speculative decoding proposers.

Every proposer -- MTP, small draft model, EAGLE-3, suffix decoding --
implements this interface so that the tree builder, tracer, and benchmark
harness can treat them uniformly.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..tree_fusion.tree_utils import DraftTree


@dataclass
class ProposerOutput:
    """Unified output from any proposer.

    All fields map directly to the shared DraftTree so that downstream
    code never needs to know which proposer produced the tokens.
    """

    tree: DraftTree
    # Per-node metadata (BFS order matching tree.flatten())
    token_ids: list[int] = field(default_factory=list)
    parent_ids: list[int] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)
    local_probs: list[float] = field(default_factory=list)
    local_logprobs: list[float] = field(default_factory=list)
    cumulative_logprobs: list[float] = field(default_factory=list)
    # Timing
    draft_latency_s: float = 0.0
    # Metadata
    proposer_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ----- helpers -----

    def to_dict(self) -> dict[str, Any]:
        tokens, parents = self.tree.flatten()
        return {
            "proposer": self.proposer_name,
            "num_nodes": self.tree.num_nodes,
            "token_ids": self.token_ids,
            "parent_ids": self.parent_ids,
            "depths": self.depths,
            "local_probs": self.local_probs,
            "local_logprobs": self.local_logprobs,
            "cumulative_logprobs": self.cumulative_logprobs,
            "draft_latency_s": self.draft_latency_s,
            "extra": self.extra,
        }


def _safe_log(p: float) -> float:
    return math.log(max(p, 1e-30))


def populate_output_metadata(output: ProposerOutput) -> ProposerOutput:
    """Fill token_ids, parent_ids, depths, logprobs from the tree."""
    tree = output.tree
    nodes = tree.get_all_nodes()
    nodes.sort(key=lambda n: (n.depth, n.node_id))

    id_to_idx: dict[int, int] = {}
    token_ids: list[int] = []
    parent_ids: list[int] = []
    depths: list[int] = []
    local_probs: list[float] = []
    local_logprobs: list[float] = []
    cumulative_logprobs: list[float] = []

    # node_id -> cumulative logprob
    cum_map: dict[int, float] = {}

    for i, node in enumerate(nodes):
        id_to_idx[node.node_id] = i
        token_ids.append(node.token_id)
        depths.append(node.depth)

        if node.parent is None or node.parent.token_id < 0:
            parent_ids.append(-1)
        else:
            parent_ids.append(id_to_idx[node.parent.node_id])

        p = node.prob if node.prob > 0 else node.suffix_score
        local_probs.append(p)
        lp = _safe_log(p)
        local_logprobs.append(lp)

        parent_cum = cum_map.get(
            node.parent.node_id if node.parent else -1, 0.0
        )
        cum = parent_cum + lp
        cum_map[node.node_id] = cum
        cumulative_logprobs.append(cum)

    output.token_ids = token_ids
    output.parent_ids = parent_ids
    output.depths = depths
    output.local_probs = local_probs
    output.local_logprobs = local_logprobs
    output.cumulative_logprobs = cumulative_logprobs
    return output


class BaseProposer(ABC):
    """Abstract proposer that produces a DraftTree from a token context."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'eagle3', 'mtp', 'draft_model', 'suffix'."""

    @abstractmethod
    def propose_tree(
        self,
        context_ids: Sequence[int],
        max_tokens: int = 64,
        **kwargs: Any,
    ) -> ProposerOutput:
        """Generate a draft tree for the given context.

        Args:
            context_ids: Token IDs of the verified prefix.
            max_tokens: Maximum number of draft tokens (tree budget).
            **kwargs: Proposer-specific overrides.

        Returns:
            ProposerOutput with a populated DraftTree plus metadata.
        """

    def propose(
        self,
        context_ids: Sequence[int],
        max_tokens: int = 64,
        **kwargs: Any,
    ) -> ProposerOutput:
        """Convenience wrapper that times the call and fills metadata."""
        t0 = time.perf_counter()
        output = self.propose_tree(context_ids, max_tokens, **kwargs)
        output.draft_latency_s = time.perf_counter() - t0
        output.proposer_name = self.name
        populate_output_metadata(output)
        return output
