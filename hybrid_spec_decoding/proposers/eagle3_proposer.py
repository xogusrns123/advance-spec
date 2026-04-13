"""
EAGLE-3 proposer -- wraps SGLang's EAGLE-3 speculative decoding.

Calls an SGLang server that runs EAGLE-3 and reconstructs the draft tree
from the server's ``/generate`` response (meta_info.spec_verify_ct and
draft token lists).  For offline / unit-test use it can also build a
synthetic tree from pre-collected draft logs.
"""

from __future__ import annotations

from typing import Any, Sequence

import requests

from ..tree_fusion.tree_utils import DraftTree
from .base import BaseProposer, ProposerOutput


class Eagle3Proposer(BaseProposer):
    """Proposer backed by an SGLang server running EAGLE-3."""

    def __init__(
        self,
        server_url: str = "http://localhost:30000",
        num_steps: int = 5,
        eagle_topk: int = 8,
        temperature: float = 0.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.num_steps = num_steps
        self.eagle_topk = eagle_topk
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "eagle3"

    def propose_tree(
        self,
        context_ids: Sequence[int],
        max_tokens: int = 64,
        **kwargs: Any,
    ) -> ProposerOutput:
        prompt_text: str | None = kwargs.get("prompt_text")
        if prompt_text is None:
            raise ValueError(
                "Eagle3Proposer.propose_tree requires prompt_text=... "
                "because SGLang accepts text, not raw token ids."
            )

        resp = requests.post(
            f"{self.server_url}/generate",
            json={
                "text": prompt_text,
                "sampling_params": {
                    "max_new_tokens": 1,
                    "temperature": self.temperature,
                },
                "return_logprob": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        meta = data.get("meta_info", {})

        # Build DraftTree from EAGLE-3 draft info when available
        tree = DraftTree()
        draft_token_ids: list[int] = meta.get("draft_token_ids", [])
        draft_parent_ids: list[int] = meta.get("draft_parent_ids", [])
        draft_probs: list[float] = meta.get("draft_probs", [0.5] * len(draft_token_ids))

        if draft_token_ids:
            self._reconstruct_tree(tree, draft_token_ids, draft_parent_ids, draft_probs)
        else:
            # Fallback: build flat chain from output tokens
            output_ids = meta.get("output_token_ids", [])
            if output_ids:
                tree.add_sequence(tree.root, output_ids, source="eagle")

        return ProposerOutput(tree=tree)

    @staticmethod
    def _reconstruct_tree(
        tree: DraftTree,
        token_ids: list[int],
        parent_ids: list[int],
        probs: list[float],
    ) -> None:
        """Rebuild a DraftTree from flat (token, parent_index) arrays."""
        idx_to_node = {-1: tree.root}
        for i, (tid, pid, p) in enumerate(zip(token_ids, parent_ids, probs)):
            parent_node = idx_to_node.get(pid, tree.root)
            node = tree.add_node(parent_node, tid, prob=p, source="eagle")
            idx_to_node[i] = node


class Eagle3OfflineProposer(BaseProposer):
    """Build EAGLE-3-style trees from pre-collected draft logs (offline)."""

    def __init__(self, draft_log: list[dict[str, Any]] | None = None):
        self._log = draft_log or []
        self._cursor = 0

    @property
    def name(self) -> str:
        return "eagle3_offline"

    def propose_tree(
        self,
        context_ids: Sequence[int],
        max_tokens: int = 64,
        **kwargs: Any,
    ) -> ProposerOutput:
        tree = DraftTree()
        if self._cursor < len(self._log):
            entry = self._log[self._cursor]
            self._cursor += 1
            token_ids = entry.get("draft_token_ids", [])
            parent_ids = entry.get("draft_parent_ids", [-1] * len(token_ids))
            probs = entry.get("draft_probs", [0.5] * len(token_ids))
            Eagle3Proposer._reconstruct_tree(tree, token_ids, parent_ids, probs)
        return ProposerOutput(tree=tree)
