"""
Multi-Token Prediction (MTP) proposer.

Uses dedicated MTP heads (as in DeepSeek-V3, Meta MTP) that predict
multiple future tokens in a single forward pass.  Each head h_k predicts
the token at position t+k given the hidden states of the target model.

When an SGLang server exposes MTP logits (e.g. via ``/generate`` with
``return_mtp_logits=True``), this proposer builds a tree by taking the
top-k tokens from each head and chaining them.

For offline / unit-test use, the proposer can also accept raw logits
directly (``raw_logits`` kwarg).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import requests

from ..tree_fusion.tree_utils import DraftTree
from .base import BaseProposer, ProposerOutput


class MTPProposer(BaseProposer):
    """Proposer backed by Multi-Token Prediction heads.

    Parameters
    ----------
    server_url : str
        SGLang server URL. Only used when ``raw_logits`` is not passed
        to :meth:`propose_tree`.
    num_heads : int
        Number of MTP heads (= max lookahead depth).
    topk_per_head : int
        How many top-k tokens to branch at each depth.
    temperature : float
        Temperature for softmax when converting logits to probabilities.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:30000",
        num_heads: int = 4,
        topk_per_head: int = 4,
        temperature: float = 1.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.num_heads = num_heads
        self.topk_per_head = topk_per_head
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "mtp"

    def propose_tree(
        self,
        context_ids: Sequence[int],
        max_tokens: int = 64,
        **kwargs: Any,
    ) -> ProposerOutput:
        """Build a draft tree from MTP head logits.

        Accepted kwargs:
            raw_logits (list[np.ndarray]): Pre-computed logits per head,
                each of shape ``(vocab_size,)``. When provided the server
                is not called.
            prompt_text (str): Required when ``raw_logits`` is absent.
        """
        raw_logits: list[np.ndarray] | None = kwargs.get("raw_logits")

        if raw_logits is None:
            raw_logits = self._fetch_logits_from_server(kwargs)

        tree = self._build_tree(raw_logits, max_tokens)
        return ProposerOutput(tree=tree)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _fetch_logits_from_server(self, kwargs: dict[str, Any]) -> list[np.ndarray]:
        """Call SGLang /generate and extract MTP head logits."""
        prompt_text: str | None = kwargs.get("prompt_text")
        if prompt_text is None:
            raise ValueError(
                "MTPProposer requires either raw_logits=... or prompt_text=..."
            )

        resp = requests.post(
            f"{self.server_url}/generate",
            json={
                "text": prompt_text,
                "sampling_params": {
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                },
                "return_logprob": True,
            },
        )
        resp.raise_for_status()
        meta = resp.json().get("meta_info", {})

        logits_list = meta.get("mtp_logits", [])
        if not logits_list:
            # Fallback: single next-token logits replicated
            next_logits = meta.get("next_token_logits", [])
            if next_logits:
                logits_list = [next_logits] * self.num_heads
            else:
                logits_list = []

        return [np.asarray(l, dtype=np.float32) for l in logits_list]

    def _build_tree(
        self,
        logits_per_head: list[np.ndarray],
        max_tokens: int,
    ) -> DraftTree:
        """Construct a DraftTree by branching top-k at each MTP head depth.

        The tree has at most ``topk_per_head ** num_heads`` leaves, capped
        by ``max_tokens``.
        """
        tree = DraftTree()
        if not logits_per_head:
            return tree

        num_heads = min(self.num_heads, len(logits_per_head))
        budget_left = max_tokens

        # BFS expansion: frontier is a list of (tree_node, depth_index)
        frontier: list[tuple[Any, int]] = [(tree.root, 0)]

        while frontier and budget_left > 0:
            next_frontier: list[tuple[Any, int]] = []
            for parent_node, head_idx in frontier:
                if head_idx >= num_heads or budget_left <= 0:
                    continue

                logits = logits_per_head[head_idx]
                probs = _softmax(logits, self.temperature)
                topk_ids = np.argsort(probs)[-self.topk_per_head:][::-1]

                for tid in topk_ids:
                    if budget_left <= 0:
                        break
                    p = float(probs[tid])
                    child = tree.add_node(
                        parent_node,
                        int(tid),
                        prob=p,
                        source="mtp",
                    )
                    budget_left -= 1
                    next_frontier.append((child, head_idx + 1))

            frontier = next_frontier

        return tree


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    if temperature <= 0:
        temperature = 1e-8
    scaled = logits / temperature
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / (exp.sum() + 1e-30)
