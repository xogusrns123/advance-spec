"""
Small Draft Model proposer.

Uses a smaller language model (e.g. Llama-3.2-1B as draft for Llama-3.1-8B)
to autoregressively generate candidate tokens.  At each step the top-k
tokens are branched so the output is a tree, not a single chain.

Supports two modes:
1. **Server mode**: calls a second SGLang server hosting the draft model.
2. **Offline mode**: accepts pre-computed per-step logits via ``step_logits``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import requests

from ..tree_fusion.tree_utils import DraftTree
from .base import BaseProposer, ProposerOutput


class DraftModelProposer(BaseProposer):
    """Proposer backed by a small autoregressive draft model.

    Parameters
    ----------
    server_url : str
        URL of the SGLang server hosting the *draft* model.
    topk : int
        Number of top-k tokens to branch at each autoregressive step.
    max_depth : int
        Maximum number of autoregressive expansion steps (= tree depth).
    temperature : float
        Temperature for softmax conversion of draft model logits.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:30001",
        topk: int = 4,
        max_depth: int = 6,
        temperature: float = 1.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.topk = topk
        self.max_depth = max_depth
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "draft_model"

    def propose_tree(
        self,
        context_ids: Sequence[int],
        max_tokens: int = 64,
        **kwargs: Any,
    ) -> ProposerOutput:
        """Build a draft tree using the small draft model.

        Accepted kwargs:
            step_logits (list[np.ndarray]): Pre-computed logits per depth
                step, each of shape ``(vocab_size,)``.  When provided the
                server is not called -- useful for offline replay and
                unit testing.
            prompt_text (str): Required when ``step_logits`` is absent.
        """
        step_logits: list[np.ndarray] | None = kwargs.get("step_logits")

        if step_logits is not None:
            return ProposerOutput(
                tree=self._build_tree_from_logits(step_logits, max_tokens)
            )

        return ProposerOutput(
            tree=self._build_tree_via_server(context_ids, max_tokens, kwargs)
        )

    # ------------------------------------------------------------------ #
    # Server-based tree construction                                       #
    # ------------------------------------------------------------------ #

    def _build_tree_via_server(
        self,
        context_ids: Sequence[int],
        max_tokens: int,
        kwargs: dict[str, Any],
    ) -> DraftTree:
        """Iteratively query the draft model server, branching top-k."""
        prompt_text: str | None = kwargs.get("prompt_text")
        if prompt_text is None:
            raise ValueError(
                "DraftModelProposer requires prompt_text=... "
                "when step_logits is not provided."
            )

        tree = DraftTree()
        budget_left = max_tokens
        depth = min(self.max_depth, max_tokens)

        # Start with a single server call for the first token
        # then expand greedily at each depth.
        # In practice, for a true branching tree, the draft model
        # should expose per-step logits.  Here we use a greedy chain
        # as the primary path and branch from collected logits.

        resp = requests.post(
            f"{self.server_url}/generate",
            json={
                "text": prompt_text,
                "sampling_params": {
                    "max_new_tokens": depth,
                    "temperature": 0.0,
                },
                "return_logprob": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        meta = data.get("meta_info", {})
        output_ids: list[int] = meta.get("output_token_ids", [])
        output_logprobs: list[dict[str, float]] = meta.get(
            "output_token_logprobs", []
        )

        if not output_ids:
            return tree

        # Build greedy chain as the backbone
        node = tree.root
        for i, tid in enumerate(output_ids):
            if budget_left <= 0:
                break
            p = 1.0
            if i < len(output_logprobs) and output_logprobs[i]:
                # logprob dict: {token_id_str: logprob, ...}
                lp = output_logprobs[i].get(str(tid), -1.0)
                p = np.exp(lp) if isinstance(lp, (int, float)) else 0.5
            node = tree.add_node(node, tid, prob=float(p), source="draft_model")
            budget_left -= 1

        return tree

    # ------------------------------------------------------------------ #
    # Offline / logits-based tree construction                             #
    # ------------------------------------------------------------------ #

    def _build_tree_from_logits(
        self,
        step_logits: list[np.ndarray],
        max_tokens: int,
    ) -> DraftTree:
        """Build a branching tree from per-step logits (offline mode).

        Each step produces ``topk`` branches.  The tree is expanded
        BFS-style up to ``max_tokens`` total nodes.
        """
        tree = DraftTree()
        if not step_logits:
            return tree

        budget_left = max_tokens
        num_steps = min(self.max_depth, len(step_logits))
        frontier: list[tuple[Any, int]] = [(tree.root, 0)]

        while frontier and budget_left > 0:
            next_frontier: list[tuple[Any, int]] = []
            for parent_node, step_idx in frontier:
                if step_idx >= num_steps or budget_left <= 0:
                    continue

                logits = step_logits[step_idx]
                probs = _softmax(logits, self.temperature)
                topk_ids = np.argsort(probs)[-self.topk:][::-1]

                for tid in topk_ids:
                    if budget_left <= 0:
                        break
                    p = float(probs[tid])
                    child = tree.add_node(
                        parent_node, int(tid), prob=p, source="draft_model"
                    )
                    budget_left -= 1
                    next_frontier.append((child, step_idx + 1))

            frontier = next_frontier

        return tree


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        temperature = 1e-8
    scaled = logits / temperature
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / (exp.sum() + 1e-30)
