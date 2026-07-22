"""DraftTree — the verifiable draft structure the composition builds.

Reconstructed to match the contract that ``run_partialwarm_tree.py`` and
``fusion_tree.py`` rely on (Dr. Lee's spec references
``specedge_serving.common.tree.DraftTree`` but did not ship it):

  - ``.tokens``   : flat list of draft token ids, topologically ordered
                    (a node's parent always has a smaller index).
  - ``.parents``  : parent index per node (-1 == virtual root / attaches to
                    the last committed context token, i.e. ``ctx_root``).
  - ``.coords``   : optional (b, e) coordinate per node used by the tree
                    budget allocator (b = backbone/head depth the tail hangs
                    off, e = depth within the grafted tail). Not required by
                    the greedy walk; kept so analysis code can bucket nodes.

The tree is consumed by ``tree_verify_hybrid.tree_verify`` (a greedy tree
walk against the target-greedy trace) which reads ``.tokens`` / ``.parents``
exactly like ``simulation/evaluation/tree_knapsack.greedy_tree_walk``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


class DraftTree:
    def __init__(
        self,
        tokens: Sequence[int],
        parents: Sequence[int],
        coords: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        self.tokens: List[int] = [int(t) for t in tokens]
        self.parents: List[int] = [int(p) for p in parents]
        if len(self.parents) != len(self.tokens):
            raise ValueError(
                f"tokens/parents length mismatch: {len(self.tokens)} vs {len(self.parents)}"
            )
        # Sanity: topologically ordered (parent index < child index, or -1).
        for i, p in enumerate(self.parents):
            if p >= i:
                raise ValueError(
                    f"DraftTree not topologically ordered at node {i}: parent {p} >= {i}"
                )
        self.coords: List[Tuple[int, int]] = (
            [tuple(c) for c in coords] if coords is not None else [(0, 0)] * len(self.tokens)
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def depth_of(self, i: int) -> int:
        """Root-to-node depth (# edges from virtual root); root nodes -> 1."""
        d = 0
        while i >= 0:
            d += 1
            i = self.parents[i]
        return d

    def path_to(self, i: int) -> List[int]:
        """Token ids from the first real node down to node i (inclusive)."""
        out: List[int] = []
        while i >= 0:
            out.append(self.tokens[i])
            i = self.parents[i]
        out.reverse()
        return out

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DraftTree(n={len(self.tokens)}, tokens={self.tokens[:8]}{'...' if len(self.tokens) > 8 else ''})"
