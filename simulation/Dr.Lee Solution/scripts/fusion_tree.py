"""Composition builders — chain (gated/linear) and tree (ungated), plus the
``adaptive_nhead`` forwarder (controller.md §5, §4).

Both builders return a ``DraftTree`` (``.tokens`` / ``.parents``), the object the
round loop advances and ``tree_verify_hybrid.tree_verify`` walks.

- ``build_extension_chain(head, tail)`` — one linear path: head tokens then the
  tail appended after the last head token. GATED: the tail is only reachable if
  every head token is accepted (the survival gate S_k). Also used for the
  standalones: ``build_extension_chain(block[:B], [])`` = DFlash-only,
  ``build_extension_chain([], suf[:B])`` = Suffix-only.
- ``build_extension_tree(head, tails)`` — UNGATED: ``head`` is the linear
  backbone, and ``tails[j]`` (a flat copy-tail) is grafted after head node j-1
  (``tails[0]`` at the virtual root). If the head breaks at depth j, the tail
  hanging at prefix j is still on a live path → the gate is removed.
"""
from __future__ import annotations

import os
import sys
from typing import List, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from specedge_serving.common.tree import DraftTree  # noqa: E402

from ext_suffix import adaptive_nhead as _adaptive_nhead_impl  # noqa: E402


def build_extension_chain(head: Sequence[int], tail: Sequence[int]) -> DraftTree:
    """head → tail as a single linear chain (parents = [-1,0,1,2,…])."""
    tokens: List[int] = [int(t) for t in head] + [int(t) for t in tail]
    parents: List[int] = [i - 1 for i in range(len(tokens))]  # 0 -> -1
    hlen = len(head)
    coords = [(i + 1, 0) for i in range(hlen)] + [
        (hlen, e + 1) for e in range(len(tokens) - hlen)
    ]
    return DraftTree(tokens, parents, coords)


def build_extension_tree(head: Sequence[int], tails: Sequence[Sequence[int]]) -> DraftTree:
    """Ungated tree: linear head backbone + a flat tail grafted at every prefix.

    ``tails`` has len(head)+1 entries; ``tails[j]`` grafts after head node j-1
    (j==0 → virtual root -1). Topological order is preserved (head backbone
    first, then tails appended in prefix order), so every parent index is < its
    child index — required by DraftTree / the greedy walk.
    """
    head = [int(t) for t in head]
    k = len(head)
    # Head backbone.
    tokens: List[int] = list(head)
    parents: List[int] = [i - 1 for i in range(k)]           # 0 -> -1
    coords: List[tuple] = [(i + 1, 0) for i in range(k)]
    # Grafts: tails[j] hangs off head node (j-1); tails[0] off the virtual root.
    for j in range(min(len(tails), k + 1)):
        anchor = j - 1                                        # head node index (or -1)
        prev = anchor
        for e, tok in enumerate(tails[j]):
            idx = len(tokens)
            tokens.append(int(tok))
            parents.append(prev)
            coords.append((j, e + 1))
            prev = idx
    return DraftTree(tokens, parents, coords)


def adaptive_nhead(conf, T, num_spec):
    """Forwards to ext_suffix (single source; controller.md §4)."""
    return _adaptive_nhead_impl(conf, T=T, num_spec=num_spec)
