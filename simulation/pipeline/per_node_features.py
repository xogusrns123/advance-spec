"""Per-node feature helpers for the build_per_node_dataset pipeline.

All functions are pure: they take tree arrays (token_ids, parents) plus
per-node payloads and return per-node feature arrays. The builder
(``build_per_node_dataset.py``) composes these.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


def compute_depths(parents: Sequence[int]) -> List[int]:
    """Depth of each node (root children have depth=1, parent=-1 sentinel)."""
    depths: List[int] = []
    for i, p in enumerate(parents):
        if p < 0:
            depths.append(1)
        else:
            depths.append(depths[p] + 1)
    return depths


def compute_children(parents: Sequence[int]) -> Dict[int, List[int]]:
    """Map parent_idx -> ordered list of child indices. Root's children
    live under key -1."""
    out: Dict[int, List[int]] = {}
    for i, p in enumerate(parents):
        out.setdefault(p, []).append(i)
    return out


def compute_n_descendants(parents: Sequence[int]) -> List[int]:
    """Number of descendants (not including self) for each node."""
    children = compute_children(parents)
    n = len(parents)
    # Iterative post-order via topological sort by depth.
    depths = compute_depths(parents)
    order = sorted(range(n), key=lambda i: -depths[i])  # deepest first
    desc = [0] * n
    for i in order:
        p = parents[i]
        if p >= 0:
            desc[p] += desc[i] + 1
    return desc


def compute_cond_probs_from_path(
    parents: Sequence[int],
    path_probs: Sequence[float],
) -> List[Optional[float]]:
    """Per-node conditional probability = path_probs[i] / path_probs[parent].

    For root-children (parent=-1) the conditional equals path_probs[i] itself
    (path_prob already encodes p(child|root)).

    Returns None for any node where the parent's prob is zero/missing.
    """
    out: List[Optional[float]] = []
    for i, p in enumerate(parents):
        pp = path_probs[i]
        if p < 0:
            out.append(float(pp))
        else:
            parent_p = path_probs[p]
            if parent_p is None or parent_p <= 0:
                out.append(None)
            else:
                out.append(float(pp) / float(parent_p))
    return out


def compute_sibling_rank(
    parents: Sequence[int],
    score: Sequence[Optional[float]],
) -> List[Optional[int]]:
    """Rank among same-parent siblings by ``score`` descending (0 = best).

    Nodes with None score get rank=None.
    """
    children = compute_children(parents)
    out: List[Optional[int]] = [None] * len(parents)
    for p, kids in children.items():
        # Stable sort by score desc (None pushed to end).
        sorted_kids = sorted(
            kids,
            key=lambda i: (-(score[i] if score[i] is not None else -1.0), i),
        )
        for rk, i in enumerate(sorted_kids):
            if score[i] is None:
                out[i] = None
            else:
                out[i] = rk
    return out


def compute_top1_prob_at_parent(
    parents: Sequence[int],
    cond_probs: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """For each node, max cond_prob among its parent's children (= top-1
    among siblings). Returns None when the parent has no children with a
    valid cond_prob."""
    children = compute_children(parents)
    parent_top1: Dict[int, Optional[float]] = {}
    for p, kids in children.items():
        best: Optional[float] = None
        for k in kids:
            cp = cond_probs[k]
            if cp is None:
                continue
            if best is None or cp > best:
                best = cp
        parent_top1[p] = best
    return [parent_top1.get(p) for p in parents]


def compute_entropy_at_parent(
    parents: Sequence[int],
    cond_probs: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """For each node, entropy of the cond-prob distribution over its
    parent's children (top-k truncated; siblings probs do not sum to 1
    in general)."""
    children = compute_children(parents)
    parent_ent: Dict[int, Optional[float]] = {}
    for p, kids in children.items():
        ps: List[float] = []
        for k in kids:
            cp = cond_probs[k]
            if cp is None or cp <= 0:
                continue
            ps.append(cp)
        if not ps:
            parent_ent[p] = None
            continue
        # Plain entropy on raw probs; document the top-k truncation in builder.
        ent = -sum(p * math.log(p) for p in ps)
        parent_ent[p] = ent
    return [parent_ent.get(p) for p in parents]


def greedy_accepted_path(
    token_ids: Sequence[int],
    parents: Sequence[int],
    gt_future: Sequence[int],
) -> List[int]:
    """Return the list of node indices on the greedy accepted path.

    Walk from the virtual root (parent=-1). At each step, pick the first
    child whose token matches gt_future[depth]. Stop on mismatch or when
    no child available.
    """
    if not gt_future:
        return []
    children = compute_children(parents)
    out: List[int] = []
    cur = -1
    depth = 0
    while depth < len(gt_future):
        kids = children.get(cur, [])
        gt = gt_future[depth]
        picked: Optional[int] = None
        for k in kids:
            if token_ids[k] == gt:
                picked = k
                break
        if picked is None:
            break
        out.append(picked)
        cur = picked
        depth += 1
    return out


def latency_lookup_target_ms(
    total_nodes: int,
    target_forward_ms: Dict[int, float],
) -> Optional[float]:
    """Lookup target verify cost (ms) for given tree size.

    Uses next-power-of-2 >= total_nodes. If total_nodes exceeds the
    largest calibrated entry, returns the largest entry's value (held
    flat; downstream can re-extrapolate from total_nodes if needed).
    """
    if not target_forward_ms:
        return None
    # next pow2 >= total_nodes (min 1)
    n = max(1, int(total_nodes))
    b = 1
    while b < n:
        b <<= 1
    keys = sorted(target_forward_ms.keys())
    if b in target_forward_ms:
        return float(target_forward_ms[b])
    # find smallest key >= b
    for k in keys:
        if k >= b:
            return float(target_forward_ms[k])
    # b exceeds all keys — return largest
    return float(target_forward_ms[keys[-1]])
