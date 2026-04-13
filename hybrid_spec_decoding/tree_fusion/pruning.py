"""
Confidence-based pruning for retrieval/suffix trees.

Implements RASD-style pruning: filter out SuffixDecoding candidates
whose first token is not in EAGLE-3's top-k probability distribution.
This prevents low-confidence suffix branches from wasting the draft
token budget.
"""

from __future__ import annotations

import numpy as np

from .tree_utils import DraftTree, TreeNode


def prune_retrieval_tree(
    retrieval_tree: DraftTree,
    eagle_first_token_probs: np.ndarray,
    topk: int = 10,
) -> DraftTree:
    """
    Prune the retrieval tree based on EAGLE-3's first-token probability.

    RASD pruning strategy: remove any branch in the retrieval tree whose
    first token (depth=1) is not in the top-k of EAGLE-3's predicted
    distribution. This ensures only plausible continuations consume the
    draft budget.

    Args:
        retrieval_tree: Tree built from SuffixDecoding candidates.
        eagle_first_token_probs: EAGLE-3's probability distribution over
            vocab for the first draft position. Shape: (vocab_size,).
        topk: Keep branches whose first token is in the top-k.

    Returns:
        New DraftTree with pruned branches.
    """
    # Get EAGLE-3's top-k token IDs
    topk_indices = set(np.argsort(eagle_first_token_probs)[-topk:].tolist())

    # Filter: keep only depth-1 children whose token is in top-k
    pruned = DraftTree()
    for child in retrieval_tree.root.children:
        if child.token_id in topk_indices:
            _copy_subtree(pruned, pruned.root, child)

    return pruned


def prune_by_confidence(
    tree: DraftTree,
    min_prob: float = 0.01,
) -> DraftTree:
    """
    Prune branches where any node's probability falls below threshold.

    More aggressive than top-k pruning -- removes low-confidence
    sub-branches at any depth, not just at the root.

    Args:
        tree: Draft tree to prune.
        min_prob: Minimum probability threshold.

    Returns:
        New DraftTree with pruned branches.
    """
    pruned = DraftTree()
    for child in tree.root.children:
        if child.prob >= min_prob or child.suffix_score > 0:
            _copy_subtree_with_threshold(pruned, pruned.root, child, min_prob)
    return pruned


def prune_to_budget(
    tree: DraftTree,
    max_tokens: int = 64,
) -> DraftTree:
    """
    Prune tree to fit within a token budget.

    Strategy: keep nodes with highest combined score (prob + suffix_score),
    prioritizing depth (longer accepted sequences are more valuable).

    Args:
        tree: Draft tree to prune.
        max_tokens: Maximum number of draft tokens to keep.

    Returns:
        New DraftTree with at most max_tokens nodes.
    """
    all_nodes = tree.get_all_nodes()

    if len(all_nodes) <= max_tokens:
        return tree

    # Score each node: probability weighted by depth bonus
    scored = []
    for node in all_nodes:
        # Depth bonus: deeper nodes in accepted paths are more valuable
        depth_bonus = 1.0 + 0.1 * node.depth
        score = (node.prob + node.suffix_score) * depth_bonus
        scored.append((score, node))

    # Sort by score, keep top max_tokens
    scored.sort(key=lambda x: x[0], reverse=True)
    keep_node_ids = set()

    for _, node in scored[:max_tokens]:
        # Must also keep all ancestors to maintain tree structure
        current = node
        while current is not None and current.token_id >= 0:
            keep_node_ids.add(current.node_id)
            current = current.parent

    # Rebuild tree with only kept nodes
    pruned = DraftTree()
    _rebuild_tree(pruned, pruned.root, tree.root, keep_node_ids)
    return pruned


def _copy_subtree(
    target_tree: DraftTree,
    target_parent: TreeNode,
    source_node: TreeNode,
) -> None:
    """Recursively copy a subtree into target_tree."""
    new_node = target_tree.add_node(
        target_parent,
        source_node.token_id,
        prob=source_node.prob,
        suffix_score=source_node.suffix_score,
        source=source_node.source,
    )
    for child in source_node.children:
        _copy_subtree(target_tree, new_node, child)


def _copy_subtree_with_threshold(
    target_tree: DraftTree,
    target_parent: TreeNode,
    source_node: TreeNode,
    min_prob: float,
) -> None:
    """Copy subtree, skipping branches below threshold."""
    new_node = target_tree.add_node(
        target_parent,
        source_node.token_id,
        prob=source_node.prob,
        suffix_score=source_node.suffix_score,
        source=source_node.source,
    )
    for child in source_node.children:
        if child.prob >= min_prob or child.suffix_score > 0:
            _copy_subtree_with_threshold(
                target_tree, new_node, child, min_prob
            )


def _rebuild_tree(
    target_tree: DraftTree,
    target_parent: TreeNode,
    source_node: TreeNode,
    keep_ids: set[int],
) -> None:
    """Rebuild tree keeping only nodes in keep_ids."""
    for child in source_node.children:
        if child.node_id in keep_ids:
            new_node = target_tree.add_node(
                target_parent,
                child.token_id,
                prob=child.prob,
                suffix_score=child.suffix_score,
                source=child.source,
            )
            _rebuild_tree(target_tree, new_node, child, keep_ids)
