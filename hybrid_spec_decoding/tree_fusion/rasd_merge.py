"""
RASD-style parallel tree fusion.

Implements the tree merge strategy from RASD (Retrieval-Augmented
Speculative Decoding): EAGLE-3 tree + SuffixDecoding candidates
are merged via longest prefix matching with probability-based pruning.

Pipeline:
1. EAGLE-3 generates draft tree
2. SuffixDecoding independently generates candidates
3. Candidates are assembled into a retrieval tree
4. Retrieval tree is pruned by EAGLE-3's first-token probability
5. Two trees are merged via longest prefix matching
"""

from __future__ import annotations

import numpy as np

from ..suffix_decoding.speculator import SuffixSpeculator
from .pruning import prune_retrieval_tree, prune_to_budget
from .tree_utils import DraftTree, TreeNode


def build_retrieval_tree(
    candidates: list[list[int]],
    scores: list[float],
) -> DraftTree:
    """
    Build a DraftTree from SuffixDecoding candidates.
    Shared prefixes are automatically merged.

    Args:
        candidates: List of candidate token sequences.
        scores: Frequency-based scores for each candidate.

    Returns:
        DraftTree with candidates as branches.
    """
    tree = DraftTree()
    for cand, score in zip(candidates, scores):
        # Add branch, using score as suffix_score for each node
        node = tree.root
        for token_id in cand:
            # Check existing children
            existing = None
            for child in node.children:
                if child.token_id == token_id:
                    existing = child
                    break

            if existing is not None:
                existing.suffix_score = max(existing.suffix_score, score)
                node = existing
            else:
                node = tree.add_node(
                    node, token_id,
                    suffix_score=score,
                    source="suffix",
                )
    return tree


def longest_prefix_merge(
    eagle_tree: DraftTree,
    retrieval_tree: DraftTree,
) -> DraftTree:
    """
    Merge two trees using longest prefix matching.

    For each branch in the retrieval tree:
    - Walk down the eagle tree matching tokens
    - At the point of divergence, graft the remaining retrieval branch
      onto the eagle tree

    This preserves the full eagle tree structure and adds new branches
    from the retrieval tree where they diverge.

    Args:
        eagle_tree: EAGLE-3's draft tree (primary).
        retrieval_tree: SuffixDecoding's retrieval tree (secondary).

    Returns:
        Merged DraftTree.
    """
    merged = DraftTree()

    # Step 1: Copy the entire eagle tree
    _deep_copy_tree(merged, merged.root, eagle_tree.root)

    # Step 2: For each path in retrieval tree, merge into the copied tree
    retrieval_paths = retrieval_tree.get_all_paths()
    retrieval_leaves = retrieval_tree.get_leaves()

    for leaf in retrieval_leaves:
        path = leaf.path_from_root()
        scores = _collect_scores_along_path(leaf)

        # Walk merged tree, matching as far as possible
        node = merged.root
        match_depth = 0

        for i, token_id in enumerate(path):
            matched_child = None
            for child in node.children:
                if child.token_id == token_id:
                    matched_child = child
                    break

            if matched_child is not None:
                # Update suffix_score on matched nodes
                matched_child.suffix_score = max(
                    matched_child.suffix_score, scores[i]
                )
                matched_child.source = "fused"
                node = matched_child
                match_depth = i + 1
            else:
                # Divergence point: graft remaining tokens
                for j in range(i, len(path)):
                    node = merged.add_node(
                        node,
                        path[j],
                        suffix_score=scores[j] if j < len(scores) else 0.0,
                        source="suffix",
                    )
                break

    return merged


def fuse_trees(
    eagle_tree: DraftTree,
    suffix_speculator: SuffixSpeculator,
    context_tokens: list[int],
    eagle_first_token_probs: np.ndarray | None = None,
    pruning_topk: int = 10,
    max_draft_tokens: int = 64,
) -> DraftTree:
    """
    Full RASD-style tree fusion pipeline.

    Args:
        eagle_tree: EAGLE-3's draft tree.
        suffix_speculator: SuffixSpeculator with populated trees.
        context_tokens: Current token context.
        eagle_first_token_probs: EAGLE-3's prob distribution at first
            draft position. If None, skip pruning.
        pruning_topk: Top-k threshold for pruning.
        max_draft_tokens: Maximum total draft tokens in merged tree.

    Returns:
        Merged and pruned DraftTree.
    """
    # Step 1: Get SuffixDecoding candidates
    result = suffix_speculator.speculate(context_tokens)

    if not result.candidates:
        return eagle_tree

    # Step 2: Build retrieval tree from candidates
    retrieval_tree = build_retrieval_tree(result.candidates, result.scores)

    # Step 3: Prune retrieval tree by EAGLE-3 probability
    if eagle_first_token_probs is not None:
        retrieval_tree = prune_retrieval_tree(
            retrieval_tree, eagle_first_token_probs, pruning_topk
        )

    # Step 4: Merge via longest prefix matching
    merged = longest_prefix_merge(eagle_tree, retrieval_tree)

    # Step 5: Enforce token budget
    if merged.num_nodes > max_draft_tokens:
        merged = prune_to_budget(merged, max_draft_tokens)

    return merged


def _deep_copy_tree(
    target: DraftTree,
    target_parent: TreeNode,
    source_node: TreeNode,
) -> None:
    """Deep copy a tree into target."""
    for child in source_node.children:
        new_node = target.add_node(
            target_parent,
            child.token_id,
            prob=child.prob,
            suffix_score=child.suffix_score,
            source=child.source,
        )
        _deep_copy_tree(target, new_node, child)


def _collect_scores_along_path(leaf: TreeNode) -> list[float]:
    """Collect suffix_scores from root to the given leaf."""
    scores = []
    node = leaf
    while node is not None and node.token_id >= 0:
        scores.append(node.suffix_score)
        node = node.parent
    return list(reversed(scores))
