"""
Sequential Extension: extend EAGLE-3 tree nodes with SuffixDecoding.

Key insight: EAGLE-3 draft tokens create new context that may enable
suffix matches that weren't possible with the original context alone.
This is Case 2a in the research framework -- the unique value of
sequential over parallel fusion.

Strategy:
- Select extension points at depth 1-3 (middle nodes, not leaves)
- At each point, form extended context = original_context + path_to_node
- Query SuffixDecoding with the extended context
- Graft matching continuations as new children
"""

from __future__ import annotations

from ..suffix_decoding.speculator import SuffixSpeculator
from .pruning import prune_to_budget
from .tree_utils import DraftTree, TreeNode


def sequential_extension(
    eagle_tree: DraftTree,
    suffix_speculator: SuffixSpeculator,
    context_tokens: list[int],
    extension_depths: list[int] | None = None,
    min_confidence: float = 0.3,
    max_extensions_per_node: int = 3,
    max_extension_length: int = 5,
    max_draft_tokens: int = 64,
) -> DraftTree:
    """
    Extend EAGLE-3 tree nodes using SuffixDecoding on extended context.

    Args:
        eagle_tree: EAGLE-3's draft tree.
        suffix_speculator: SuffixSpeculator with populated trees.
        context_tokens: Original context tokens.
        extension_depths: Which depths to try extension at (default [1,2,3]).
        min_confidence: Minimum EAGLE-3 probability to consider a node
            as an extension point. Low-confidence nodes are likely wrong,
            so extending from them wastes budget.
        max_extensions_per_node: Max suffix candidates to add per node.
        max_extension_length: Max token length of each extension.
        max_draft_tokens: Total draft token budget.

    Returns:
        Extended DraftTree.
    """
    if extension_depths is None:
        extension_depths = [1, 2, 3]

    # Collect extension candidates, scored by expected value
    extension_points = _select_extension_points(
        eagle_tree, extension_depths, min_confidence
    )

    # Apply extensions
    for node, path_tokens in extension_points:
        if eagle_tree.num_nodes >= max_draft_tokens:
            break

        # Form extended context
        extended_context = context_tokens + path_tokens

        # Query SuffixDecoding with extended context
        result = suffix_speculator.speculate(
            extended_context,
            suffix_len=min(len(extended_context), suffix_speculator.suffix_match_len),
        )

        if not result.candidates:
            continue

        # Add top candidates as new children of this node
        added = 0
        for cand, score in zip(result.candidates, result.scores):
            if added >= max_extensions_per_node:
                break
            if eagle_tree.num_nodes >= max_draft_tokens:
                break

            # Trim to max_extension_length
            cand = cand[:max_extension_length]

            # Skip if this continuation already exists
            if _continuation_exists(node, cand):
                continue

            # Graft the continuation
            _graft_continuation(eagle_tree, node, cand, score)
            added += 1

    # Enforce budget
    if eagle_tree.num_nodes > max_draft_tokens:
        eagle_tree = prune_to_budget(eagle_tree, max_draft_tokens)

    return eagle_tree


def sequential_extension_with_parallel(
    eagle_tree: DraftTree,
    suffix_speculator: SuffixSpeculator,
    context_tokens: list[int],
    eagle_first_token_probs=None,
    pruning_topk: int = 10,
    extension_depths: list[int] | None = None,
    min_confidence: float = 0.3,
    max_draft_tokens: int = 64,
) -> DraftTree:
    """
    Combined approach: RASD parallel merge + sequential extension.
    This is condition (e) in the experiment plan.

    Pipeline:
    1. RASD parallel merge (fuse_trees)
    2. Sequential extension on the merged tree
    """
    from .rasd_merge import fuse_trees

    # Step 1: Parallel merge
    merged = fuse_trees(
        eagle_tree,
        suffix_speculator,
        context_tokens,
        eagle_first_token_probs,
        pruning_topk,
        max_draft_tokens,
    )

    # Step 2: Sequential extension on merged tree
    # Use remaining budget for extensions
    remaining_budget = max_draft_tokens - merged.num_nodes
    if remaining_budget > 5:
        merged = sequential_extension(
            merged,
            suffix_speculator,
            context_tokens,
            extension_depths,
            min_confidence,
            max_extensions_per_node=2,
            max_extension_length=3,
            max_draft_tokens=max_draft_tokens,
        )

    return merged


def _select_extension_points(
    tree: DraftTree,
    depths: list[int],
    min_confidence: float,
) -> list[tuple[TreeNode, list[int]]]:
    """
    Select nodes suitable for sequential extension.

    Criteria:
    - At specified depths
    - Confidence (prob) above threshold
    - Not already heavily branched

    Returns:
        List of (node, path_tokens_from_root) sorted by expected value.
    """
    candidates = []

    for depth in depths:
        for node in tree.get_nodes_at_depth(depth):
            # Skip low-confidence nodes
            if node.prob < min_confidence:
                continue

            # Prefer nodes that are not already heavily branched
            branching_penalty = len(node.children) * 0.1

            # Expected value: P_accept * (1 - branching_penalty)
            expected_value = node.prob * max(0.0, 1.0 - branching_penalty)

            path = node.path_from_root()
            candidates.append((expected_value, node, path))

    # Sort by expected value descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [(node, path) for _, node, path in candidates]


def _continuation_exists(node: TreeNode, continuation: list[int]) -> bool:
    """Check if a continuation already exists as a path from this node."""
    current = node
    for token in continuation:
        found = False
        for child in current.children:
            if child.token_id == token:
                current = child
                found = True
                break
        if not found:
            return False
    return True


def _graft_continuation(
    tree: DraftTree,
    parent: TreeNode,
    continuation: list[int],
    score: float,
) -> None:
    """Graft a continuation onto a node, merging shared prefixes."""
    node = parent
    for i, token_id in enumerate(continuation):
        # Check if this token already exists as a child
        existing = None
        for child in node.children:
            if child.token_id == token_id:
                existing = child
                break

        if existing is not None:
            existing.suffix_score = max(existing.suffix_score, score)
            existing.source = "fused"
            node = existing
        else:
            # New branch: add remaining tokens
            for j in range(i, len(continuation)):
                node = tree.add_node(
                    node,
                    continuation[j],
                    suffix_score=score,
                    source="sequential",
                )
            break
