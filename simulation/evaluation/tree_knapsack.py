"""Greedy tree walk utility — counts ground-truth-matching tokens in a tree."""

from __future__ import annotations


def greedy_tree_walk(
    token_ids: list[int],
    parents: list[int],
    ground_truth: list[int],
) -> int:
    """Count accepted tokens by greedy tree walk matching ground truth.

    Walks from virtual root, at each level picking the child matching
    the next ground truth token.  Returns consecutive match count.
    """
    accepted = 0
    node = -1  # virtual root
    for gt_token in ground_truth:
        matched = False
        for i in range(len(parents)):
            if parents[i] == node and token_ids[i] == gt_token:
                accepted += 1
                node = i
                matched = True
                break
        if not matched:
            break
    return accepted


def position_accept_rates(
    token_ids: list[int],
    parents: list[int],
    ground_truth: list[int],
    max_position: int,
) -> tuple[list[int], list[int], int]:
    """Per-position accept indicators for a single (tree, ground_truth) pair.

    Position d (1-based) corresponds to depth d in the draft tree (depth 1 =
    direct children of the virtual root, i.e. the first draft token after
    the verified prefix).

    Returns ``(seq, ind, applicable_depth)``:
      * ``seq[d-1] = 1`` iff position d is *sequentially* accepted, i.e. the
        greedy tree walk reached depth d (so all of positions 1..d matched).
      * ``ind[d-1] = 1`` iff there exists *some* node at depth d whose
        token matches ``ground_truth[d-1]``, regardless of whether earlier
        positions were accepted.
      * ``applicable_depth = min(tree_max_depth, len(ground_truth), max_position)``
        — the largest position at which the indicator is meaningful (positions
        beyond this are not measurable for this step and the caller should
        skip them when accumulating denominators).

    Both lists have length ``max_position``; positions ≥ ``applicable_depth``
    stay 0.

    Linear chains (e.g. draft_model output) are a special case where
    ``parents[i] == i - 1``; the function treats them uniformly with branchy
    EAGLE3/suffix trees.
    """
    n = len(token_ids)
    seq = [0] * max_position
    ind = [0] * max_position
    if n == 0 or not ground_truth or max_position <= 0:
        return seq, ind, 0

    # Compute depth per node. The capture is stored in BFS order (parents
    # always have lower index than children), so a single forward pass works.
    depths = [1] * n
    tree_max = 0
    for i in range(n):
        if parents[i] >= 0:
            depths[i] = depths[parents[i]] + 1
        if depths[i] > tree_max:
            tree_max = depths[i]
    cap = min(tree_max, len(ground_truth), max_position)
    if cap <= 0:
        return seq, ind, 0

    # Sequential greedy walk (mirror of greedy_tree_walk, but record per-depth).
    node = -1
    accepted = 0
    for gt_token in ground_truth[:cap]:
        matched = False
        for i in range(n):
            if parents[i] == node and token_ids[i] == gt_token:
                accepted += 1
                node = i
                seq[accepted - 1] = 1
                matched = True
                break
        if not matched:
            break

    # Independent per-depth match: at each depth d, scan all nodes at that
    # depth for one whose token equals ground_truth[d-1]. O(N) per depth.
    for d in range(1, cap + 1):
        gt_token = ground_truth[d - 1]
        for i in range(n):
            if depths[i] == d and token_ids[i] == gt_token:
                ind[d - 1] = 1
                break

    return seq, ind, cap
