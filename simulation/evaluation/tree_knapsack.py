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
) -> tuple[list[int], list[int], list[int], list[int], int]:
    """Per-position accept indicators for a single (tree, ground_truth) pair.

    Position d (1-based) corresponds to depth d in the draft tree (depth 1 =
    direct children of the virtual root, i.e. the first draft token after
    the verified prefix).

    Returns ``(seq, ind, cond_accept, cond_denom, denom_depth)``:
      * ``seq[d-1] = 1`` iff position d is *sequentially* accepted, i.e. the
        greedy tree walk reached depth d (so all of positions 1..d matched).
        Denominator is unconditional (use ``depth_ge`` from caller).
      * ``ind[d-1] = 1`` iff there exists *some* node at depth d whose
        token matches ``ground_truth[d-1]``, regardless of whether earlier
        positions were accepted. Denominator is unconditional.
      * ``cond_accept[d-1] = 1`` iff position d was accepted AND we
        actually reached position d in the greedy walk (i.e., position d-1
        was accepted). Equals ``seq[d-1]`` numerically.
      * ``cond_denom[d-1] = 1`` iff position d was *evaluable* — i.e.,
        position d-1 was accepted (or d == 1). When pos d-1 was rejected,
        ``cond_denom[d-1] = 0`` and pos d does NOT contribute to the
        conditional denominator. ``cond_rate[d] = cond_accept[d-1] /
        sum(cond_denom[d-1])`` measures P(pos d accepts | pos d-1 accepted).
      * ``denom_depth = min(len(ground_truth), max_position)`` — the
        unconditional denominator cap. Caller MUST increment
        ``depth_ge[d]`` for every ``d`` in ``[0, denom_depth)`` regardless
        of tree depth, because positions where the tree has no nodes are
        still "could have drafted but didn't / didn't match" — counting
        them as not-accepted is the right per-step rate. Skipping them
        would bias deep-position accept rates upward.

    All lists have length ``max_position``; positions where the tree has no
    matching node, or beyond ``min(tree_max, gt_len)``, stay 0.

    Linear chains (e.g. draft_model output) are a special case where
    ``parents[i] == i - 1``; the function treats them uniformly with branchy
    EAGLE3/suffix trees.
    """
    n = len(token_ids)
    seq = [0] * max_position
    ind = [0] * max_position
    cond_accept = [0] * max_position
    cond_denom = [0] * max_position
    if max_position <= 0 or not ground_truth:
        return seq, ind, cond_accept, cond_denom, 0

    denom_depth = min(len(ground_truth), max_position)
    if n == 0:
        # Tree empty: pos 1 was evaluable (we were going to look at it) but
        # had no candidates → cond_denom[0] = 1, cond_accept[0] = 0; deeper
        # positions never got evaluated → cond_denom[d>0] = 0.
        if denom_depth > 0:
            cond_denom[0] = 1
        return seq, ind, cond_accept, cond_denom, denom_depth

    # Compute depth per node. The capture is stored in BFS order (parents
    # always have lower index than children), so a single forward pass works.
    depths = [1] * n
    tree_max = 0
    for i in range(n):
        if parents[i] >= 0:
            depths[i] = depths[parents[i]] + 1
        if depths[i] > tree_max:
            tree_max = depths[i]
    match_cap = min(tree_max, denom_depth)
    if match_cap <= 0:
        return seq, ind, cond_accept, cond_denom, denom_depth

    # Sequential greedy walk. cond_denom[d-1] = 1 iff position d was
    # *evaluated* (= pos d-1 was accepted, OR d == 1). cond_accept[d-1] = 1
    # iff also matched. Once a depth fails (mismatch OR tree exhausted at
    # match_cap), deeper positions are not evaluable.
    node = -1
    for d in range(1, denom_depth + 1):
        cond_denom[d - 1] = 1  # we evaluate position d
        if d > match_cap:
            # Tree has no nodes at this depth — counts as not accepted,
            # and no deeper evaluation either.
            break
        gt_token = ground_truth[d - 1]
        matched_idx = -1
        for i in range(n):
            if parents[i] == node and token_ids[i] == gt_token:
                matched_idx = i
                break
        if matched_idx >= 0:
            seq[d - 1] = 1
            cond_accept[d - 1] = 1
            node = matched_idx
        else:
            break  # mismatch — do not evaluate deeper positions

    # Independent per-depth match: at each depth d, scan all nodes at that
    # depth for one whose token equals ground_truth[d-1]. O(N) per depth.
    for d in range(1, match_cap + 1):
        gt_token = ground_truth[d - 1]
        for i in range(n):
            if depths[i] == d and token_ids[i] == gt_token:
                ind[d - 1] = 1
                break

    return seq, ind, cond_accept, cond_denom, denom_depth


def position_accept_rates_2d(
    token_ids: list[int],
    parents: list[int],
    coords: list[tuple[int, int]],
    ground_truth: list[int],
    max_b: int,
    max_e: int,
) -> tuple[list[list[int]], list[list[int]],
           list[list[int]], list[list[int]],
           list[list[int]], int]:
    """2-D, all-paths variant for the (no-dedup) extension method.

    Each draft node carries a coord ``(b, e)`` where:
      * ``b`` = depth in the eagle3 backbone anchor (0 = virtual root).
      * ``e`` = depth into the suffix extension grafted at that anchor
        (0 = the backbone node itself; e ≥ 1 = suffix-cache children).
      * (0, 0) = virtual root / last accepted token — never scored.
      * Total depth in the merged tree equals ``b + e``.

    No-dedup convention: when the tree builder grafts overlapping
    suffix chains we keep duplicate nodes at the same coord. This
    function expects that — it OR-aggregates per cell, so duplicates
    are harmless.

    All-paths convention: instead of a greedy single-path walk we
    consider every root→node path. A cell ``(b, e)`` is *sequentially
    accepted* iff some node at that coord has its full root→node path
    matching ``gt[0..b+e-1]``. *Conditionally evaluable* iff some node
    at ``(b, e)`` has its parent-path fully matched (i.e. some path
    reached the position one step shallower).

    Returns ``(seq, ind, cond_accept, cond_denom, depth_ge,
    denom_depth)`` where the five 2-D arrays have shape
    ``[max_b + 1][max_e + 1]`` and:

      * ``seq[b][e]``         — ∃ node at (b, e) on a fully-matched path.
      * ``ind[b][e]``         — ∃ node at (b, e) with token = gt[b+e-1]
                                (independent of ancestors).
      * ``cond_accept[b][e]`` — same numerator as seq.
      * ``cond_denom[b][e]``  — ∃ node at (b, e) whose parent-path is
                                fully matched (= "the cell was reached").
      * ``depth_ge[b][e]``    — 1 iff 1 ≤ b+e ≤ len(gt), set
                                independently of tree shape so the
                                unconditional denominator follows the
                                every-target-token policy.

    Plus ``denom_depth = min(len(gt), max_b + max_e)``.

    The token-id arrays are assumed to be in topological order (parent
    index < child index) so that a single forward pass computes
    path-match flags.
    """
    n = len(token_ids)
    rows = max_b + 1
    cols = max_e + 1

    def _grid():
        return [[0] * cols for _ in range(rows)]
    seq = _grid()
    ind = _grid()
    cond_accept = _grid()
    cond_denom = _grid()
    depth_ge = _grid()

    if rows <= 0 or cols <= 0 or not ground_truth:
        return seq, ind, cond_accept, cond_denom, depth_ge, 0

    L = len(ground_truth)
    denom_depth = min(L, max_b + max_e)

    # Unconditional denominator — every cell with 1 ≤ b+e ≤ L is
    # evaluable, regardless of tree shape.
    for b in range(rows):
        for e in range(cols):
            if 1 <= b + e <= L:
                depth_ge[b][e] = 1

    if n == 0:
        return seq, ind, cond_accept, cond_denom, depth_ge, denom_depth

    # path_match[i] = True iff every non-root node on root→i path
    # (including i) has token matching gt at its own position. Computed
    # in topological order — virtual root is vacuously matched.
    path_match = [False] * n

    for i in range(n):
        b_i, e_i = coords[i]
        d_i = b_i + e_i
        # Skip nodes outside the (max_b, max_e) box or the virtual root.
        in_grid = (0 <= b_i < rows) and (0 <= e_i < cols) and d_i >= 1
        # Independent — token presence at this coord (any path).
        token_matches = (d_i >= 1 and d_i <= L
                         and token_ids[i] == ground_truth[d_i - 1])
        if in_grid and token_matches:
            ind[b_i][e_i] = 1

        p = parents[i]
        # Parent path-match flag. Virtual root (-1) is vacuously
        # reachable. Otherwise lookup the parent's path_match.
        parent_reached = True if p < 0 else path_match[p]

        # Conditional denominator — this cell *was reached* if the
        # parent's path was fully matched.
        if in_grid and parent_reached:
            cond_denom[b_i][e_i] = 1

        # Sequential / cond_accept — this cell has a fully-matched path
        # ending at node i.
        if parent_reached and token_matches:
            path_match[i] = True
            if in_grid:
                seq[b_i][e_i] = 1
                cond_accept[b_i][e_i] = 1

    return seq, ind, cond_accept, cond_denom, depth_ge, denom_depth
