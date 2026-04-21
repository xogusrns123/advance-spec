"""Tree knapsack DP for optimal subtree selection under budget constraint.

Given a union trie with per-node acceptance probabilities p_t(v|parent(v)),
find the subtree of at most B nodes that maximises expected accepted tokens.

DP formulation (SEQUOIA-style):
    dp[u][b] = p_t(u) * (1 + max_{sum t_v <= b-1} sum_v dp[v][t_v])

    Children are mutually exclusive in acceptance (target model picks one
    token), so summing their expected contributions is correct.
"""

from __future__ import annotations


def _build_children(parents: list[int]) -> dict[int, list[int]]:
    """Build adjacency list from parents array.  -1 = child of virtual root."""
    children: dict[int, list[int]] = {-1: []}
    for i in range(len(parents)):
        children.setdefault(i, [])
    for i, p in enumerate(parents):
        children.setdefault(p, []).append(i)
    return children


def _topo_order_bottom_up(children: dict[int, list[int]], roots: list[int]) -> list[int]:
    """Return nodes in bottom-up order (leaves first)."""
    order: list[int] = []
    stack = list(roots)
    while stack:
        node = stack.pop()
        order.append(node)
        for c in children.get(node, []):
            stack.append(c)
    order.reverse()
    return order


def _knapsack_over_items(
    items: list[tuple[int, list[float]]],
    capacity: int,
) -> tuple[list[float], list[list[tuple[int, int]]]]:
    """Standard bounded knapsack over items with variable-size value functions.

    Each item is (item_id, values) where values[t] is the value of allocating
    t units to this item (values[0] = 0 implicitly if not included).

    Returns (best_values, allocations) where:
        best_values[b] = max total value using budget b
        allocations[b] = list of (item_id, budget) pairs achieving best_values[b]
    """
    prev = [0.0] * (capacity + 1)
    prev_alloc: list[list[tuple[int, int]]] = [[] for _ in range(capacity + 1)]

    for item_id, values in items:
        curr = [0.0] * (capacity + 1)
        curr_alloc: list[list[tuple[int, int]]] = [[] for _ in range(capacity + 1)]

        for b in range(capacity + 1):
            best = prev[b]
            best_t = 0
            max_t = min(b, len(values) - 1)
            for t in range(1, max_t + 1):
                val = prev[b - t] + values[t]
                if val > best:
                    best = val
                    best_t = t
            curr[b] = best
            if best_t == 0:
                curr_alloc[b] = list(prev_alloc[b])
            else:
                curr_alloc[b] = list(prev_alloc[b - best_t])
                curr_alloc[b].append((item_id, best_t))

        prev = curr
        prev_alloc = curr_alloc

    return prev, prev_alloc


def tree_knapsack_dp(
    token_ids: list[int],
    parents: list[int],
    p_t: list[float],
    budget: int,
) -> tuple[float, list[int]]:
    """Find optimal subtree under budget B maximising expected utility.

    Parameters
    ----------
    token_ids : list[int]
        Token id at each node (len N).
    parents : list[int]
        Parent index for each node (-1 = child of virtual root).
    p_t : list[float]
        p_t(v | parent(v)) — acceptance probability for each node.
    budget : int
        Maximum number of nodes in the selected subtree.

    Returns
    -------
    (expected_utility, selected_node_indices)
    """
    n = len(token_ids)
    if n == 0 or budget <= 0:
        return 0.0, []

    children = _build_children(parents)
    roots = children[-1]
    B = min(budget, n)

    # dp[node][b] = expected accepted tokens from subtree rooted at node,
    #               using b nodes total (including node itself).
    # dp[node][b] = p_t(node) * (1 + children_knapsack(b - 1))
    dp: dict[int, list[float]] = {}
    # For backtracking
    child_allocs: dict[int, list[list[tuple[int, int]]]] = {}

    order = _topo_order_bottom_up(children, roots)

    for node in order:
        ch = children.get(node, [])
        dp[node] = [0.0] * (B + 1)
        child_allocs[node] = [[] for _ in range(B + 1)]

        if not ch:
            # Leaf: including costs 1, gives p_t(node) * 1
            for b in range(1, B + 1):
                dp[node][b] = p_t[node]
        else:
            # Build knapsack items: each child with its dp table
            items = [(c, dp[c]) for c in ch]
            best_children, allocs = _knapsack_over_items(items, B)

            for b in range(1, B + 1):
                remaining = b - 1
                dp[node][b] = p_t[node] * (1.0 + best_children[remaining])
                child_allocs[node][b] = list(allocs[remaining])

    # Root level: virtual root distributes budget among root nodes
    if not roots:
        return 0.0, []

    root_items = [(r, dp[r]) for r in roots]
    best_root, root_allocs = _knapsack_over_items(root_items, B)

    best_eu = best_root[B]
    top_allocation = root_allocs[B]

    # Backtrack to collect selected nodes
    selected: list[int] = []
    _backtrack(top_allocation, child_allocs, selected)

    return best_eu, sorted(selected)


def _backtrack(
    allocation: list[tuple[int, int]],
    child_allocs: dict[int, list[list[tuple[int, int]]]],
    selected: list[int],
) -> None:
    """Recursively collect selected node indices."""
    for node, budget in allocation:
        selected.append(node)
        if budget > 1:
            _backtrack(child_allocs[node][budget], child_allocs, selected)


def tree_knapsack_dp_all_budgets(
    token_ids: list[int],
    parents: list[int],
    p_t: list[float],
    budgets: list[int],
) -> dict[int, tuple[float, list[int]]]:
    """Run DP once and extract results for all budgets.

    Much faster than calling tree_knapsack_dp separately for each budget.
    Returns {budget: (expected_utility, selected_node_indices)}.
    """
    n = len(token_ids)
    if n == 0:
        return {b: (0.0, []) for b in budgets}

    max_budget = min(max(budgets), n)
    if max_budget <= 0:
        return {b: (0.0, []) for b in budgets}

    children = _build_children(parents)
    roots = children[-1]
    B = max_budget

    dp: dict[int, list[float]] = {}
    child_allocs: dict[int, list[list[tuple[int, int]]]] = {}

    order = _topo_order_bottom_up(children, roots)

    for node in order:
        ch = children.get(node, [])
        dp[node] = [0.0] * (B + 1)
        child_allocs[node] = [[] for _ in range(B + 1)]

        if not ch:
            for b in range(1, B + 1):
                dp[node][b] = p_t[node]
        else:
            items = [(c, dp[c]) for c in ch]
            best_children, allocs = _knapsack_over_items(items, B)
            for b in range(1, B + 1):
                remaining = b - 1
                dp[node][b] = p_t[node] * (1.0 + best_children[remaining])
                child_allocs[node][b] = list(allocs[remaining])

    if not roots:
        return {b: (0.0, []) for b in budgets}

    root_items = [(r, dp[r]) for r in roots]
    best_root, root_allocs = _knapsack_over_items(root_items, B)

    results = {}
    for budget in budgets:
        b = min(budget, n)
        if b <= 0:
            results[budget] = (0.0, [])
            continue
        best_eu = best_root[b]
        top_allocation = root_allocs[b]
        selected: list[int] = []
        _backtrack(top_allocation, child_allocs, selected)
        results[budget] = (best_eu, sorted(selected))

    return results


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
