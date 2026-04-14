"""Tests for tree_knapsack DP solver."""

from __future__ import annotations

import math
from itertools import combinations

import pytest

from hybrid_spec_decoding.analysis.tree_knapsack import (
    greedy_tree_walk,
    tree_knapsack_dp,
)


# ── greedy_tree_walk tests ──────────────────────────────────────────────


class TestGreedyTreeWalk:
    def test_single_chain_full_match(self):
        # Chain: 0 → 1 → 2  (tokens: A, B, C)
        token_ids = [10, 20, 30]
        parents = [-1, 0, 1]
        ground_truth = [10, 20, 30]
        assert greedy_tree_walk(token_ids, parents, ground_truth) == 3

    def test_single_chain_partial_match(self):
        token_ids = [10, 20, 30]
        parents = [-1, 0, 1]
        ground_truth = [10, 20, 99]  # mismatch at depth 3
        assert greedy_tree_walk(token_ids, parents, ground_truth) == 2

    def test_single_chain_no_match(self):
        token_ids = [10, 20, 30]
        parents = [-1, 0, 1]
        ground_truth = [99, 20, 30]  # first token wrong
        assert greedy_tree_walk(token_ids, parents, ground_truth) == 0

    def test_branching_tree(self):
        #       root
        #      /    \
        #    10(0)  20(1)
        #    |       |
        #   30(2)  40(3)
        token_ids = [10, 20, 30, 40]
        parents = [-1, -1, 0, 1]
        # Ground truth follows right branch
        assert greedy_tree_walk(token_ids, parents, [20, 40]) == 2
        # Ground truth follows left branch
        assert greedy_tree_walk(token_ids, parents, [10, 30]) == 2
        # Ground truth starts left but second token doesn't match
        assert greedy_tree_walk(token_ids, parents, [10, 40]) == 1

    def test_empty_tree(self):
        assert greedy_tree_walk([], [], [1, 2, 3]) == 0

    def test_empty_ground_truth(self):
        assert greedy_tree_walk([10], [-1], []) == 0

    def test_ground_truth_longer_than_tree(self):
        token_ids = [10, 20]
        parents = [-1, 0]
        ground_truth = [10, 20, 30, 40]
        assert greedy_tree_walk(token_ids, parents, ground_truth) == 2


# ── tree_knapsack_dp tests ──────────────────────────────────────────────


class TestTreeKnapsackDP:
    def test_single_node(self):
        token_ids = [10]
        parents = [-1]
        p_t = [0.8]
        eu, selected = tree_knapsack_dp(token_ids, parents, p_t, budget=1)
        assert math.isclose(eu, 0.8)
        assert selected == [0]

    def test_single_chain_budget_covers_all(self):
        # Chain: u(p=0.9) → v(p=0.8) → w(p=0.7)
        # E[accepted] = 0.9 * (1 + 0.8 * (1 + 0.7)) = 0.9 * 2.36 = 2.124
        token_ids = [10, 20, 30]
        parents = [-1, 0, 1]
        p_t = [0.9, 0.8, 0.7]
        eu, selected = tree_knapsack_dp(token_ids, parents, p_t, budget=3)
        assert math.isclose(eu, 2.124, rel_tol=1e-9)
        assert sorted(selected) == [0, 1, 2]

    def test_single_chain_budget_1(self):
        # Only include root: E[accepted] = 0.9
        token_ids = [10, 20, 30]
        parents = [-1, 0, 1]
        p_t = [0.9, 0.8, 0.7]
        eu, selected = tree_knapsack_dp(token_ids, parents, p_t, budget=1)
        assert math.isclose(eu, 0.9)
        assert selected == [0]

    def test_single_chain_budget_2(self):
        # u(p=0.9) → v(p=0.8): E = 0.9 * (1 + 0.8) = 1.62
        token_ids = [10, 20, 30]
        parents = [-1, 0, 1]
        p_t = [0.9, 0.8, 0.7]
        eu, selected = tree_knapsack_dp(token_ids, parents, p_t, budget=2)
        assert math.isclose(eu, 1.62, rel_tol=1e-9)
        assert sorted(selected) == [0, 1]

    def test_two_roots_pick_better(self):
        # Two independent root nodes: A(p=0.9), B(p=0.3)
        # Budget=1: should pick A
        token_ids = [10, 20]
        parents = [-1, -1]
        p_t = [0.9, 0.3]
        eu, selected = tree_knapsack_dp(token_ids, parents, p_t, budget=1)
        assert math.isclose(eu, 0.9)
        assert selected == [0]

    def test_two_roots_budget_2(self):
        # Budget=2: include both, E = 0.9 + 0.3 = 1.2
        token_ids = [10, 20]
        parents = [-1, -1]
        p_t = [0.9, 0.3]
        eu, selected = tree_knapsack_dp(token_ids, parents, p_t, budget=2)
        assert math.isclose(eu, 1.2)
        assert sorted(selected) == [0, 1]

    def test_branching_budget_allocation(self):
        #       root(0, p=1.0)
        #      /              \
        #    A(1, p=0.7)     B(2, p=0.2)
        #    |                |
        #   C(3, p=0.6)    D(4, p=0.9)
        #
        # Budget=3: root(1) + 2 for children
        # Option 1: root + A + C → 1.0 * (1 + 0.7*(1+0.6)) = 1 + 1.12 = 2.12
        # Option 2: root + A + B → 1.0 * (1 + 0.7 + 0.2) = 1.9
        # Option 3: root + B + D → 1.0 * (1 + 0.2*(1+0.9)) = 1 + 0.38 = 1.38
        # Best: option 1
        token_ids = [10, 20, 30, 40, 50]
        parents = [-1, 0, 0, 1, 2]
        p_t = [1.0, 0.7, 0.2, 0.6, 0.9]
        eu, selected = tree_knapsack_dp(token_ids, parents, p_t, budget=3)
        assert math.isclose(eu, 2.12, rel_tol=1e-9)
        assert 0 in selected and 1 in selected and 3 in selected

    def test_branching_budget_4(self):
        # Same tree, budget=4: root + A + C + B
        # = 1.0 * (1 + 0.7*(1+0.6) + 0.2) = 1 + 1.12 + 0.2 = 2.32
        # vs root + A + B + D
        # = 1.0 * (1 + 0.7 + 0.2*(1+0.9)) = 1 + 0.7 + 0.38 = 2.08
        # Best: first option
        token_ids = [10, 20, 30, 40, 50]
        parents = [-1, 0, 0, 1, 2]
        p_t = [1.0, 0.7, 0.2, 0.6, 0.9]
        eu, selected = tree_knapsack_dp(token_ids, parents, p_t, budget=4)
        assert math.isclose(eu, 2.32, rel_tol=1e-9)

    def test_budget_exceeds_tree_size(self):
        token_ids = [10, 20]
        parents = [-1, 0]
        p_t = [0.9, 0.8]
        eu1, sel1 = tree_knapsack_dp(token_ids, parents, p_t, budget=2)
        eu2, sel2 = tree_knapsack_dp(token_ids, parents, p_t, budget=100)
        assert math.isclose(eu1, eu2)
        assert sel1 == sel2

    def test_zero_prob_excluded(self):
        # Two roots: A(p=0.0), B(p=0.5). Budget=1: pick B
        token_ids = [10, 20]
        parents = [-1, -1]
        p_t = [0.0, 0.5]
        eu, selected = tree_knapsack_dp(token_ids, parents, p_t, budget=1)
        assert math.isclose(eu, 0.5)
        assert selected == [1]

    def test_empty_tree(self):
        eu, selected = tree_knapsack_dp([], [], [], budget=5)
        assert eu == 0.0
        assert selected == []

    def test_zero_budget(self):
        eu, selected = tree_knapsack_dp([10], [-1], [0.9], budget=0)
        assert eu == 0.0
        assert selected == []

    def test_dp_vs_brute_force_small_tree(self):
        """Compare DP result against brute-force enumeration on a small tree."""
        #       root(0, p=0.8)
        #      /    \
        #   A(1,0.6)  B(2,0.5)
        #   |          |
        #  C(3,0.4)  D(4,0.7)
        token_ids = [10, 20, 30, 40, 50]
        parents = [-1, 0, 0, 1, 2]
        p_t = [0.8, 0.6, 0.5, 0.4, 0.7]

        for budget in range(1, 6):
            dp_eu, dp_sel = tree_knapsack_dp(token_ids, parents, p_t, budget)
            bf_eu = _brute_force_eu(token_ids, parents, p_t, budget)
            assert math.isclose(dp_eu, bf_eu, rel_tol=1e-9), (
                f"budget={budget}: DP={dp_eu} != BF={bf_eu}"
            )


def _brute_force_eu(
    token_ids: list[int],
    parents: list[int],
    p_t: list[float],
    budget: int,
) -> float:
    """Brute-force: enumerate all valid subtrees up to budget, compute EU."""
    n = len(token_ids)
    best = 0.0
    for size in range(1, min(budget, n) + 1):
        for subset in combinations(range(n), size):
            subset_set = set(subset)
            if not _is_valid_subtree(parents, subset_set):
                continue
            eu = _compute_eu(parents, p_t, subset_set)
            best = max(best, eu)
    return best


def _is_valid_subtree(parents: list[int], subset: set[int]) -> bool:
    """Check that subset forms a valid subtree (all ancestors included)."""
    for node in subset:
        p = parents[node]
        if p != -1 and p not in subset:
            return False
    return True


def _compute_eu(
    parents: list[int],
    p_t: list[float],
    subset: set[int],
) -> float:
    """Compute expected utility of a subtree subset.

    E[node] = p_t[node] * (1 + sum of E[child] for included children)
    """
    children: dict[int, list[int]] = {-1: []}
    for node in subset:
        children.setdefault(node, [])
    for node in subset:
        p = parents[node]
        children.setdefault(p, []).append(node)

    def _eu(node: int) -> float:
        ch = [c for c in children.get(node, []) if c in subset]
        return p_t[node] * (1.0 + sum(_eu(c) for c in ch))

    return sum(_eu(r) for r in children[-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
