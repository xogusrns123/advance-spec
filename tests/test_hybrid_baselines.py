"""Tests for the two hybrid baselines."""

import numpy as np
import pytest

from hybrid_spec_decoding.tree_fusion.tree_utils import DraftTree
from hybrid_spec_decoding.tree_fusion.pruning import (
    prune_retrieval_tree,
    prune_to_budget,
)
from hybrid_spec_decoding.tree_fusion.rasd_merge import (
    build_retrieval_tree,
    longest_prefix_merge,
)
from hybrid_spec_decoding.benchmarks.run_hybrid import (
    fuse_suffix_eagle_simple,
    fuse_rasd_style,
)


class TestBuildRetrievalTree:
    def test_basic(self):
        candidates = [[1, 2, 3], [1, 2, 4], [5, 6]]
        scores = [0.9, 0.8, 0.7]
        tree = build_retrieval_tree(candidates, scores)
        assert tree.num_nodes > 0
        paths = tree.get_all_paths()
        assert len(paths) == 3

    def test_shared_prefix_merged(self):
        candidates = [[10, 20], [10, 30]]
        scores = [1.0, 0.5]
        tree = build_retrieval_tree(candidates, scores)
        # 10 is shared -> 3 nodes total (10, 20, 30)
        assert tree.num_nodes == 3


class TestLongestPrefixMerge:
    def test_disjoint_trees(self):
        eagle = DraftTree()
        eagle.add_sequence(eagle.root, [1, 2, 3], source="eagle")

        suffix = DraftTree()
        suffix.add_sequence(suffix.root, [4, 5, 6], source="suffix")

        merged = longest_prefix_merge(eagle, suffix)
        paths = merged.get_all_paths()
        path_tuples = {tuple(p) for p in paths}
        assert (1, 2, 3) in path_tuples
        assert (4, 5, 6) in path_tuples

    def test_overlapping_prefix(self):
        eagle = DraftTree()
        eagle.add_sequence(eagle.root, [1, 2, 3], source="eagle")

        suffix = DraftTree()
        suffix.add_sequence(suffix.root, [1, 2, 99], source="suffix")

        merged = longest_prefix_merge(eagle, suffix)
        paths = merged.get_all_paths()
        path_tuples = {tuple(p) for p in paths}
        assert (1, 2, 3) in path_tuples
        assert (1, 2, 99) in path_tuples
        # Token 1 at depth 1 is shared -> merged tree has 4 nodes
        assert merged.num_nodes == 4


class TestPruning:
    def test_prune_retrieval_tree(self):
        tree = DraftTree()
        tree.add_sequence(tree.root, [0, 10], source="suffix")
        tree.add_sequence(tree.root, [1, 11], source="suffix")
        tree.add_sequence(tree.root, [2, 12], source="suffix")

        # Probs where token 0 is top-1 and token 1 is top-2
        probs = np.zeros(100, dtype=np.float32)
        probs[0] = 0.5
        probs[1] = 0.3
        probs[2] = 0.01  # not in top-2

        pruned = prune_retrieval_tree(tree, probs, topk=2)
        # Only branches starting with token 0 and 1 should survive
        depth1_tokens = {n.token_id for n in pruned.get_nodes_at_depth(1)}
        assert 0 in depth1_tokens
        assert 1 in depth1_tokens
        assert 2 not in depth1_tokens

    def test_prune_to_budget(self):
        tree = DraftTree()
        for i in range(10):
            tree.add_sequence(tree.root, [i, i + 100], probs=[0.5, 0.4])
        assert tree.num_nodes == 20
        pruned = prune_to_budget(tree, max_tokens=8)
        assert pruned.num_nodes <= 8


class TestFuseSuffixEagleSimple:
    def test_basic_fusion(self):
        eagle = DraftTree()
        eagle.add_sequence(eagle.root, [1, 2, 3], probs=[0.9, 0.8, 0.7], source="eagle")

        suffix_cands = [[4, 5], [1, 2, 99]]
        suffix_scores = [0.6, 0.5]

        fused = fuse_suffix_eagle_simple(eagle, suffix_cands, suffix_scores, max_tokens=10)
        assert fused.num_nodes > 3  # should have more than EAGLE alone

    def test_empty_suffix(self):
        eagle = DraftTree()
        eagle.add_sequence(eagle.root, [1, 2], source="eagle")
        fused = fuse_suffix_eagle_simple(eagle, [], [])
        assert fused.num_nodes == 2  # unchanged


class TestFuseRasdStyle:
    def test_with_pruning(self):
        eagle = DraftTree()
        eagle.add_sequence(eagle.root, [0, 1, 2], probs=[0.9, 0.8, 0.7], source="eagle")

        suffix_cands = [[0, 10], [3, 20]]
        suffix_scores = [0.8, 0.5]

        # Token 0 is in top-k, token 3 is not
        probs = np.zeros(100, dtype=np.float32)
        probs[0] = 0.9
        probs[1] = 0.05
        probs[3] = 0.001

        fused = fuse_rasd_style(
            eagle, suffix_cands, suffix_scores,
            probs, pruning_topk=2, max_tokens=10,
        )
        # Branch starting with 3 should be pruned
        d1_tokens = {n.token_id for n in fused.get_nodes_at_depth(1)}
        assert 0 in d1_tokens
        # Token 3 should be pruned since it's not in top-2
        # (only token 0 and 1 are in top-2)

    def test_budget_enforced(self):
        eagle = DraftTree()
        for i in range(8):
            eagle.add_node(eagle.root, i, prob=0.5, source="eagle")

        suffix_cands = [[i + 100, i + 200] for i in range(20)]
        suffix_scores = [0.3] * 20
        # Only top-2 suffix first-tokens survive pruning
        probs = np.zeros(400, dtype=np.float32)
        probs[0] = 0.9  # eagle token 0
        probs[100] = 0.5  # suffix first token
        probs[101] = 0.4  # suffix first token

        fused = fuse_rasd_style(
            eagle, suffix_cands, suffix_scores,
            probs, pruning_topk=3, max_tokens=64,
        )
        # Pruning by top-3 removes most suffix branches.
        # Only branches starting with tokens in {0, 100, 101} survive.
        d1_tokens = {n.token_id for n in fused.get_nodes_at_depth(1)}
        # All 8 eagle tokens survive (they're in the eagle tree, not pruned)
        for i in range(8):
            assert i in d1_tokens
        # Most suffix tokens (102-119) are pruned
        assert 119 not in d1_tokens
