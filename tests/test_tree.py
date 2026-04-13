"""Tests for DraftTree and TreeNode."""

import pytest

from hybrid_spec_decoding.tree_fusion.tree_utils import DraftTree, TreeNode


class TestTreeNode:
    def test_add_child(self):
        root = TreeNode(token_id=-1, depth=0)
        child = root.add_child(token_id=42, prob=0.9, source="eagle")
        assert child.token_id == 42
        assert child.depth == 1
        assert child.parent is root
        assert root.children == [child]

    def test_path_from_root(self):
        root = TreeNode(token_id=-1, depth=0)
        a = root.add_child(token_id=10)
        a.depth = 1
        b = a.add_child(token_id=20)
        b.depth = 2
        c = b.add_child(token_id=30)
        c.depth = 3
        assert c.path_from_root() == [10, 20, 30]

    def test_is_leaf_and_root(self):
        root = TreeNode(token_id=-1, depth=0)
        assert root.is_root
        assert root.is_leaf  # no children yet
        child = root.add_child(token_id=1)
        assert not root.is_leaf
        assert child.is_leaf
        assert not child.is_root


class TestDraftTree:
    def test_empty_tree(self):
        tree = DraftTree()
        assert tree.num_nodes == 0
        assert tree.max_depth() == 0
        tokens, parents = tree.flatten()
        assert tokens == []
        assert parents == []

    def test_add_node(self):
        tree = DraftTree()
        n1 = tree.add_node(tree.root, 100, prob=0.8, source="eagle")
        assert tree.num_nodes == 1
        assert n1.token_id == 100
        assert n1.depth == 1

    def test_add_sequence(self):
        tree = DraftTree()
        last = tree.add_sequence(tree.root, [10, 20, 30], probs=[0.9, 0.8, 0.7])
        assert tree.num_nodes == 3
        assert last.token_id == 30
        assert last.depth == 3
        assert last.path_from_root() == [10, 20, 30]

    def test_add_branch_from_root_shared_prefix(self):
        tree = DraftTree()
        tree.add_branch_from_root([10, 20, 30])
        tree.add_branch_from_root([10, 20, 40])
        tree.add_branch_from_root([10, 50])
        # 10 -> 20 -> {30, 40}, 10 -> 50
        assert tree.num_nodes == 5  # 10, 20, 30, 40, 50
        paths = tree.get_all_paths()
        path_sets = [tuple(p) for p in paths]
        assert (10, 20, 30) in path_sets
        assert (10, 20, 40) in path_sets
        assert (10, 50) in path_sets

    def test_flatten(self):
        tree = DraftTree()
        tree.add_sequence(tree.root, [1, 2, 3])
        tokens, parents = tree.flatten()
        assert tokens == [1, 2, 3]
        assert parents == [-1, 0, 1]

    def test_flatten_branching(self):
        tree = DraftTree()
        a = tree.add_node(tree.root, 10)
        tree.add_node(tree.root, 20)
        tree.add_node(a, 30)
        tokens, parents = tree.flatten()
        # BFS order: depth 1 first (10, 20), then depth 2 (30)
        assert 10 in tokens
        assert 20 in tokens
        assert 30 in tokens
        assert len(tokens) == 3

    def test_get_nodes_at_depth(self):
        tree = DraftTree()
        tree.add_node(tree.root, 10)
        tree.add_node(tree.root, 20)
        n = tree.add_node(tree.root, 30)
        tree.add_node(n, 40)
        assert len(tree.get_nodes_at_depth(1)) == 3
        assert len(tree.get_nodes_at_depth(2)) == 1

    def test_get_leaves(self):
        tree = DraftTree()
        tree.add_sequence(tree.root, [1, 2])
        tree.add_node(tree.root, 3)
        leaves = tree.get_leaves()
        leaf_tokens = {l.token_id for l in leaves}
        assert leaf_tokens == {2, 3}

    def test_from_token_paths(self):
        paths = [[1, 2, 3], [1, 2, 4], [1, 5]]
        tree = DraftTree.from_token_paths(paths)
        assert tree.num_nodes == 5
        all_paths = tree.get_all_paths()
        assert len(all_paths) == 3

    def test_attention_mask_shape(self):
        tree = DraftTree()
        tree.add_sequence(tree.root, [1, 2, 3])
        mask = tree.compute_tree_attention_mask(seq_len=10)
        assert len(mask) == 3  # 3 draft nodes
        assert len(mask[0]) == 13  # 10 context + 3 draft

    def test_attention_mask_causal(self):
        tree = DraftTree()
        tree.add_sequence(tree.root, [1, 2, 3])
        mask = tree.compute_tree_attention_mask(seq_len=2)
        # Node 0 (depth 1): sees context (0,1) and self
        assert mask[0][0] is True   # context
        assert mask[0][1] is True   # context
        assert mask[0][2] is True   # self
        assert mask[0][3] is False  # child
        assert mask[0][4] is False  # grandchild

    def test_position_ids(self):
        tree = DraftTree()
        tree.add_sequence(tree.root, [1, 2, 3])
        pos = tree.compute_position_ids(seq_len=5)
        assert pos == [5, 6, 7]  # depth 1,2,3 -> 5,6,7

    def test_max_depth(self):
        tree = DraftTree()
        tree.add_sequence(tree.root, [1, 2, 3, 4])
        assert tree.max_depth() == 4
