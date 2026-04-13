"""
Tree data structures and utilities for speculative decoding draft trees.

Provides TreeNode and DraftTree that are compatible with SGLang's
tree attention mechanism. Handles tree construction, traversal,
attention mask computation, and position ID assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TreeNode:
    """A node in the draft token tree."""

    token_id: int
    depth: int
    parent: Optional[TreeNode] = None
    children: list[TreeNode] = field(default_factory=list)
    prob: float = 0.0           # draft probability from EAGLE-3
    suffix_score: float = 0.0   # score from SuffixDecoding
    source: str = "eagle"       # "eagle", "suffix", or "fused"
    node_id: int = -1           # unique ID for attention mask indexing

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def path_from_root(self) -> list[int]:
        """Get the token sequence from root to this node."""
        tokens = []
        node = self
        while node is not None:
            if node.token_id >= 0:  # skip sentinel root
                tokens.append(node.token_id)
            node = node.parent
        return list(reversed(tokens))

    def add_child(self, token_id: int, **kwargs) -> TreeNode:
        """Add a child node and return it."""
        child = TreeNode(
            token_id=token_id,
            depth=self.depth + 1,
            parent=self,
            **kwargs,
        )
        self.children.append(child)
        return child

    def __repr__(self) -> str:
        return f"TreeNode(id={self.node_id}, token={self.token_id}, depth={self.depth}, src={self.source})"


class DraftTree:
    """
    Draft token tree for speculative decoding.

    Manages a tree of draft tokens with utilities for:
    - Tree construction from EAGLE-3 output or suffix candidates
    - Flattening to token list + tree attention mask (SGLang format)
    - Position ID assignment for tree attention
    - Node traversal and manipulation
    """

    def __init__(self):
        self.root = TreeNode(token_id=-1, depth=0, source="root")
        self.root.node_id = 0
        self._next_id = 1
        self._nodes: list[TreeNode] = [self.root]

    @property
    def num_nodes(self) -> int:
        return len(self._nodes) - 1  # exclude root

    def add_node(
        self,
        parent: TreeNode,
        token_id: int,
        prob: float = 0.0,
        suffix_score: float = 0.0,
        source: str = "eagle",
    ) -> TreeNode:
        """Add a token node to the tree."""
        child = parent.add_child(
            token_id=token_id,
            prob=prob,
            suffix_score=suffix_score,
            source=source,
        )
        child.node_id = self._next_id
        self._next_id += 1
        self._nodes.append(child)
        return child

    def add_sequence(
        self,
        parent: TreeNode,
        token_ids: list[int],
        probs: list[float] | None = None,
        source: str = "eagle",
    ) -> TreeNode:
        """Add a chain of tokens starting from parent. Returns the last node."""
        if probs is None:
            probs = [0.0] * len(token_ids)

        node = parent
        for token_id, prob in zip(token_ids, probs):
            node = self.add_node(node, token_id, prob=prob, source=source)
        return node

    def add_branch_from_root(
        self,
        token_ids: list[int],
        probs: list[float] | None = None,
        source: str = "eagle",
    ) -> TreeNode:
        """Add a branch starting from root, merging shared prefixes."""
        if probs is None:
            probs = [0.0] * len(token_ids)

        node = self.root
        for i, (token_id, prob) in enumerate(zip(token_ids, probs)):
            # Check if this token already exists as a child
            existing = None
            for child in node.children:
                if child.token_id == token_id:
                    existing = child
                    break

            if existing is not None:
                # Merge: update prob if higher
                existing.prob = max(existing.prob, prob)
                node = existing
            else:
                # New branch from here
                node = self.add_sequence(
                    node, token_ids[i:], probs[i:], source=source
                )
                break
        return node

    def get_nodes_at_depth(self, depth: int) -> list[TreeNode]:
        """Get all nodes at a specific depth."""
        return [n for n in self._nodes if n.depth == depth]

    def get_all_nodes(self) -> list[TreeNode]:
        """Get all nodes except root, in BFS order."""
        return [n for n in self._nodes if n.token_id >= 0]

    def get_leaves(self) -> list[TreeNode]:
        """Get all leaf nodes."""
        return [n for n in self._nodes if n.is_leaf and n.token_id >= 0]

    def get_all_paths(self) -> list[list[int]]:
        """Get all root-to-leaf token paths."""
        paths = []
        for leaf in self.get_leaves():
            paths.append(leaf.path_from_root())
        return paths

    def flatten(self) -> tuple[list[int], list[int]]:
        """
        Flatten the tree to a token list and parent index list.

        Returns:
            tokens: List of token IDs in BFS order.
            parent_ids: For each token, the index of its parent in the
                flattened list (-1 for root children).
        """
        nodes = self.get_all_nodes()
        # Sort by (depth, node_id) for deterministic BFS order
        nodes.sort(key=lambda n: (n.depth, n.node_id))

        # Build node_id -> flat_index mapping
        id_to_idx: dict[int, int] = {}
        tokens: list[int] = []
        parent_ids: list[int] = []

        for i, node in enumerate(nodes):
            id_to_idx[node.node_id] = i
            tokens.append(node.token_id)

            if node.parent is None or node.parent.token_id < 0:
                parent_ids.append(-1)  # child of root
            else:
                parent_ids.append(id_to_idx[node.parent.node_id])

        return tokens, parent_ids

    def compute_tree_attention_mask(self, seq_len: int = 0) -> list[list[bool]]:
        """
        Compute the tree attention mask for SGLang-compatible verification.

        Each draft token can attend to:
        1. All prefix tokens (seq_len context tokens)
        2. All ancestor draft tokens on its path from root

        Args:
            seq_len: Number of context tokens before the draft tree.

        Returns:
            mask: (num_draft_tokens, seq_len + num_draft_tokens) boolean mask.
                  mask[i][j] = True means token i can attend to position j.
        """
        nodes = self.get_all_nodes()
        nodes.sort(key=lambda n: (n.depth, n.node_id))
        n = len(nodes)
        total = seq_len + n

        id_to_idx: dict[int, int] = {}
        for i, node in enumerate(nodes):
            id_to_idx[node.node_id] = i

        mask = [[False] * total for _ in range(n)]

        for i, node in enumerate(nodes):
            # Attend to all context tokens
            for j in range(seq_len):
                mask[i][j] = True

            # Attend to self and all ancestors
            current = node
            while current is not None and current.token_id >= 0:
                if current.node_id in id_to_idx:
                    mask[i][seq_len + id_to_idx[current.node_id]] = True
                current = current.parent

        return mask

    def compute_position_ids(self, seq_len: int = 0) -> list[int]:
        """
        Compute position IDs for each draft token.
        Position = seq_len + depth - 1 (tree-aware positioning).

        Args:
            seq_len: Number of context tokens before the draft tree.

        Returns:
            position_ids: List of position IDs for each draft token.
        """
        nodes = self.get_all_nodes()
        nodes.sort(key=lambda n: (n.depth, n.node_id))
        return [seq_len + node.depth - 1 for node in nodes]

    def max_depth(self) -> int:
        """Maximum depth of the tree."""
        if not self._nodes:
            return 0
        return max(n.depth for n in self._nodes)

    @classmethod
    def from_token_paths(
        cls,
        paths: list[list[int]],
        probs: list[list[float]] | None = None,
        source: str = "eagle",
    ) -> DraftTree:
        """
        Build a DraftTree from a list of token paths.
        Shared prefixes are automatically merged.

        Args:
            paths: List of token sequences (each is a root-to-leaf path).
            probs: Optional probabilities for each token in each path.
            source: Source label for all nodes.
        """
        tree = cls()
        if probs is None:
            probs = [None] * len(paths)

        for path, path_probs in zip(paths, probs):
            tree.add_branch_from_root(path, path_probs, source=source)

        return tree

    def __repr__(self) -> str:
        return (
            f"DraftTree(nodes={self.num_nodes}, "
            f"depth={self.max_depth()}, "
            f"leaves={len(self.get_leaves())})"
        )
