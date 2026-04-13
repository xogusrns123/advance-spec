"""Tests for proposer interface and all four proposers."""

import numpy as np
import pytest

from hybrid_spec_decoding.proposers.base import (
    BaseProposer,
    ProposerOutput,
    populate_output_metadata,
)
from hybrid_spec_decoding.tree_fusion.tree_utils import DraftTree


class TestProposerOutput:
    def test_populate_output_metadata(self):
        tree = DraftTree()
        tree.add_sequence(tree.root, [10, 20, 30], probs=[0.9, 0.8, 0.7])
        output = ProposerOutput(tree=tree)
        populate_output_metadata(output)

        assert output.token_ids == [10, 20, 30]
        assert output.parent_ids == [-1, 0, 1]
        assert output.depths == [1, 2, 3]
        assert len(output.local_probs) == 3
        assert len(output.local_logprobs) == 3
        assert len(output.cumulative_logprobs) == 3

        # Cumulative logprob should decrease (become more negative)
        assert output.cumulative_logprobs[0] > output.cumulative_logprobs[2]

    def test_to_dict(self):
        tree = DraftTree()
        tree.add_node(tree.root, 42, prob=0.5)
        output = ProposerOutput(tree=tree, proposer_name="test")
        populate_output_metadata(output)
        d = output.to_dict()
        assert d["proposer"] == "test"
        assert d["num_nodes"] == 1
        assert d["token_ids"] == [42]


class TestMTPProposer:
    def test_build_tree_from_logits(self):
        from hybrid_spec_decoding.proposers.mtp_proposer import MTPProposer

        proposer = MTPProposer(num_heads=3, topk_per_head=2, temperature=1.0)
        logits = [
            np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=np.float32),
            np.array([0.8, 0.1, 0.4, 0.2, 0.6], dtype=np.float32),
            np.array([0.3, 0.7, 0.1, 0.5, 0.2], dtype=np.float32),
        ]
        output = proposer.propose(
            context_ids=[1, 2, 3],
            max_tokens=20,
            raw_logits=logits,
        )
        assert output.tree.num_nodes > 0
        assert output.proposer_name == "mtp"
        assert output.draft_latency_s > 0
        assert len(output.token_ids) == output.tree.num_nodes

    def test_budget_respected(self):
        from hybrid_spec_decoding.proposers.mtp_proposer import MTPProposer

        proposer = MTPProposer(num_heads=4, topk_per_head=4)
        logits = [np.random.randn(100).astype(np.float32) for _ in range(4)]
        output = proposer.propose(
            context_ids=[1, 2, 3],
            max_tokens=10,
            raw_logits=logits,
        )
        assert output.tree.num_nodes <= 10

    def test_empty_logits(self):
        from hybrid_spec_decoding.proposers.mtp_proposer import MTPProposer

        proposer = MTPProposer()
        output = proposer.propose(context_ids=[1], max_tokens=10, raw_logits=[])
        assert output.tree.num_nodes == 0


class TestDraftModelProposer:
    def test_build_tree_from_step_logits(self):
        from hybrid_spec_decoding.proposers.draft_model_proposer import DraftModelProposer

        proposer = DraftModelProposer(topk=3, max_depth=3)
        logits = [np.random.randn(50).astype(np.float32) for _ in range(3)]
        output = proposer.propose(
            context_ids=[1, 2, 3],
            max_tokens=30,
            step_logits=logits,
        )
        assert output.tree.num_nodes > 0
        assert output.proposer_name == "draft_model"
        assert output.tree.max_depth() <= 3

    def test_budget_respected(self):
        from hybrid_spec_decoding.proposers.draft_model_proposer import DraftModelProposer

        proposer = DraftModelProposer(topk=5, max_depth=5)
        logits = [np.random.randn(100).astype(np.float32) for _ in range(5)]
        output = proposer.propose(
            context_ids=[1, 2],
            max_tokens=8,
            step_logits=logits,
        )
        assert output.tree.num_nodes <= 8


class TestEagle3OfflineProposer:
    def test_from_draft_log(self):
        from hybrid_spec_decoding.proposers.eagle3_proposer import Eagle3OfflineProposer

        log = [
            {
                "draft_token_ids": [10, 20, 30, 40],
                "draft_parent_ids": [-1, 0, 1, 0],
                "draft_probs": [0.9, 0.8, 0.7, 0.6],
            },
        ]
        proposer = Eagle3OfflineProposer(draft_log=log)
        output = proposer.propose(context_ids=[1, 2, 3])
        assert output.tree.num_nodes == 4
        assert output.proposer_name == "eagle3_offline"

        # Second call with no more log entries -> empty tree
        output2 = proposer.propose(context_ids=[1, 2, 3])
        assert output2.tree.num_nodes == 0


class TestAllProposersShareTree:
    """Verify that all four proposers produce standard DraftTree output."""

    def _check_output(self, output: ProposerOutput):
        assert isinstance(output.tree, DraftTree)
        assert output.proposer_name != ""
        assert output.draft_latency_s >= 0
        if output.tree.num_nodes > 0:
            assert len(output.token_ids) == output.tree.num_nodes
            assert len(output.parent_ids) == output.tree.num_nodes
            assert len(output.depths) == output.tree.num_nodes

    def test_mtp(self):
        from hybrid_spec_decoding.proposers.mtp_proposer import MTPProposer
        p = MTPProposer(num_heads=2, topk_per_head=2)
        logits = [np.random.randn(50).astype(np.float32) for _ in range(2)]
        self._check_output(p.propose([1, 2], raw_logits=logits))

    def test_draft_model(self):
        from hybrid_spec_decoding.proposers.draft_model_proposer import DraftModelProposer
        p = DraftModelProposer(topk=2, max_depth=2)
        logits = [np.random.randn(50).astype(np.float32) for _ in range(2)]
        self._check_output(p.propose([1, 2], step_logits=logits))

    def test_eagle3_offline(self):
        from hybrid_spec_decoding.proposers.eagle3_proposer import Eagle3OfflineProposer
        log = [{"draft_token_ids": [5, 6], "draft_parent_ids": [-1, 0]}]
        p = Eagle3OfflineProposer(draft_log=log)
        self._check_output(p.propose([1, 2]))
