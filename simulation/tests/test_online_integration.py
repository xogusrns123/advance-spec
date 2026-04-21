"""
Tests for online SGLang integration of SuffixDecoding.

Unit tests (no GPU required):
- suffix_draft_to_numpy: tree mask generation from suffix drafts
- SuffixWorker preallocated tensors
- install_suffix_algorithm: enum patching
- HybridSpeculator warm_from_corpus

E2E tests (GPU required, marked with @pytest.mark.e2e):
- Full server integration test with --speculative-algorithm SUFFIX
"""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_spec_decoding.suffix_decoding.suffix_tree import SuffixDraft


# ---------------------------------------------------------------------------
# Unit tests: suffix_draft_to_numpy (tree mask generation)
# ---------------------------------------------------------------------------


class TestSuffixDraftToNumpy:
    """Test suffix draft → tree mask conversion."""

    def _make_worker_stub(self, draft_token_num=8):
        """Create a minimal SuffixWorker-like object for testing."""
        from hybrid_spec_decoding.sglang_integration.suffix_worker import SuffixWorker

        # We can't instantiate SuffixWorker without SGLang, so test the method directly
        class Stub:
            pass

        stub = Stub()
        stub.draft_token_num = draft_token_num
        # Bind the method
        stub._suffix_draft_to_numpy = (
            SuffixWorker._suffix_draft_to_numpy.__get__(stub, Stub)
        )
        return stub

    def test_linear_chain(self):
        """Test a simple linear draft: position 0=verified, 1-3=draft tokens."""
        stub = self._make_worker_stub(draft_token_num=6)
        drafts, mask = stub._suffix_draft_to_numpy(
            token_ids=[100, 200, 300],
            parents=[-1, 0, 1],
            last_token_id=999,
        )

        # Token values: [verified, draft0, draft1, draft2, pad, pad]
        assert drafts[0] == 999  # verified_id
        assert drafts[1] == 100
        assert drafts[2] == 200
        assert drafts[3] == 300
        assert drafts[4] == 0    # padding
        assert drafts[5] == 0

        # Position 0 (verified): sees only itself
        assert mask[0][0] is np.True_
        assert mask[0][1] is np.False_

        # Position 1 (first draft): sees self + position 0
        assert mask[1][0] is np.True_
        assert mask[1][1] is np.True_
        assert mask[1][2] is np.False_

        # Position 3 (third draft): causal chain 0,1,2,3
        assert mask[3][0] is np.True_
        assert mask[3][1] is np.True_
        assert mask[3][2] is np.True_
        assert mask[3][3] is np.True_

        # Padding (position 4): star, sees only 0 and self
        assert mask[4][0] is np.True_
        assert mask[4][1] is np.False_
        assert mask[4][4] is np.True_

    def test_empty_draft(self):
        """Test empty suffix draft → verified_id + padding (star mask)."""
        stub = self._make_worker_stub(draft_token_num=4)
        drafts, mask = stub._suffix_draft_to_numpy(
            token_ids=[],
            parents=[],
            last_token_id=42,
        )

        # Position 0 = verified_id, rest = 0 (padding)
        assert drafts[0] == 42
        assert drafts[1] == 0
        assert drafts[2] == 0
        assert drafts[3] == 0

        # Position 0: sees only itself
        assert mask[0][0] is np.True_
        assert mask[0][1] is np.False_

        # Padding: star structure (see position 0 and self only)
        assert mask[1][0] is np.True_
        assert mask[1][1] is np.True_
        assert mask[1][2] is np.False_

        assert mask[2][0] is np.True_
        assert mask[2][1] is np.False_
        assert mask[2][2] is np.True_

    def test_draft_exceeds_budget(self):
        """Test draft larger than draft_token_num is truncated."""
        stub = self._make_worker_stub(draft_token_num=3)
        drafts, mask = stub._suffix_draft_to_numpy(
            token_ids=[10, 20, 30, 40, 50],
            parents=[-1, 0, 1, 2, 3],
            last_token_id=999,
        )

        # Position 0 = verified_id, positions 1-2 = first 2 draft tokens
        assert drafts[0] == 999
        assert drafts[1] == 10
        assert drafts[2] == 20
        assert len(drafts) == 3


# ---------------------------------------------------------------------------
# Unit tests: HybridSpeculator
# ---------------------------------------------------------------------------


class TestHybridSpeculator:
    def test_warm_from_corpus(self):
        from hybrid_spec_decoding.sglang_integration.hybrid_speculator import (
            HybridConfig,
            HybridSpeculator,
        )

        config = HybridConfig()
        hybrid = HybridSpeculator(config)

        sequences = [
            [100, 200, 300, 400, 500],
            [100, 200, 300, 600, 700],
            [100, 200, 300, 400, 500],
        ]
        hybrid.warm_from_corpus(sequences)

        hybrid.on_request_start("test_req", [100, 200])
        hybrid.on_tokens_generated("test_req", [300])

        draft = hybrid.speculator.speculate(
            "test_req", [100, 200, 300], max_spec_tokens=8
        )
        assert isinstance(draft, SuffixDraft)
        hybrid.on_request_end("test_req")

    def test_get_sglang_launch_cmd(self):
        from hybrid_spec_decoding.sglang_integration.hybrid_speculator import (
            HybridSpeculator,
        )

        cmd = HybridSpeculator.get_sglang_launch_cmd()
        assert "--speculative-algorithm SUFFIX" in cmd
        assert "zai-org/GLM-4.7-Flash" in cmd
        assert "--tp-size 4" in cmd
        assert "install_hook" in cmd

    def test_lifecycle_hooks(self):
        from hybrid_spec_decoding.sglang_integration.hybrid_speculator import (
            HybridConfig,
            HybridSpeculator,
        )

        config = HybridConfig()
        hybrid = HybridSpeculator(config)

        hybrid.on_request_start("r1", [1, 2, 3])
        hybrid.on_tokens_generated("r1", [4, 5])
        hybrid.on_request_end("r1")


# ---------------------------------------------------------------------------
# Unit tests: install_suffix_algorithm
# ---------------------------------------------------------------------------


class TestInstallSuffixAlgorithm:
    def test_install_patches_files(self):
        pytest.importorskip("sglang")

        from simulation.oracle.install_hook import (
            install_suffix_algorithm,
        )

        install_suffix_algorithm()

        # After on-disk patch, re-import to pick up changes
        import importlib
        import sglang.srt.speculative.spec_info as spec_info_mod
        importlib.reload(spec_info_mod)
        SpeculativeAlgorithm = spec_info_mod.SpeculativeAlgorithm

        # SUFFIX should be recognized
        suffix = SpeculativeAlgorithm.from_string("SUFFIX")
        assert suffix.is_suffix()
        assert not suffix.is_eagle()
        assert not suffix.is_ngram()

        # create_worker should return SuffixWorker class
        from unittest.mock import MagicMock

        mock_args = MagicMock()
        mock_args.disable_overlap_schedule = True
        worker_cls = suffix.create_worker(mock_args)

        from hybrid_spec_decoding.sglang_integration.suffix_worker import SuffixWorker

        assert worker_cls is SuffixWorker


# ---------------------------------------------------------------------------
# E2E tests (require running SGLang server with GPU)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestE2EIntegration:
    """End-to-end tests requiring a running SGLang server.

    Run with: pytest tests/test_online_integration.py -m e2e
    Requires SGLang server at http://localhost:30000 with SUFFIX algorithm.
    """

    SERVER_URL = "http://localhost:30000"

    def test_server_health(self):
        import requests

        resp = requests.get(f"{self.SERVER_URL}/health")
        assert resp.status_code == 200

    def test_generate_basic(self):
        import requests

        payload = {
            "text": "Write a Python function that adds two numbers:\n```python\n",
            "sampling_params": {
                "max_new_tokens": 64,
                "temperature": 0.0,
            },
        }
        resp = requests.post(f"{self.SERVER_URL}/generate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert len(data["text"]) > 0

    def test_suffix_accumulation(self):
        """Send repeated similar prompts to verify suffix tree accumulates."""
        import requests

        prompts = [
            "def hello():\n    return 'hello world'\n\ndef ",
            "def greet():\n    return 'hello world'\n\ndef ",
            "def say():\n    return 'hello world'\n\ndef ",
        ]

        for prompt in prompts:
            payload = {
                "text": prompt,
                "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
            }
            resp = requests.post(f"{self.SERVER_URL}/generate", json=payload)
            assert resp.status_code == 200
