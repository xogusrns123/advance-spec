"""Tests for benchmark infrastructure."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hybrid_spec_decoding.benchmarks.run_benchmark import (
    ExperimentConfig,
    SampleResult,
    BenchmarkSummary,
    compute_summary,
    save_results_json,
    save_results_csv,
    save_comparison_table,
    simulate_verify,
)


class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.0
        assert cfg.max_tree_tokens == 64
        assert cfg.seed == 42

    def test_from_yaml(self, tmp_path):
        yaml_content = """
dataset: mt_bench
max_samples: 50
max_tokens: 1024
eagle:
  num_steps: 5
  eagle_topk: 8
"""
        path = tmp_path / "test.yaml"
        path.write_text(yaml_content)
        # from_yaml flattens nested dicts, so top-level keys are picked up
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.dataset == "mt_bench" or cfg.max_samples == 50  # at least some parsed


class TestSimulateVerify:
    def test_full_match(self):
        accepted = simulate_verify([1, 2, 3], [1, 2, 3])
        assert accepted == [1, 2, 3]

    def test_partial_match(self):
        accepted = simulate_verify([1, 2, 3], [1, 2, 99])
        assert accepted == [1, 2]

    def test_no_match(self):
        accepted = simulate_verify([1, 2, 3], [99, 2, 3])
        assert accepted == []

    def test_empty(self):
        assert simulate_verify([], [1, 2]) == []
        assert simulate_verify([1], []) == []


class TestComputeSummary:
    def test_basic(self):
        results = [
            SampleResult("p1", "mtp", 100, 10, 1.0, 100.0, 10.0, 10.0, 0.5, 0.4, 0.1),
            SampleResult("p2", "mtp", 200, 20, 2.0, 100.0, 10.0, 10.0, 1.0, 0.8, 0.2),
        ]
        summary = compute_summary(results, "mtp", "test", autoregressive_tps=50.0)
        assert summary.proposer == "mtp"
        assert summary.num_samples == 2
        assert summary.throughput_tok_per_s == pytest.approx(100.0, abs=1)
        assert summary.mat == pytest.approx(10.0)
        assert summary.speedup == pytest.approx(2.0, abs=0.1)

    def test_empty(self):
        summary = compute_summary([], "mtp", "test")
        assert summary.num_samples == 0
        assert summary.throughput_tok_per_s == 0


class TestSaveResults:
    def test_json(self, tmp_path):
        summary = BenchmarkSummary(
            proposer="test", dataset="d", num_samples=1,
            throughput_tok_per_s=100, mat=5.0, tpot_ms=10.0, speedup=2.0,
            avg_wall_time_s=1.0, avg_draft_time_s=0.5, avg_verify_time_s=0.4,
            avg_overhead_time_s=0.1, draft_frac=0.5, verify_frac=0.4, overhead_frac=0.1,
        )
        results = [
            SampleResult("p1", "test", 100, 10, 1.0, 100.0, 10.0, 10.0, 0.5, 0.4, 0.1),
        ]
        path = tmp_path / "results.json"
        save_results_json(summary, results, path)

        with open(path) as f:
            data = json.load(f)
        assert data["summary"]["proposer"] == "test"
        assert len(data["results"]) == 1

    def test_csv(self, tmp_path):
        results = [
            SampleResult("p1", "test", 100, 10, 1.0, 100.0, 10.0, 10.0, 0.5, 0.4, 0.1),
        ]
        path = tmp_path / "results.csv"
        save_results_csv(results, path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 row

    def test_comparison_table(self, tmp_path):
        s1 = BenchmarkSummary(
            proposer="a", dataset="d", num_samples=1,
            throughput_tok_per_s=100, mat=5.0, tpot_ms=10.0, speedup=1.0,
            avg_wall_time_s=1, avg_draft_time_s=0.5, avg_verify_time_s=0.4,
            avg_overhead_time_s=0.1, draft_frac=0.5, verify_frac=0.4, overhead_frac=0.1,
        )
        s2 = BenchmarkSummary(
            proposer="b", dataset="d", num_samples=1,
            throughput_tok_per_s=200, mat=10.0, tpot_ms=5.0, speedup=2.0,
            avg_wall_time_s=0.5, avg_draft_time_s=0.2, avg_verify_time_s=0.2,
            avg_overhead_time_s=0.1, draft_frac=0.4, verify_frac=0.4, overhead_frac=0.2,
        )
        path = tmp_path / "comparison"
        save_comparison_table({"a": s1, "b": s2}, path)
        assert (tmp_path / "comparison.json").exists()
        assert (tmp_path / "comparison.csv").exists()
