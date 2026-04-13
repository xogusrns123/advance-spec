"""Tests for the tracing / instrumentation system."""

import csv
import json
import tempfile
from io import StringIO
from pathlib import Path

import pytest

from hybrid_spec_decoding.tracing.tracer import DecodingTracer, StepTrace, GenerationTrace


class TestStepTrace:
    def test_defaults(self):
        s = StepTrace(step_idx=0)
        assert s.step_idx == 0
        assert s.accepted_length == 0
        assert s.draft_latency_s == 0.0

    def test_to_dict(self):
        s = StepTrace(
            step_idx=1,
            proposer_name="mtp",
            num_draft_nodes=5,
            accepted_token_ids=[10, 20],
            accepted_length=2,
        )
        d = s.to_dict()
        assert d["step_idx"] == 1
        assert d["proposer_name"] == "mtp"
        assert d["accepted_length"] == 2


class TestDecodingTracer:
    def _make_tracer_with_data(self) -> DecodingTracer:
        tracer = DecodingTracer()
        tracer.begin_generation("req-1", prompt_len=10)

        for step in range(3):
            tracer.begin_step()
            tracer.record_draft(
                proposer_name="mtp",
                tree_token_ids=[100 + step, 200 + step],
                tree_parent_ids=[-1, 0],
                tree_depths=[1, 2],
                local_probs=[0.9, 0.8],
                local_logprobs=[-0.1, -0.22],
                cumulative_logprobs=[-0.1, -0.32],
                draft_latency_s=0.001 * (step + 1),
            )
            tracer.begin_verify()
            tracer.end_verify(accepted_token_ids=[100 + step])
            tracer.end_step()

        tracer.end_generation(total_tokens=3)
        return tracer

    def test_step_count(self):
        tracer = self._make_tracer_with_data()
        assert len(tracer.all_steps) == 3

    def test_generation_trace(self):
        tracer = self._make_tracer_with_data()
        assert len(tracer.generations) == 1
        gen = tracer.generations[0]
        assert gen.request_id == "req-1"
        assert gen.total_generated == 3
        assert gen.total_wall_s > 0

    def test_summary(self):
        tracer = self._make_tracer_with_data()
        summary = tracer.compute_summary()
        assert summary["num_generations"] == 1
        assert summary["num_steps"] == 3
        assert summary["mean_accepted_tokens"] == 1.0
        assert summary["total_accepted_tokens"] == 3
        assert summary["throughput_tok_per_s"] > 0

    def test_save_json(self, tmp_path):
        tracer = self._make_tracer_with_data()
        path = tmp_path / "trace.json"
        tracer.save_json(path)

        with open(path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "generations" in data
        assert len(data["generations"]) == 1
        assert len(data["generations"][0]["steps"]) == 3

    def test_save_csv(self, tmp_path):
        tracer = self._make_tracer_with_data()
        path = tmp_path / "trace.csv"
        tracer.save_csv(path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["request_id"] == "req-1"
        assert rows[0]["proposer_name"] == "mtp"
        assert float(rows[0]["draft_latency_s"]) > 0

    def test_to_csv_string(self):
        tracer = self._make_tracer_with_data()
        csv_str = tracer.to_csv_string()
        reader = csv.DictReader(StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 3

    def test_record_draft_from_proposer_output(self):
        from hybrid_spec_decoding.proposers.base import ProposerOutput, populate_output_metadata
        from hybrid_spec_decoding.tree_fusion.tree_utils import DraftTree

        tree = DraftTree()
        tree.add_sequence(tree.root, [1, 2, 3], probs=[0.9, 0.8, 0.7])
        output = ProposerOutput(tree=tree, proposer_name="test", draft_latency_s=0.005)
        populate_output_metadata(output)

        tracer = DecodingTracer()
        tracer.begin_generation("req-2")
        tracer.begin_step()
        tracer.record_draft_from_proposer_output(output)
        tracer.begin_verify()
        tracer.end_verify([1, 2])
        step = tracer.end_step()
        tracer.end_generation(2)

        assert step.proposer_name == "test"
        assert step.num_draft_nodes == 3
        assert step.accepted_length == 2
        assert step.node_token_ids == [1, 2, 3]

    def test_multiple_generations(self):
        tracer = DecodingTracer()
        for req_id in ["a", "b"]:
            tracer.begin_generation(req_id)
            tracer.begin_step()
            tracer.record_draft(proposer_name="x", tree_token_ids=[1], tree_depths=[1])
            tracer.begin_verify()
            tracer.end_verify([1])
            tracer.end_step()
            tracer.end_generation(1)

        assert len(tracer.generations) == 2
        assert len(tracer.all_steps) == 2


class TestTracerEdgeCases:
    def test_empty_tracer(self):
        tracer = DecodingTracer()
        assert tracer.compute_summary() == {}
        assert tracer.all_steps == []

    def test_zero_accepted(self):
        tracer = DecodingTracer()
        tracer.begin_generation("r1")
        tracer.begin_step()
        tracer.record_draft(proposer_name="x")
        tracer.begin_verify()
        tracer.end_verify([])
        tracer.end_step()
        tracer.end_generation(0)

        summary = tracer.compute_summary()
        assert summary["mean_accepted_tokens"] == 0
