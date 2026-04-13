"""
Tracing hooks for every speculative decoding step.

Records per-step information:
- proposer tree structure (token ids, parent ids, depth per node)
- local probability / logprob per node
- cumulative (root-to-node) path logprob
- accepted path after verification
- draft latency, verify latency, total step latency

The DecodingTracer collects a list of StepTrace objects across a full
generation and can export the trace as JSON or CSV.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Sequence


@dataclass
class StepTrace:
    """Trace of a single speculative decoding step."""

    step_idx: int

    # ---- proposer tree ----
    proposer_name: str = ""
    num_draft_nodes: int = 0
    tree_depth: int = 0

    # Per-node arrays (BFS order)
    node_token_ids: list[int] = field(default_factory=list)
    node_parent_ids: list[int] = field(default_factory=list)
    node_depths: list[int] = field(default_factory=list)
    node_local_probs: list[float] = field(default_factory=list)
    node_local_logprobs: list[float] = field(default_factory=list)
    node_cumulative_logprobs: list[float] = field(default_factory=list)

    # ---- accepted path ----
    accepted_token_ids: list[int] = field(default_factory=list)
    accepted_length: int = 0  # number of accepted tokens (MAT contribution)

    # ---- latency ----
    draft_latency_s: float = 0.0
    verify_latency_s: float = 0.0
    total_step_latency_s: float = 0.0

    # ---- extra ----
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationTrace:
    """Trace of a full generation (multiple steps)."""

    request_id: str = ""
    prompt_len: int = 0
    total_generated: int = 0
    steps: list[StepTrace] = field(default_factory=list)

    # Aggregate timing
    total_wall_s: float = 0.0
    total_draft_s: float = 0.0
    total_verify_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_len": self.prompt_len,
            "total_generated": self.total_generated,
            "total_wall_s": self.total_wall_s,
            "total_draft_s": self.total_draft_s,
            "total_verify_s": self.total_verify_s,
            "num_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


class DecodingTracer:
    """Collects step-level traces for speculative decoding runs.

    Usage::

        tracer = DecodingTracer()
        tracer.begin_generation("req-1", prompt_len=128)

        for step in decoding_loop:
            tracer.begin_step()

            # draft
            tracer.record_draft(proposer_output)

            # verify
            tracer.begin_verify()
            accepted = verify(...)
            tracer.end_verify(accepted)

            tracer.end_step()

        tracer.end_generation(total_tokens=256)
        tracer.save_json("trace.json")
        tracer.save_csv("trace.csv")
    """

    def __init__(self) -> None:
        self._generations: list[GenerationTrace] = []
        self._current_gen: GenerationTrace | None = None
        self._current_step: StepTrace | None = None
        self._step_start: float = 0.0
        self._verify_start: float = 0.0
        self._step_counter: int = 0

    # ---- generation lifecycle ----

    def begin_generation(self, request_id: str = "", prompt_len: int = 0) -> None:
        self._current_gen = GenerationTrace(
            request_id=request_id,
            prompt_len=prompt_len,
        )
        self._step_counter = 0
        self._gen_start = time.perf_counter()

    def end_generation(self, total_tokens: int = 0) -> GenerationTrace:
        gen = self._current_gen
        assert gen is not None, "end_generation called without begin_generation"
        gen.total_generated = total_tokens
        gen.total_wall_s = time.perf_counter() - self._gen_start
        gen.total_draft_s = sum(s.draft_latency_s for s in gen.steps)
        gen.total_verify_s = sum(s.verify_latency_s for s in gen.steps)
        self._generations.append(gen)
        self._current_gen = None
        return gen

    # ---- step lifecycle ----

    def begin_step(self) -> None:
        self._current_step = StepTrace(step_idx=self._step_counter)
        self._step_start = time.perf_counter()
        self._step_counter += 1

    def record_draft(
        self,
        proposer_name: str = "",
        tree_token_ids: Sequence[int] = (),
        tree_parent_ids: Sequence[int] = (),
        tree_depths: Sequence[int] = (),
        local_probs: Sequence[float] = (),
        local_logprobs: Sequence[float] = (),
        cumulative_logprobs: Sequence[float] = (),
        draft_latency_s: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record draft proposer output for the current step."""
        step = self._current_step
        assert step is not None, "record_draft called outside a step"

        step.proposer_name = proposer_name
        step.node_token_ids = list(tree_token_ids)
        step.node_parent_ids = list(tree_parent_ids)
        step.node_depths = list(tree_depths)
        step.node_local_probs = list(local_probs)
        step.node_local_logprobs = list(local_logprobs)
        step.node_cumulative_logprobs = list(cumulative_logprobs)
        step.num_draft_nodes = len(tree_token_ids)
        step.tree_depth = max(tree_depths) if tree_depths else 0
        step.draft_latency_s = draft_latency_s
        if extra:
            step.extra.update(extra)

    def record_draft_from_proposer_output(self, output: Any) -> None:
        """Convenience: fill step from a ProposerOutput object."""
        self.record_draft(
            proposer_name=output.proposer_name,
            tree_token_ids=output.token_ids,
            tree_parent_ids=output.parent_ids,
            tree_depths=output.depths,
            local_probs=output.local_probs,
            local_logprobs=output.local_logprobs,
            cumulative_logprobs=output.cumulative_logprobs,
            draft_latency_s=output.draft_latency_s,
            extra=output.extra,
        )

    def begin_verify(self) -> None:
        self._verify_start = time.perf_counter()

    def end_verify(self, accepted_token_ids: Sequence[int] = ()) -> None:
        step = self._current_step
        assert step is not None
        step.verify_latency_s = time.perf_counter() - self._verify_start
        step.accepted_token_ids = list(accepted_token_ids)
        step.accepted_length = len(accepted_token_ids)

    def end_step(self) -> StepTrace:
        step = self._current_step
        assert step is not None
        step.total_step_latency_s = time.perf_counter() - self._step_start

        gen = self._current_gen
        assert gen is not None
        gen.steps.append(step)

        self._current_step = None
        return step

    # ---- access ----

    @property
    def generations(self) -> list[GenerationTrace]:
        return list(self._generations)

    @property
    def all_steps(self) -> list[StepTrace]:
        return [s for g in self._generations for s in g.steps]

    # ---- aggregation ----

    def compute_summary(self) -> dict[str, Any]:
        """Compute aggregate metrics across all traced generations."""
        all_steps = self.all_steps
        if not all_steps:
            return {}

        accepted_lens = [s.accepted_length for s in all_steps]
        draft_lats = [s.draft_latency_s for s in all_steps]
        verify_lats = [s.verify_latency_s for s in all_steps]
        total_lats = [s.total_step_latency_s for s in all_steps]

        total_accepted = sum(accepted_lens)
        total_wall = sum(g.total_wall_s for g in self._generations)

        return {
            "num_generations": len(self._generations),
            "num_steps": len(all_steps),
            "mean_accepted_tokens": _mean(accepted_lens),
            "median_accepted_tokens": _median(accepted_lens),
            "total_accepted_tokens": total_accepted,
            "mean_draft_latency_s": _mean(draft_lats),
            "mean_verify_latency_s": _mean(verify_lats),
            "mean_step_latency_s": _mean(total_lats),
            "total_wall_s": total_wall,
            "throughput_tok_per_s": total_accepted / max(total_wall, 1e-9),
        }

    # ---- export ----

    def save_json(self, path: str | Path) -> None:
        """Save all generation traces as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.compute_summary(),
            "generations": [g.to_dict() for g in self._generations],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_csv(self, path: str | Path) -> None:
        """Save step-level traces as CSV (one row per step)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "request_id",
            "step_idx",
            "proposer_name",
            "num_draft_nodes",
            "tree_depth",
            "accepted_length",
            "draft_latency_s",
            "verify_latency_s",
            "total_step_latency_s",
            "node_token_ids",
            "node_parent_ids",
            "node_depths",
            "node_local_probs",
            "node_local_logprobs",
            "node_cumulative_logprobs",
            "accepted_token_ids",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for gen in self._generations:
                for step in gen.steps:
                    writer.writerow({
                        "request_id": gen.request_id,
                        "step_idx": step.step_idx,
                        "proposer_name": step.proposer_name,
                        "num_draft_nodes": step.num_draft_nodes,
                        "tree_depth": step.tree_depth,
                        "accepted_length": step.accepted_length,
                        "draft_latency_s": f"{step.draft_latency_s:.6f}",
                        "verify_latency_s": f"{step.verify_latency_s:.6f}",
                        "total_step_latency_s": f"{step.total_step_latency_s:.6f}",
                        "node_token_ids": json.dumps(step.node_token_ids),
                        "node_parent_ids": json.dumps(step.node_parent_ids),
                        "node_depths": json.dumps(step.node_depths),
                        "node_local_probs": json.dumps(step.node_local_probs),
                        "node_local_logprobs": json.dumps(step.node_local_logprobs),
                        "node_cumulative_logprobs": json.dumps(
                            step.node_cumulative_logprobs
                        ),
                        "accepted_token_ids": json.dumps(step.accepted_token_ids),
                    })

    def to_csv_string(self) -> str:
        """Return CSV as a string (for testing)."""
        buf = StringIO()
        # Reuse save_csv logic through a file-like object
        fieldnames = [
            "request_id", "step_idx", "proposer_name", "num_draft_nodes",
            "tree_depth", "accepted_length", "draft_latency_s",
            "verify_latency_s", "total_step_latency_s",
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for gen in self._generations:
            for step in gen.steps:
                writer.writerow({
                    "request_id": gen.request_id,
                    "step_idx": step.step_idx,
                    "proposer_name": step.proposer_name,
                    "num_draft_nodes": step.num_draft_nodes,
                    "tree_depth": step.tree_depth,
                    "accepted_length": step.accepted_length,
                    "draft_latency_s": f"{step.draft_latency_s:.6f}",
                    "verify_latency_s": f"{step.verify_latency_s:.6f}",
                    "total_step_latency_s": f"{step.total_step_latency_s:.6f}",
                })
        return buf.getvalue()


# ---- helpers ----

def _mean(xs: list[float | int]) -> float:
    return sum(xs) / max(len(xs), 1)


def _median(xs: list[float | int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return float(s[mid])
