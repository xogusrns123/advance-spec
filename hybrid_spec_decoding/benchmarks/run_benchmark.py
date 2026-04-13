"""
Unified benchmark script for speculative decoding proposers.

Reports:
- Speedup (vs autoregressive baseline)
- Throughput (tokens/sec)
- MAT (Mean Accepted Tokens per step)
- TPOT (Time Per Output Token)
- Full pipeline time breakdown (draft / verify / overhead)

Saves results as both JSON and CSV.

Usage:
    python -m hybrid_spec_decoding.benchmarks.run_benchmark \
        --proposer eagle3 \
        --server-url http://localhost:30000 \
        --config hybrid_spec_decoding/benchmarks/configs/humaneval.yaml \
        --output-dir results/benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..proposers.base import BaseProposer, ProposerOutput
from ..tracing.tracer import DecodingTracer


# ---------------------------------------------------------------------------
# Experiment configuration (also fixes the missing ExperimentConfig bug)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Reproducible experiment configuration shared across all baselines."""

    # Model
    target_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    draft_model: str = ""

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.0

    # Tree budget (shared across ALL proposers)
    max_tree_tokens: int = 64

    # EAGLE-3 specific
    num_steps: int = 5
    eagle_topk: int = 8
    num_draft_tokens: int = 64

    # Suffix specific
    suffix_match_len: int = 16
    max_candidates: int = 10

    # Fusion
    fusion_mode: str = "parallel"
    pruning_topk: int = 10
    suffix_budget_ratio: float = 0.3

    # Dataset
    dataset: str = "humaneval"
    max_samples: int = 100

    # Reproducibility
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        flat: dict[str, Any] = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)
            else:
                flat.update(raw)
                break
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in flat.items() if k in known_fields})


# ---------------------------------------------------------------------------
# Per-sample and summary result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    """Result for a single prompt / generation."""

    prompt_id: str
    proposer: str
    num_tokens: int
    num_steps: int
    wall_time_s: float
    tokens_per_sec: float
    mat: float  # mean accepted tokens this sample
    tpot_ms: float  # time per output token (ms)
    draft_time_s: float
    verify_time_s: float
    overhead_time_s: float


@dataclass
class BenchmarkSummary:
    """Aggregate benchmark results for one proposer / condition."""

    proposer: str
    dataset: str
    num_samples: int

    # Core metrics
    throughput_tok_per_s: float
    mat: float
    tpot_ms: float
    speedup: float  # vs autoregressive

    # Time breakdown (averages over samples)
    avg_wall_time_s: float
    avg_draft_time_s: float
    avg_verify_time_s: float
    avg_overhead_time_s: float

    # Pipeline breakdown fractions
    draft_frac: float
    verify_frac: float
    overhead_frac: float


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def simulate_verify(
    draft_token_ids: list[int],
    ground_truth_ids: list[int],
) -> list[int]:
    """Simulate verification by comparing draft tokens to ground truth.

    In a real setting this would call the target model.  For offline
    benchmarking we compare against pre-recorded ground truth.
    """
    accepted: list[int] = []
    for dt, gt in zip(draft_token_ids, ground_truth_ids):
        if dt == gt:
            accepted.append(dt)
        else:
            break
    return accepted


def run_benchmark_offline(
    proposer: BaseProposer,
    prompts: list[dict[str, Any]],
    config: ExperimentConfig,
    ground_truth: dict[str, list[int]] | None = None,
) -> tuple[list[SampleResult], DecodingTracer]:
    """Run an offline benchmark loop for a given proposer.

    Each prompt is fed to the proposer to generate a draft tree.
    If ``ground_truth`` is provided, acceptance is computed by comparing
    the greedy path of the tree against the reference tokens.

    Returns per-sample results and a populated DecodingTracer.
    """
    tracer = DecodingTracer()
    results: list[SampleResult] = []

    for i, prompt_data in enumerate(prompts):
        pid = prompt_data.get("id", str(i))
        context_ids: list[int] = prompt_data.get("token_ids", [])
        gt_ids = (ground_truth or {}).get(pid, [])

        tracer.begin_generation(request_id=pid, prompt_len=len(context_ids))

        total_generated = 0
        step_count = 0
        sample_start = time.perf_counter()

        # Simulate decoding loop
        cursor = 0
        remaining = config.max_new_tokens
        while remaining > 0:
            tracer.begin_step()

            # Draft -- supply synthetic logits for offline-capable proposers
            extra_kwargs: dict[str, Any] = {
                "prompt_text": prompt_data.get("prompt", ""),
            }
            if proposer.name in ("mtp", "draft_model"):
                rng = np.random.RandomState(config.seed + i + step_count)
                n_steps = getattr(proposer, "num_heads",
                                  getattr(proposer, "max_depth", 4))
                syn_logits = [
                    rng.randn(32000).astype(np.float32) for _ in range(n_steps)
                ]
                if proposer.name == "mtp":
                    extra_kwargs["raw_logits"] = syn_logits
                else:
                    extra_kwargs["step_logits"] = syn_logits

            output = proposer.propose(
                context_ids, max_tokens=config.max_tree_tokens,
                **extra_kwargs,
            )
            tracer.record_draft_from_proposer_output(output)

            # Verify
            tracer.begin_verify()
            verify_start = time.perf_counter()

            # Get greedy path from draft tree (first leaf path)
            paths = output.tree.get_all_paths()
            greedy_path = paths[0] if paths else []
            gt_slice = gt_ids[cursor: cursor + len(greedy_path)] if gt_ids else []
            accepted = simulate_verify(greedy_path, gt_slice)

            # If no ground truth, accept the full greedy path (for latency benchmarks)
            if not gt_ids and greedy_path:
                accepted = greedy_path

            verify_time = time.perf_counter() - verify_start
            tracer.end_verify(accepted)
            tracer.end_step()

            n_accepted = max(len(accepted), 1)  # at least 1 token per step
            context_ids = list(context_ids) + accepted[:n_accepted]
            cursor += n_accepted
            total_generated += n_accepted
            remaining -= n_accepted
            step_count += 1

            # Safety: break if proposer produces empty tree
            if output.tree.num_nodes == 0:
                break

        wall_time = time.perf_counter() - sample_start
        gen_trace = tracer.end_generation(total_tokens=total_generated)

        mat = (
            sum(s.accepted_length for s in gen_trace.steps)
            / max(len(gen_trace.steps), 1)
        )
        tpot_ms = (wall_time / max(total_generated, 1)) * 1000
        overhead = wall_time - gen_trace.total_draft_s - gen_trace.total_verify_s

        results.append(SampleResult(
            prompt_id=pid,
            proposer=proposer.name,
            num_tokens=total_generated,
            num_steps=step_count,
            wall_time_s=wall_time,
            tokens_per_sec=total_generated / max(wall_time, 1e-9),
            mat=mat,
            tpot_ms=tpot_ms,
            draft_time_s=gen_trace.total_draft_s,
            verify_time_s=gen_trace.total_verify_s,
            overhead_time_s=overhead,
        ))

    return results, tracer


def compute_summary(
    results: list[SampleResult],
    proposer_name: str,
    dataset: str,
    autoregressive_tps: float = 0.0,
) -> BenchmarkSummary:
    """Aggregate per-sample results into a summary."""
    n = len(results)
    if n == 0:
        return BenchmarkSummary(
            proposer=proposer_name, dataset=dataset, num_samples=0,
            throughput_tok_per_s=0, mat=0, tpot_ms=0, speedup=0,
            avg_wall_time_s=0, avg_draft_time_s=0, avg_verify_time_s=0,
            avg_overhead_time_s=0, draft_frac=0, verify_frac=0, overhead_frac=0,
        )

    total_tokens = sum(r.num_tokens for r in results)
    total_wall = sum(r.wall_time_s for r in results)
    total_draft = sum(r.draft_time_s for r in results)
    total_verify = sum(r.verify_time_s for r in results)
    total_overhead = sum(r.overhead_time_s for r in results)

    throughput = total_tokens / max(total_wall, 1e-9)
    avg_mat = sum(r.mat for r in results) / n
    avg_tpot = sum(r.tpot_ms for r in results) / n
    speedup = throughput / max(autoregressive_tps, 1e-9) if autoregressive_tps > 0 else 0.0

    return BenchmarkSummary(
        proposer=proposer_name,
        dataset=dataset,
        num_samples=n,
        throughput_tok_per_s=throughput,
        mat=avg_mat,
        tpot_ms=avg_tpot,
        speedup=speedup,
        avg_wall_time_s=total_wall / n,
        avg_draft_time_s=total_draft / n,
        avg_verify_time_s=total_verify / n,
        avg_overhead_time_s=total_overhead / n,
        draft_frac=total_draft / max(total_wall, 1e-9),
        verify_frac=total_verify / max(total_wall, 1e-9),
        overhead_frac=total_overhead / max(total_wall, 1e-9),
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_results_json(
    summary: BenchmarkSummary,
    results: list[SampleResult],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_results_csv(
    results: list[SampleResult],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id", "proposer", "num_tokens", "num_steps",
        "wall_time_s", "tokens_per_sec", "mat", "tpot_ms",
        "draft_time_s", "verify_time_s", "overhead_time_s",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def save_comparison_table(
    summaries: dict[str, BenchmarkSummary],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table: dict[str, Any] = {}
    for name, s in summaries.items():
        table[name] = {
            "throughput_tok_per_s": round(s.throughput_tok_per_s, 2),
            "mat": round(s.mat, 3),
            "tpot_ms": round(s.tpot_ms, 3),
            "speedup": round(s.speedup, 3),
            "draft_frac": round(s.draft_frac, 4),
            "verify_frac": round(s.verify_frac, 4),
            "overhead_frac": round(s.overhead_frac, 4),
        }

    # JSON
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(table, f, indent=2)

    # CSV
    csv_path = path.with_suffix(".csv")
    fieldnames = ["proposer", "throughput_tok_per_s", "mat", "tpot_ms",
                  "speedup", "draft_frac", "verify_frac", "overhead_frac"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in table.items():
            writer.writerow({"proposer": name, **row})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_proposer(name: str, config: ExperimentConfig) -> BaseProposer:
    if name == "mtp":
        from ..proposers.mtp_proposer import MTPProposer
        return MTPProposer(num_heads=config.num_steps, topk_per_head=config.eagle_topk)
    elif name == "draft_model":
        from ..proposers.draft_model_proposer import DraftModelProposer
        return DraftModelProposer(topk=config.eagle_topk, max_depth=config.num_steps)
    elif name == "eagle3":
        from ..proposers.eagle3_proposer import Eagle3Proposer
        return Eagle3Proposer(num_steps=config.num_steps, eagle_topk=config.eagle_topk)
    elif name == "suffix":
        from ..proposers.suffix_proposer import SuffixProposer
        return SuffixProposer(max_spec_tokens=config.max_candidates)
    else:
        raise ValueError(f"Unknown proposer: {name}")


def _make_dummy_prompts(n: int = 5) -> list[dict[str, Any]]:
    """Generate tiny dummy prompts for quick smoke tests."""
    prompts = []
    for i in range(n):
        prompts.append({
            "id": f"dummy_{i}",
            "prompt": f"def hello_{i}():\n    ",
            "token_ids": list(range(10 + i)),
        })
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified speculative decoding benchmark")
    parser.add_argument("--proposer", nargs="+", default=["mtp"],
                        choices=["mtp", "draft_model", "eagle3", "suffix"],
                        help="Proposer(s) to benchmark")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-tree-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="results/benchmark")
    parser.add_argument("--autoregressive-tps", type=float, default=0.0,
                        help="Autoregressive tokens/sec baseline for speedup calc")
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy prompts for smoke testing")
    args = parser.parse_args()

    # Config
    if args.config and Path(args.config).exists():
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig(
            dataset=args.dataset,
            max_samples=args.max_samples,
            max_new_tokens=args.max_tokens,
            max_tree_tokens=args.max_tree_tokens,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Load prompts
    if args.dummy:
        prompts = _make_dummy_prompts(5)
    else:
        try:
            from ..analysis.collect_eagle3_drafts import load_dataset
            prompts = load_dataset(config.dataset, config.max_samples)
        except Exception:
            print("Could not load dataset, using dummy prompts")
            prompts = _make_dummy_prompts(5)

    print(f"Loaded {len(prompts)} prompts")
    print(f"Config: max_tree_tokens={config.max_tree_tokens}, "
          f"max_new_tokens={config.max_new_tokens}")

    summaries: dict[str, BenchmarkSummary] = {}

    for proposer_name in args.proposer:
        print(f"\n{'='*60}")
        print(f"Proposer: {proposer_name}")
        print(f"{'='*60}")

        proposer = _make_proposer(proposer_name, config)
        results, tracer = run_benchmark_offline(proposer, prompts, config)

        summary = compute_summary(
            results, proposer_name, config.dataset, args.autoregressive_tps
        )
        summaries[proposer_name] = summary

        # Save per-proposer outputs
        save_results_json(summary, results, output_dir / f"{proposer_name}.json")
        save_results_csv(results, output_dir / f"{proposer_name}.csv")
        tracer.save_json(output_dir / f"{proposer_name}_trace.json")
        tracer.save_csv(output_dir / f"{proposer_name}_trace.csv")

        # Print
        print(f"  Throughput:   {summary.throughput_tok_per_s:.1f} tok/s")
        print(f"  MAT:          {summary.mat:.3f}")
        print(f"  TPOT:         {summary.tpot_ms:.3f} ms")
        print(f"  Speedup:      {summary.speedup:.3f}x")
        print(f"  Time breakdown: "
              f"draft={summary.draft_frac:.1%} "
              f"verify={summary.verify_frac:.1%} "
              f"overhead={summary.overhead_frac:.1%}")

    # Comparison table
    if len(summaries) > 1:
        save_comparison_table(summaries, output_dir / "comparison")
        print(f"\n{'='*60}")
        print("Comparison")
        print(f"{'='*60}")
        print(f"{'Proposer':<15} {'tok/s':>8} {'MAT':>6} {'TPOT':>8} {'Speedup':>8}")
        print("-" * 50)
        for name, s in summaries.items():
            print(f"{name:<15} {s.throughput_tok_per_s:>8.1f} "
                  f"{s.mat:>6.3f} {s.tpot_ms:>7.3f}ms "
                  f"{s.speedup:>7.3f}x")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
