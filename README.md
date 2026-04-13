# Hybrid Speculative Decoding Research Codebase

Modular speculative decoding research framework built on SGLang. Four pluggable proposers, full step-level tracing, two hybrid baselines, and reproducible benchmark infrastructure.

## Background

Speculative decoding accelerates LLM inference by having a lightweight draft model predict multiple tokens ahead, which the target model verifies in a single forward pass. This project provides a **modular research codebase** with four proposers and full instrumentation:

- **MTP** (Multi-Token Prediction): Dedicated prediction heads that forecast multiple future tokens in a single forward pass (DeepSeek-V3 style).
- **Small Draft Model**: A smaller LM (e.g. Llama-3.2-1B) generates candidates autoregressively with top-k branching.
- **EAGLE-3**: Autoregressive draft head using target model hidden states. Generates a tree of candidates via top-k branching at each depth.
- **SuffixDecoding**: Model-free suffix tree pattern matching via Arctic Inference C++ backend. Extremely fast (~20us/token on CPU), effective on repetitive workloads (code, SQL).

All four proposers share a common `DraftTree` data structure and `BaseProposer` interface, making them interchangeable in the benchmark and tracing pipeline.

## Key Ideas

### 1. RASD-style Parallel Merge
EAGLE-3 tree + SuffixDecoding candidates merged via longest prefix matching. SuffixDecoding candidates whose first token falls outside EAGLE-3's top-k are pruned to avoid wasting the draft budget.

### 2. Sequential Extension
EAGLE-3 generates the tree first, then SuffixDecoding extends nodes at depth 1-3 using the **new context created by EAGLE-3's draft tokens**. This can find suffix matches that were impossible with the original context alone (Case 2a).

### 3. Agreement-Guided Tree Construction
When EAGLE-3 and SuffixDecoding agree on a token, reduce branching and extend depth (save budget on confident positions). When they disagree, widen branching (invest budget in uncertain positions).

### 4. Case Analysis Framework

| Case | EAGLE-3 | Suffix | Who contributes? |
|------|---------|--------|------------------|
| 1 | O | O | Both correct -- RASD merge covers longer suffix continuations |
| 2a | O | X -> O | Sequential extension's unique value |
| 2b | O | X | EAGLE-3 alone is sufficient |
| 3 | X | O | SuffixDecoding compensates EAGLE-3's failure |
| 4 | X | X | Both fail -- no fusion can help |

## Project Structure

```
hybrid_spec_decoding/
  proposers/                    --- Four pluggable proposers ---
    base.py                     BaseProposer ABC + ProposerOutput (shared DraftTree)
    mtp_proposer.py             Multi-Token Prediction heads (top-k per head, BFS tree)
    draft_model_proposer.py     Small draft model (autoregressive top-k branching)
    eagle3_proposer.py          EAGLE-3 via SGLang server + offline replay
    suffix_proposer.py          SuffixDecoding via Arctic Inference C++ trees

  tracing/                      --- Step-level instrumentation ---
    tracer.py                   DecodingTracer: per-step tree structure, logprobs,
                                accepted path, draft/verify/total latency. JSON+CSV export.

  tree_fusion/                  --- Shared tree data structure & fusion algorithms ---
    tree_utils.py               TreeNode, DraftTree, attention mask & position ID computation
    pruning.py                  EAGLE-3 probability-based pruning + token budget enforcement
    rasd_merge.py               Parallel merge via longest prefix matching
    sequential_extension.py     Extend EAGLE-3 nodes with SuffixDecoding + combined mode

  suffix_decoding/
    suffix_tree.py              Arctic Inference C++ SuffixDecodingCache wrapper
    speculator.py               Dual tree (global + per-request) candidate generation

  sglang_integration/
    patched_eagle_worker.py     Wraps SGLang's EagleWorker with fusion hooks
    hybrid_speculator.py        ExperimentConfig + HybridSpeculator orchestration

  benchmarks/
    run_benchmark.py            Unified benchmark: speedup, throughput, MAT, TPOT, breakdown
    run_hybrid.py               Two hybrid baselines: suffix+EAGLE-3, RASD fusion
    run_baseline.py             Autoregressive / EAGLE-3 only baselines
    run_fusion.py               5-condition comparison (a-e)
    configs/                    Per-task YAML configs (HumanEval, MT-Bench, DocQA, AgenticSQL)

  analysis/
    collect_eagle3_drafts.py    Collect per-step draft tokens from SGLang EAGLE-3 server
    collect_suffix_candidates.py  Run SuffixDecoding standalone on the same inputs
    compute_complementarity.py  Measure Case 1-4 ratios + per-depth P_accept / P_match
    compute_agreement.py        Agreement score + correlation with actual correctness
    plot_results.py             Visualization (case distribution, depth curves, etc.)

tests/
  test_tree.py                  DraftTree construction, flatten, attention mask, positions
  test_proposers.py             All 4 proposers, budget enforcement, shared output format
  test_tracing.py               StepTrace, DecodingTracer, JSON/CSV export
  test_benchmarks.py            ExperimentConfig, verify, summary, output formats
  test_hybrid_baselines.py      Tree merging, RASD pruning, both fusion baselines

scripts/
  run_all.sh                    Tests + benchmarks + hybrid baselines (no GPU needed)
  run_tests.sh                  pytest suite
  run_benchmark_offline.sh      MTP + DraftModel offline benchmark
  run_hybrid_baselines.sh       Both hybrid baselines with dummy data
```

## Setup

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies (including SGLang)
uv venv && uv pip install -e ".[dev]"

# Activate
source .venv/bin/activate
```

## Quick Start (no GPU required)

```bash
# Run everything: tests + offline benchmarks + hybrid baselines
bash scripts/run_all.sh

# Or individually:
bash scripts/run_tests.sh                       # 60 unit tests
bash scripts/run_benchmark_offline.sh            # MTP + DraftModel benchmark
bash scripts/run_hybrid_baselines.sh             # Both hybrid baselines
```

## Usage

### Unified Benchmark (all proposers)

```bash
# Offline benchmark with dummy prompts (no server needed)
python -m hybrid_spec_decoding.benchmarks.run_benchmark \
  --proposer mtp draft_model \
  --dummy \
  --output-dir results/benchmark

# With real data + SGLang server
python -m hybrid_spec_decoding.benchmarks.run_benchmark \
  --proposer eagle3 \
  --server-url http://localhost:30000 \
  --config hybrid_spec_decoding/benchmarks/configs/humaneval.yaml \
  --output-dir results/benchmark
```

### Hybrid Baselines

```bash
# Run both hybrid baselines (suffix+EAGLE-3 and RASD fusion)
python -m hybrid_spec_decoding.benchmarks.run_hybrid \
  --baselines suffix_eagle_simple rasd_fusion \
  --dummy \
  --output-dir results/hybrid
```

### Tracing

The `DecodingTracer` records every decoding step automatically in benchmarks.
Per-step trace fields:
- Proposer tree: node token IDs, parent IDs, depth
- Local probability / logprob per node
- Cumulative (root-to-node) path logprob
- Accepted path after verification
- Draft latency, verify latency, total step latency

Trace outputs are saved as `*_trace.json` and `*_trace.csv` alongside benchmark results.

### Phase 1: Analysis (run first to decide go/no-go)

```bash
# 1. Start SGLang server with EAGLE-3
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8 \
  --speculative-num-draft-tokens 64

# 2. Collect EAGLE-3 draft tokens
python -m hybrid_spec_decoding.analysis.collect_eagle3_drafts \
  --server-url http://localhost:30000 \
  --dataset humaneval \
  --output-dir results/eagle3_drafts

# 3. Collect SuffixDecoding candidates for the same inputs
python -m hybrid_spec_decoding.analysis.collect_suffix_candidates \
  --eagle3-results results/eagle3_drafts \
  --output-dir results/suffix_candidates

# 4. Compute case distribution (Case 1-4 ratios)
python -m hybrid_spec_decoding.analysis.compute_complementarity \
  --eagle3-results results/eagle3_drafts \
  --suffix-results results/suffix_candidates \
  --check-sequential \
  --output-dir results/complementarity

# 5. Compute agreement scores
python -m hybrid_spec_decoding.analysis.compute_agreement \
  --eagle3-results results/eagle3_drafts \
  --suffix-results results/suffix_candidates \
  --output-dir results/agreement

# 6. Generate plots
python -m hybrid_spec_decoding.analysis.plot_results \
  --complementarity-file results/complementarity/complementarity.json \
  --agreement-file results/agreement/agreement.json \
  --output-dir results/plots
```

### Phase 2: Tree Fusion Experiments

```bash
# Run baselines
python -m hybrid_spec_decoding.benchmarks.run_baseline \
  --mode all \
  --config hybrid_spec_decoding/benchmarks/configs/humaneval.yaml \
  --output-dir results/baselines

# Run all fusion conditions
python -m hybrid_spec_decoding.benchmarks.run_fusion \
  --config hybrid_spec_decoding/benchmarks/configs/humaneval.yaml \
  --output-dir results/fusion
```

## Experiment Conditions

| Condition | Mode | Description |
|-----------|------|-------------|
| (a) | `none` | EAGLE-3 only (baseline) |
| (b) | -- | SuffixDecoding only (baseline) |
| (c) | `parallel` | RASD-style parallel merge |
| (d) | `sequential` | Sequential extension at depth 1-3 |
| (e) | `combined` | (c) + (d) |

## Benchmark Metrics

| Metric | Description |
|--------|-------------|
| **Speedup** | tokens/sec relative to autoregressive baseline |
| **Throughput** | tokens/sec raw generation speed |
| **MAT** | Mean Accepted Tokens per decoding step |
| **TPOT** | Time Per Output Token (ms) |
| **Pipeline breakdown** | draft% / verify% / overhead% time fractions |

All results are saved as both **JSON** and **CSV**. Step-level traces include per-node tree structure, logprobs, accepted paths, and latencies.

## Hybrid Baselines

| Baseline | Description |
|----------|-------------|
| `suffix_eagle_simple` | Suffix + EAGLE-3 merge via longest prefix matching (no pruning) |
| `rasd_fusion` | RASD-style: prune suffix tree by EAGLE-3 top-k + longest-prefix merge + budget cap |

Both baselines use the same tokenizer, sampling setup (temperature=0, greedy), max tree budget, and seed for reproducibility.

## Proposer Interface

All four proposers implement `BaseProposer.propose() -> ProposerOutput`:

```python
from hybrid_spec_decoding.proposers import MTPProposer

proposer = MTPProposer(num_heads=4, topk_per_head=4)
output = proposer.propose(context_ids=[1, 2, 3], max_tokens=64, raw_logits=logits)

output.tree          # DraftTree (shared across all proposers)
output.token_ids     # per-node token IDs (BFS order)
output.parent_ids    # per-node parent indices
output.depths        # per-node depths
output.local_probs   # per-node local probabilities
output.cumulative_logprobs  # root-to-node cumulative logprobs
output.draft_latency_s      # wall-clock drafting time
```

## Settings

- Batch size 1, greedy decoding (temperature=0)
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- EAGLE-3 checkpoint: `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`
- Max tree budget: 64 tokens (shared across all proposers/baselines)
- GPU: A100 80G+ recommended
- Lossless: tree verification uses standard rejection sampling, output distribution is identical to the target model
