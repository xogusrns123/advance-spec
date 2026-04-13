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

### Prerequisites

- NVIDIA GPU (RTX 4090 x4 권장, CUDA 12.2+)
- Docker + Docker Compose + NVIDIA Container Toolkit
- HuggingFace 토큰 ([생성](https://huggingface.co/settings/tokens))

### Docker 환경 구성 (권장)

```bash
# 1. HF 토큰 설정
echo "HF_TOKEN=hf_your_token_here" > .env

# 2. 빌드 & 컨테이너 시작
docker compose up -d --build

# 3. 컨테이너 진입
docker compose exec workspace bash
```

모델 가중치는 `hf-cache` Docker 볼륨에 캐시되어 컨테이너 재생성 시에도 재다운로드 불필요.

### SGLang 서버 실행

컨테이너 안에서 실행:

```bash
# Vanilla (speculative decoding 없음)
python3 -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-Flash \
    --tp-size 4 \
    --mem-fraction-static 0.8 \
    --disable-cuda-graph \
    --host 0.0.0.0 --port 30000

# MTP (모델 내장 Multi-Token Prediction)
python3 -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-Flash \
    --tp-size 4 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.8 \
    --disable-cuda-graph \
    --host 0.0.0.0 --port 30000

# EAGLE3 (외부 드래프트 모델)
python3 -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-Flash \
    --tp-size 4 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path thoughtworks/GLM-4.7-Flash-Eagle3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.8 \
    --disable-cuda-graph \
    --host 0.0.0.0 --port 30000
```

### API 요청 테스트

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-4.7-Flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

### MoE 커널 튜닝 (선택)

RTX 4090에 최적화된 MoE 커널 설정 생성:

```bash
git clone https://github.com/sgl-project/sglang.git /tmp/sglang-repo
cd /tmp/sglang-repo/benchmark/kernels/fused_moe_triton
python tuning_fused_moe_triton.py \
    --model zai-org/GLM-4.7-Flash \
    --tp-size 4 \
    --tune
cp *.json /opt/venv/lib/python3.11/site-packages/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/
```

### 로컬 환경 (Docker 없이)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && uv pip install -e ".[dev]"
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
- Target model: `zai-org/GLM-4.7-Flash` (31B MoE, 30B-A3B)
- EAGLE-3 draft: `thoughtworks/GLM-4.7-Flash-Eagle3` (277MB)
- Max tree budget: 64 tokens (shared across all proposers/baselines)
- GPU: RTX 4090 x4 (tp-size 4), CUDA 12.2
- Lossless: tree verification uses standard rejection sampling, output distribution is identical to the target model

## Docker Architecture

```
docker-compose.yml
├── workspace (container: sglang-bench)
│   ├── /workspace        ← 호스트 소스코드 마운트 (실시간 반영)
│   ├── /opt/venv         ← Python venv (이미지 내, 마운트 영향 없음)
│   └── /root/.cache/hf   ← 모델 캐시 (Docker volume, 영속)
└── volumes
    └── hf-cache          ← HuggingFace 모델 가중치 캐시
```

### 알려진 이슈

- **CUDA Graph OOM**: RTX 4090에서 `--disable-cuda-graph` 필요. `--cuda-graph-max-bs 8`로 대체 가능
- **SGLang + GLM4MoeLite**: `enable_a2a_moe` AttributeError 발생 — Dockerfile에서 자동 패치됨
- **EAGLE3 context_length 불일치**: `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` 환경변수 필요 — docker-compose.yml에 포함됨
