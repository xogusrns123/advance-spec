# Hybrid Speculative Decoding Research Codebase

Modular speculative decoding research framework built on SGLang. Four pluggable proposers, full step-level tracing, two hybrid baselines, and a reproducible **oracle simulation pipeline** for upper-bound analysis.

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
hybrid_spec_decoding/           --- Core runtime libraries ---
  proposers/                    Four pluggable draft proposers
    base.py                     BaseProposer ABC + ProposerOutput (shared DraftTree)
    mtp_proposer.py             Multi-Token Prediction heads (top-k per head, BFS tree)
    draft_model_proposer.py     Small draft model (autoregressive top-k branching)
    eagle3_proposer.py          EAGLE-3 via SGLang server + offline replay
    suffix_proposer.py          SuffixDecoding via Arctic Inference C++ trees

  tree_fusion/                  Shared tree data structure & fusion algorithms
    tree_utils.py               TreeNode, DraftTree, attention mask, position IDs
    pruning.py                  EAGLE-3 probability-based pruning + budget enforcement
    rasd_merge.py               Parallel merge via longest prefix matching
    sequential_extension.py     Extend EAGLE-3 nodes with SuffixDecoding

  suffix_decoding/
    suffix_tree.py              Arctic Inference C++ SuffixDecodingCache wrapper
    speculator.py               Dual tree (global + per-request) candidate generation

  sglang_integration/           SGLang runtime integration
    hybrid_speculator.py        ExperimentConfig + HybridSpeculator orchestration
    suffix_worker.py            SuffixDecoding SGLang worker (--speculative-algorithm SUFFIX)

  tracing/
    tracer.py                   DecodingTracer: per-step tree, logprobs, latencies

  benchmarks/
    run_benchmark.py            Unified benchmark: speedup, throughput, MAT, TPOT
    run_hybrid.py               Two hybrid baselines: suffix+EAGLE-3, RASD fusion
    run_baseline.py             Autoregressive / EAGLE-3 only baselines
    run_fusion.py               5-condition comparison (a-e)
    configs/                    Per-task YAML configs (HumanEval, MT-Bench, DocQA, ...)

simulation/                     --- Oracle simulation pipeline ---
  agents/                       Stage 1 per-benchmark agent runners
    bfcl_agent.py               BFCLv3 multi-turn
    bfcl_v4_agent.py            BFCLv4 agentic (WebSearch + Memory)
    specbench_agent.py          SpecBench / MT-Bench
    swebench_agent.py           SWE-bench (LangChain tool-calling)
    tools/                      Tool implementations (BFCL WebSearch, SWE-bench repos)

  oracle/                       SGLang patching + runtime hooks
    install_hook.py             Install SUFFIX algorithm + oracle patches into SGLang
    oracle_patch.py             Oracle vanilla: accept_length=0, log drafts
    oracle_verify_patch.py      Verification-tries replay + latency instrumentation

  pipeline/                     Oracle pipeline data collection
    extract_trajectory.py       Stage 2: extract token sequences (for MTP replay)
    collect_suffix_drafts.py    Stage 3a: per-step suffix drafts
    collect_draft_model.py      Stage 3b: per-step draft-LM proposals
    collect_union_trie.py       Stage 4: merge all proposers into union trie
    collect_target_probs.py     Stage 5: compute p_t via tree attention
    verify_server.py            Lightweight tree verification server for p_t
    calibrate_latency.py        Measure per-token latencies via SGLang server
    save_results.py             Helper for unified results JSON

  evaluation/                   Stage 6 simulation
    run_tree_oracle_sim.py      Tree-budget oracle simulation (DP knapsack + skip-ahead)
    run_oracle_sim.py           Legacy 88+ method flat-chain simulation
    tree_knapsack.py            DP tree knapsack solver

  analysis/                     Offline Phase-1 analysis
    collect_eagle3_drafts.py    Collect per-step drafts from SGLang EAGLE-3 server
    collect_suffix_candidates.py Run SuffixDecoding standalone on the same inputs
    compute_complementarity.py  Case 1-4 ratios + per-depth P_accept / P_match
    compute_agreement.py        Agreement score + correlation with correctness
    plot_results.py             Visualization (case distribution, depth curves, ...)

  scripts/                      Shell drivers + utility scripts
    run_pipeline.sh             Canonical end-to-end oracle pipeline (Stages 1-6)
    run_parallel_stage1.sh      Stage 1: multi-GPU EAGLE3 oracle vanilla
    run_parallel_draft_model.sh Stage 3b: multi-GPU SGLang draft-model proposals
    run_parallel_p_t.sh         Stage 5: multi-GPU target-model p_t collection
    merge_shards.sh             Merge REQ_START/REQ_END partial runs
    rerun_from_stage4.sh        Re-run Stages 3a + 4 + 6 reusing 3b/3c outputs
    rerun_stage6_sharded.sh     Re-run Stage 6 only, sharding budgets
    run_online_test.sh          SGLang + SUFFIX integration test
    replay_oracle.py            Stage 3c worker (MTP replay) + verify-tries replay
    prepare_bfcl_data.py        Prepare BFCLv3 / BFCLv4 dataset.jsonl
    prepare_specbench_data.py   Prepare SpecBench dataset.jsonl
    measure_eagle3_cost.py      EAGLE3 target/draft cost — real-speculative mode
                                across (workload, budget, steps), topk fixed
    measure_draft_model_cost.py Small draft LM per-token latency (Qwen3-0.6B default)
    measure_suffix_cost.py      SuffixDecodingCache.speculate() CPU call time
    _workload_prompts.py        Shared prompt loader (SpecBench/BFCLv4/SWE-Bench)
    measure_{latency,step_latency,verify_latency}.py  Legacy thin HTTP timers
    bench_eagle3_configs.py     EAGLE3 benchmark across configs (real workload)

  notebooks/
    analyze_eagle3_bench.ipynb  EAGLE3 latency / acceptance analysis
    analyze_oracle_sim.ipynb    Oracle simulation cross-workload views
    compare_methods.ipynb       Per-workload method comparison bar charts

  tests/
    test_tree_knapsack.py       DP solver correctness
    test_online_integration.py  SGLang server + oracle patches integration

tests/                          --- Core library tests ---
  test_tree.py                  DraftTree construction, flatten, attention mask
  test_proposers.py             All 4 proposers, budget enforcement
  test_tracing.py               StepTrace, DecodingTracer, JSON/CSV export
  test_benchmarks.py            ExperimentConfig, verify, summary
  test_hybrid_baselines.py      Tree merging, RASD pruning

scripts/                        --- Top-level convenience ---
  run_all.sh                    Tests + benchmarks + hybrid baselines
  run_tests.sh                  pytest suite (core + simulation)
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
    --speculative-algorithm NEXTN \
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

# Qwen3-8B + EAGLE3 (단일 GPU)
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp-size 1 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.85 \
    --disable-cuda-graph \
    --host 0.0.0.0 --port 30000

# SUFFIX (model-free) — simulation.oracle.install_hook 로 SGLang 패치 후 사용
python3 -m simulation.oracle.install_hook -- \
    --model-path zai-org/GLM-4.7-Flash \
    --tp-size 4 \
    --speculative-algorithm SUFFIX \
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
bash scripts/run_tests.sh                       # core + simulation unit tests
bash scripts/run_benchmark_offline.sh            # MTP + DraftModel benchmark
bash scripts/run_hybrid_baselines.sh             # Both hybrid baselines
```

## Core Usage (SGLang runtime)

### Unified Benchmark

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

## Oracle Simulation Pipeline

서버 없이 heterogeneous speculative decoding의 이론적 상한을 측정하는 offline 시뮬레이션 파이프라인. 6-stage + 3-substage 구조, 4 benchmark × 2 model preset × 3 실행 모드.

### 한 줄 실행

```bash
bash simulation/scripts/run_pipeline.sh <benchmark> <model_preset> [num_requests]
```

- `benchmark`: `bfcl_v3` / `bfcl_v4` / `specbench` / `swebench`
- `model_preset`: `glm4_flash` (GLM-4.7-Flash, TP=4) / `qwen3_8b` (Qwen3-8B, TP=1)

### Execution Toggles

| 환경변수 | 기본값 | 역할 |
|---|---|---|
| `UNION_TRIE` | `0` | `1` → Stage 4 (union trie 생성) 실행 + Stage 6의 `union_trie_*` 메소드 활성화 |
| `EU_ORACLE` | `0` | `1` → Stage 5 (p_t 수집) 실행 + Stage 6의 EU oracle 활성화. `UNION_TRIE=1` 필수 |
| `REQ_START` / `REQ_END` | (unset) | 입력 dataset을 `[start:end)` 범위로 slice — 머신별 shard 분산 |
| `NUM_GPUS` | auto | 사용 GPU 수 (미설정시 `nvidia-smi -L` 자동 감지) |
| `GPU_IDS` | (unset) | 사용할 GPU 인덱스 (e.g. `"0,2,3"`) — 설정 시 `NUM_GPUS`를 목록 길이로 override |
| `PORT` | `30000` | Stage 3c (MTP SGLang 서버) baseport |

조합별 동작:

| `UNION_TRIE` | `EU_ORACLE` | Stage 4 | Stage 5 | Stage 6 input | Stage 6 methods |
|---|---|---|---|---|---|
| **0 (기본)** | **0 (기본)** | skip | skip | artifacts 즉석 조립 | choose_one / single / hybrid / extension / c1_e3sfx |
| 1 | 0 | run | skip | `union_trie_data.jsonl` | 위 + `union_trie_*` |
| 1 | 1 | run | run | `union_trie_data_with_pt.jsonl` | 전체 (EU 포함) |
| 0 | 1 | — | — | — | ❌ 진입 시 에러 |

### Pipeline 개요

```
dataset.jsonl
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: EAGLE3 Oracle Vanilla                  (multi-GPU)  │
│   SGLANG_ORACLE_VANILLA=1 → accept_length=0 강제             │
│   agent_results_eagle3.json                                  │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: Extract Trajectory                                  │
│   trajectory.json (Stage 3c의 replay 입력)                   │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: Draft Token Collection                              │
│   3a  Suffix Decoding             (공통, CPU)                │
│       arctic_inference.SuffixDecodingCache + sequential      │
│       → suffix_drafts.jsonl                                  │
│   3b  Draft Model                 (Qwen3 전용, GPU)          │
│       SGLang (Qwen3-0.6B) autoregressive + prefix caching    │
│       → draft_model_drafts.jsonl                             │
│   3c  MTP Oracle Replay           (GLM 전용, GPU)            │
│       SGLang NEXTN + SGLANG_ORACLE_REPLAY=trajectory         │
│       → agent_results_mtp.json                               │
└──────────────────────────────────────────────────────────────┘
    │
    ▼  (UNION_TRIE=1일 때만)
┌──────────────────────────────────────────────────────────────┐
│ Stage 4: Collect Union Trie                                  │
│   3a/3b/3c 결과를 per-step union trie로 병합                 │
│   → union_trie_data.jsonl                                    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼  (EU_ORACLE=1일 때만)
┌──────────────────────────────────────────────────────────────┐
│ Stage 5: Collect Target Model p_t                            │
│   Tree attention으로 각 trie node의 target probability       │
│   → union_trie_data_with_pt.jsonl                            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 6: Oracle Simulation                                   │
│   budget sweep + latency-aware speedup                       │
│   methods: choose_one, single:*, hybrid_e3/dm, extension,    │
│            [union_trie_*], [eu]                              │
│   → tree_oracle_sim.json                                     │
└──────────────────────────────────────────────────────────────┘
```

### Preset별 실행 매트릭스

| Preset | TP | DRAFT_LM | Stage 3 구성 |
|---|---|---|---|
| `qwen3_8b` | 1 | `Qwen/Qwen3-0.6B` | 3a + 3b (MTP head 없음) |
| `glm4_flash` | 4 | — | 3a + 3c (MTP head 존재) |

### Artifact Flow

```
input_slice.jsonl ──▶ Stage 1 ──▶ agent_results_eagle3.json
                                      │
                                      ├─▶ Stage 2  ──▶ trajectory.json
                                      ├─▶ Stage 3a ──▶ suffix_drafts.jsonl        ┐
                                      ├─▶ Stage 3b ──▶ draft_model_drafts.jsonl   │ Qwen3
                                      └─▶ Stage 3c ──▶ agent_results_mtp.json     ┘ GLM
                                                           │
                                                           ▼  (UNION_TRIE=1)
                                                     Stage 4 ──▶ union_trie_data.jsonl
                                                                       │
                                                                       ▼  (EU_ORACLE=1)
                                                                 Stage 5 ──▶ union_trie_data_with_pt.jsonl
                                                                                     │
                                                                                     ▼
                                                                               Stage 6 ──▶ tree_oracle_sim.json
```

### Stage 상세

#### Stage 1: EAGLE3 Oracle Vanilla

각 decoding step의 draft tree를 전체 기록. `oracle_patch.patch_eagle_worker_full`이 `verify_tree_greedy_func`을 패치하여 accept_length=0 강제 → 매 step 1 token씩 진행하면서 tree 구조와 verification logits 기반 p_t를 로깅.

Oracle entry 구조: `{req_id, tokens, eagle3 (flat drafts), eagle3_tree ({token_ids, parents}, BFS), eagle3_tree_p_t}`.

#### Stage 2: Extract Trajectory

```bash
python3 -m simulation.pipeline.extract_trajectory \
    --agent-results agent_results_eagle3.json \
    --output trajectory.json
```

Stage 3c의 NEXTN replay에 사용.

#### Stage 3a: Suffix Decoding

```bash
python3 -m simulation.pipeline.collect_suffix_drafts \
    --agent-results agent_results_eagle3.json \
    --output suffix_drafts.jsonl \
    --model Qwen/Qwen3-8B
```

- `arctic_inference.SuffixDecodingCache`로 각 step context에서 speculate
- 요청 순차 iteration, cache 누적 (sequential determinism 유지)
- Output schema:
  ```json
  {"request_id": "...", "call_idx": 0, "step_idx": 5,
   "token_ids": [...], "parents": [...], "score": 0.9}
  ```

#### Stage 3b: Draft Model (`DRAFT_LM` 지정 시)

```bash
bash simulation/scripts/run_parallel_draft_model.sh \
    agent_results_eagle3.json draft_model_drafts.jsonl \
    Qwen/Qwen3-0.6B 4 16 \
    --target-model Qwen/Qwen3-8B
```

- N GPU에 SGLang (draft LM) 병렬 기동, prefix caching 활용
- 각 step context → autoregressive 생성 → flat chain draft
- request 단위 bin-packing으로 shard 분산

#### Stage 3c: MTP Oracle Replay (MTP-capable 모델만)

```bash
# run_pipeline.sh 가 다음을 순서대로 처리:
# 1) replay_trajectory.json 생성 (dry-run)
# 2) SGLang NEXTN 서버 기동 + SGLANG_ORACLE_REPLAY 설정
# 3) replay_oracle.py 로 MTP draft 수집
```

Output: `agent_results_mtp.json` (포맷은 Stage 1과 동일).

#### Stage 4: Collect Union Trie (UNION_TRIE=1)

```bash
python3 -m simulation.pipeline.collect_union_trie \
    --agent-results agent_results_eagle3.json \
    --suffix-drafts suffix_drafts.jsonl \
    --draft-model-drafts draft_model_drafts.jsonl \
    --mtp-agent-results agent_results_mtp.json \
    --output union_trie_data.jsonl \
    --model Qwen/Qwen3-8B
```

- 3a/3b/3c 결과를 `(request_id, call_idx, step_idx)` 키로 O(1) lookup해서 merge
- `build_union_trie()`: 각 proposer의 root-to-leaf path를 trie에 삽입, BFS flatten
- BFS 보장: `parent[i] < i` (budget truncation 안전)
- `context_token_ids`, `ground_truth_future` 포함 (Stage 5/6 입력)

#### Stage 5: Collect Target p_t (EU_ORACLE=1)

```bash
bash simulation/scripts/run_parallel_p_t.sh \
    union_trie_data.jsonl union_trie_data_with_pt.jsonl \
    Qwen/Qwen3-8B 4
```

- Tree attention: trie node는 context + trie 내 ancestor에만 attend
- `p_t(v) = softmax(logits[parent(v)])[v.token_id]`
- KV cache 재사용: 동일 `(request_id, call_idx)` 내 incremental forward

#### Stage 6: Oracle Simulation

**UNION_TRIE=1, EU_ORACLE=1 (전체):**
```bash
python3 -m simulation.evaluation.run_tree_oracle_sim \
    --union-trie-data union_trie_data_with_pt.jsonl \
    --budgets 1,2,4,8,16,32,64,128,256,512 \
    --p-t-key p_t \
    --enable-eu \
    --latency-config latency_config.json \
    --output tree_oracle_sim.json --print-summary
```

**UNION_TRIE=0 (Stage 4/5 skip, 즉석 조립):**
```bash
python3 -m simulation.evaluation.run_tree_oracle_sim \
    --agent-results agent_results_eagle3.json \
    --suffix-drafts suffix_drafts.jsonl \
    --draft-model-drafts draft_model_drafts.jsonl \
    --budgets 1,2,4,8,16,32,64,128,256,512 \
    --p-t-key p_t_oracle \
    --no-union-trie \
    --latency-config latency_config.json \
    --output tree_oracle_sim.json --print-summary
```

**Oracle 전략**:
- **Choose-One (`c1`)** — 각 step에서 최고 acceptance를 주는 proposer 선택
- **Single (`single:<name>`)** — 특정 proposer 하나만 사용 (baseline)
- **Hybrid (`hybrid_e3:t`, `hybrid_dm:t`)** — suffix score ≥ threshold이면 suffix, 아니면 fallback (eagle3 또는 draft_model)
- **Extension (`extension`, `extension_dmsfx`)** — base tree의 모든 node에서 suffix decoding으로 확장
- **C1 subset (`c1_e3sfx`)** — eagle3+suffix 서브셋 choose-one
- **Union Trie (`union_trie_e3sfx`, `union_trie_all`)** — budget B 내 BFS truncation + greedy walk *(UNION_TRIE=1)*
- **Expected-Utility (`eu`, `eu_e3sfx`)** — DP tree knapsack으로 budget 내 최적 subtree *(EU_ORACLE=1)*

**Latency-aware simulation**:
- 각 step에서 tree 선택 → `greedy_tree_walk`로 acceptance 측정
- step cost = target_forward(B) + max(draft costs); suffix = 0 (CPU)
- `speedup = (total_tokens × vanilla_ms) / total_time_ms`

### 머신간 분산 실행

```bash
# Machine A
REQ_START=0 REQ_END=50  bash simulation/scripts/run_pipeline.sh bfcl_v4 glm4_flash

# Machine B
REQ_START=50 REQ_END=100 bash simulation/scripts/run_pipeline.sh bfcl_v4 glm4_flash

# Merge + Stage 6 재실행
bash simulation/scripts/merge_shards.sh simulation/results/glm4_flash/bfcl_v4
```

### 부분 재실행

Suffix 파라미터 튜닝 시 (Stage 3b/3c 결과 재사용):

```bash
bash simulation/scripts/rerun_from_stage4.sh \
    simulation/results/qwen3_8b/bfcl_v4 qwen3_8b
```

Stage 6만 budget shard 병렬:

```bash
bash simulation/scripts/rerun_stage6_sharded.sh \
    simulation/results/qwen3_8b/bfcl_v4
```

### Latency Measurement (pre-pipeline)

파이프라인 실행 전에 **real-mode speculative decoding** 상태에서 target / draft / suffix 비용을 각각 실측하는 3개 독립 스크립트. Oracle-vanilla (`accept_length=0` 강제) 기반의 구 측정 로직은 전면 교체됨 — 모든 신규 스크립트는 실제 accept 분포와 KV cache 성장 패턴을 그대로 반영.

#### 측정 데이터

Workload 는 `data/{specbench,bfcl_agent,swebench}/dataset.jsonl` 에서 **첫 user 메시지 기준 2개 prompt** 만 사용 (1 = warmup 버림, 1 = measured). SWE-Bench 데이터가 없으면 자동 skip.

#### 1. EAGLE3 step cost — `measure_eagle3_cost.py`

```bash
python3 simulation/scripts/measure_eagle3_cost.py \
    --model Qwen/Qwen3-8B \
    --draft-model Tengyunw/qwen3_8b_eagle3 \
    --workloads specbench,bfcl_v4,swebench \
    --budgets 4,16,32,64,128,256,512 \
    --steps 2,4,6,8 \
    --topk 16 \
    --output simulation/results/qwen3_8b/eagle3_cost.json
```

- `(budget, steps)` pair 당 SGLang 서버 1회 재기동 (`len(budgets) × len(steps) = 28` 회), 같은 서버에서 3 workload × 2 prompt 를 연속 처리해 server on/off 최소화.
- 내부적으로 `SGLANG_ORACLE_VANILLA=1 SGLANG_LATENCY_ONLY=1` 을 고정 export — oracle patch 의 timing instrumentation 만 활성화하고 accept-force 는 비활성. 서버는 **실제 speculative decoding** 으로 동작.
- `topk` 는 16 고정 (큰 budget 수용용 branching 여유).
- Resume 지원: 같은 `--output` 존재 시 `(workload, budget, steps)` 캐시된 엔트리 skip. `(budget, steps)` pair 전부 캐시되면 서버도 안 띄움.
- 병렬: `--tp-size N` 지원 (단 Stage 1 의 per-GPU sharding 과 다름 — 1 서버).

**출력 `eagle3_cost.json`** (핵심 필드):

| 필드 | 의미 |
|---|---|
| `target_cost_ms` | median(`target_forward_ms + verify_overhead_ms`) — target 모델이 budget 트리를 검증하는 전체 시간 |
| `draft_cost_ms` | median(`eagle3_draft_ms`) — EAGLE3 draft head 가 트리를 생성하는 시간 |
| `step_ms` | median(`step_total_ms`) — decode step 전체 wall time (sanity) |
| `accept_length_{mean,median,max}` | 실제 accept 분포 (real-mode 검증: 0 이 아니어야 함) |
| `committed_tokens_mean` | `accept_length_mean + 1` (bonus 포함) |
| `n_samples` | median 계산에 들어간 decode step 수 |
| `overhead_ms` | `step_ms − target_cost_ms − draft_cost_ms` 잔차 |

추가로 `<output_dir>/timing_logs/b{B}_s{S}.jsonl` 에 per-step raw JSONL 이 보존돼 사후 분해 분석 가능 (`eagle3_draft_ms`, `target_forward_ms`, `verify_total_ms`, `verify_overhead_ms`, `post_verify_ms`, `accept_lengths[]`, `committed_tokens[]`).

#### 2. Draft model per-token cost — `measure_draft_model_cost.py`

```bash
python3 simulation/scripts/measure_draft_model_cost.py \
    --model Qwen/Qwen3-0.6B \
    --workloads specbench,bfcl_v4,swebench \
    --num-draft-tokens 1,3,5 \
    --output simulation/results/qwen3_8b/draft_model_cost.json
```

- Draft LM (기본 Qwen3-0.6B) vanilla SGLang 서버 **1회** 기동. 오라클 패치 없음.
- 각 workload 에서 warmup call 후, `num_draft_tokens ∈ {1,3,5}` 각각 `max_tokens=N` 으로 `/v1/chat/completions` 호출, wall time 측정.
- N 개 증가에 따라 `per_token_ms = total_ms / n_actual_tokens` 감소 (fixed overhead 분산) → linear fit 하면 `per_token_ms(∞) ≈ asymptotic TPOT`.

**출력 `draft_model_cost.json`** per entry: `workload`, `num_draft_tokens`, `n_actual_tokens`, `total_ms`, `per_token_ms`.

#### 3. Suffix decoding cost — `measure_suffix_cost.py`

```bash
python3 simulation/scripts/measure_suffix_cost.py \
    --workloads specbench,bfcl_v4,swebench \
    --model Qwen/Qwen3-8B \
    --output simulation/results/qwen3_8b/suffix_cost.json
```

- GPU 불필요 — CPU only, `arctic_inference.SuffixDecodingCache.speculate(use_tree_spec=True)` 1회 호출 wall time.
- warmup prompt 로 cache 를 warm 한 뒤 measure prompt 로 1회 timed call (JIT 영향 배제용 1회 warmup speculate 선행).
- 1초 이내 완료.

**출력 `suffix_cost.json`** per entry: `workload`, `prompt_len`, `draft_size`, `speculate_ms`.

#### 파이프라인 연동 (out of scope)

세 스크립트 결과를 Stage 6 의 `--latency-config` JSON (`{vanilla_step_ms, target_forward_ms:{B:ms}, eagle3_draft_ms:{B:ms}, draft_lm_tpot_ms}`) 로 자동 변환하는 단계는 **별도 후속 작업**. Stage 6 는 `--latency-config` 를 optional 로 수용하므로, 파이프라인 자체는 이 단계 없이도 (stub latency로) 동작.

### 대안: 88+ Method Flat Simulation

Chain 기반 방법론 비교 (레거시 `run_oracle_sim.py`):

```bash
python3 -m simulation.evaluation.run_oracle_sim \
    --agent-results agent_results_eagle3.json \
    --output oracle_sim.json \
    --model Qwen/Qwen3-8B --print-summary
```

88+ 방법: standalone (EAGLE3/Suffix/DraftModel × depth), hybrid (threshold), sequential extension (⊕), tree extension (⊗), hybrid+extension 조합.

### Phase 1: 독립 분석 (선택)

Oracle 파이프라인과 별개로, suffix와 EAGLE3의 상호보완성을 직접 분석:

```bash
# EAGLE3 drafts 수집
python -m simulation.analysis.collect_eagle3_drafts \
    --server-url http://localhost:30000 --dataset humaneval \
    --output-dir results/eagle3_drafts

# Suffix candidates 동일 입력에서 수집
python -m simulation.analysis.collect_suffix_candidates \
    --eagle3-results results/eagle3_drafts \
    --output-dir results/suffix_candidates

# Case 1-4 비율 + P_accept / P_match
python -m simulation.analysis.compute_complementarity \
    --eagle3-results results/eagle3_drafts \
    --suffix-results results/suffix_candidates \
    --check-sequential \
    --output-dir results/complementarity

# Agreement score
python -m simulation.analysis.compute_agreement \
    --eagle3-results results/eagle3_drafts \
    --suffix-results results/suffix_candidates \
    --output-dir results/agreement

# Plots
python -m simulation.analysis.plot_results \
    --complementarity-file results/complementarity/complementarity.json \
    --agreement-file results/agreement/agreement.json \
    --output-dir results/plots
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

### GLM-4.7-Flash (31B MoE)

- Target model: `zai-org/GLM-4.7-Flash` (31B MoE, 30B-A3B)
- EAGLE-3 draft: `thoughtworks/GLM-4.7-Flash-Eagle3` (277MB)
- GPU: RTX 4090 x4 (tp-size 4)
- Speculative config: num_steps=3, topk=4, num_draft_tokens=16
- Stage 3 구성: 3a (Suffix) + 3c (MTP)

### Qwen3-8B (8B Dense)

- Target model: `Qwen/Qwen3-8B`
- EAGLE-3 draft: `Tengyunw/qwen3_8b_eagle3`
- Small draft LM: `Qwen/Qwen3-0.6B`
- GPU: RTX 4090 x1 (tp-size 1)
- Speculative config: num_steps=3, topk=4, num_draft_tokens=16
- Stage 3 구성: 3a (Suffix) + 3b (Draft Model)

### Common

- Batch size 1, greedy decoding (temperature=0)
- Max tree budget: 64 tokens (shared across all proposers/baselines)
- CUDA 12.2
- Lossless: tree verification uses standard rejection sampling, output distribution is identical to the target model

## SGLang Environment Variables

파이프라인 스크립트가 자동으로 export하는 SGLang 환경변수 (디버깅 참고용):

| 변수 | 설정 위치 | 역할 |
|---|---|---|
| `SGLANG_ORACLE_VANILLA=1` | Stage 1, latency 측정 | `oracle_patch.patch_eagle_worker_full` 훅 활성화 (timing instrumentation 기본) |
| `SGLANG_LATENCY_ONLY=1` | `measure_eagle3_cost.py` | `SGLANG_ORACLE_VANILLA=1` 과 함께 쓰면 accept_length=0 강제 생략 → real speculative decoding + timing 만 수집 |
| `SGLANG_ORACLE_REPLAY=<path>` | Stage 3c | NEXTN 서버가 trajectory.json 을 replay |
| `SGLANG_ORACLE_VERIFY_TRIES=<path>` | 선택적 | Suffix worker가 pre-built union trie로 speculation 대체 |
| `SGLANG_DRAFT_BUDGET=<N>` | (legacy) | `speculative_num_draft_tokens` runtime override — 현재 신규 측정 스크립트에서 미사용 |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | Docker | 긴 컨텍스트 허용 |
| `TORCHINDUCTOR_COMPILE_THREADS=1` | 모든 SGLang 호출 | torch.compile fork bomb 방지 |

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
- **EAGLE3 3D sweep**: `budget > topk + (steps-1)·topk² + 1` 조합은 SGLang의 `organize_draft_results` 에서 크래시 — sweep 스크립트에서 사전 필터링 권장
