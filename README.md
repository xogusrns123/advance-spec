# Hybrid Speculative Decoding Research Codebase

Modular speculative decoding research framework built on SGLang. Four
pluggable proposers, full step-level tracing, and a reproducible **oracle
simulation pipeline** for upper-bound analysis.

## Background

Speculative decoding accelerates LLM inference by having a lightweight
draft model predict multiple tokens ahead, which the target model
verifies in a single forward pass. This project provides a **modular
research codebase** with four proposers and full instrumentation:

- **MTP** (Multi-Token Prediction): Dedicated prediction heads.
- **Small Draft Model**: A smaller LM generates candidates autoregressively.
- **EAGLE-3**: Autoregressive draft head using target hidden states.
  Generates a tree of candidates via top-k branching at each depth.
- **SuffixDecoding**: Model-free suffix tree pattern matching via
  ArcticInference C++ backend (~20us/token CPU). Effective on
  repetitive workloads (code, SQL, agentic tool calls).

All four share a common `DraftTree` interface in `hybrid_spec_decoding/`,
making them interchangeable in benchmarks and traces.

## Two subsystems

### 1. `hybrid_spec_decoding/` — research runtime library

Online runtime for end-to-end benchmarking. Provides the proposers, tree
fusion algorithms (RASD parallel merge, sequential extension), and
SGLang integration for hybrid speculative decoding. See
`hybrid_spec_decoding/benchmarks/` for entry points.

### 2. `simulation/` — oracle simulation pipeline

Offline pipeline for **upper-bound analysis** of speculative decoding
methods. The simulator computes per-step accept-length and step-cost
under different proposers/budgets without re-running the LLM each time.

The pipeline is **3 stages** (RR-only since commit `d1e8247` removed
the legacy Stage 1-6 sweep mode):

| Stage | Module | Output |
|---|---|---|
| 1 | `simulation/scripts/run_experiment.py` (RR mode) | `agent_results_eagle3.json` (per-step EAGLE3 draft + verify logs + full pool capture) |
| 2 | `simulation/pipeline/collect_draft_model.py` (optional, latency 측정용) | `draft_model_drafts.jsonl` |
| 3 | `simulation/evaluation/run_tree_oracle_sim.py` | `tree_oracle_sim.json` (mat / accept_rate / speedup_real) |

Detailed docs in `simulation/docs/00_OVERVIEW.md`.

## Quick start — Oracle simulation (RR pipeline)

### Stage 1: RR collection

Single GPU (mango3 / qwen3_14b):
```bash
docker exec -u root -d sglang-bench bash -c \
  "cd /workspace && python3 simulation/scripts/run_experiment.py \
    simulation/config/rr_qwen3_14b.yaml > /tmp/rr_stage1.log 2>&1"
```

4-shard parallel (mango1 / qwen3_8b, 4 GPUs):
```bash
docker exec -u root -d sglang-bench bash -c \
  "cd /workspace && python3 simulation/scripts/run_experiment.py \
    simulation/config/mango1.yaml > /tmp/rr_stage1.log 2>&1"
```

Monitor:
```bash
tail -F /tmp/rr_stage1.log | grep -E 'iter [0-9]+\]|ok  done=|ERROR|FAIL'
```

### Stage 3: Reslice sweep

```bash
python3 simulation/scripts/experiments/run_reslice_sweep.py \
    --workload swebench_verified \
    --reslices s2k16,s4k16,s6k16 \
    --budgets 8,16,32,64 \
    --in-docker
```

### Latency measurement (사전 1회)

```bash
bash simulation/scripts/experiments/remeasure_latency.sh
# → simulation/config/latency/<preset>.json
```

### Analysis

`simulation/notebooks/compare_methods.ipynb` 등이 결과 시각화.

## Project structure

```
hybrid_spec_decoding/           — Core research runtime
  proposers/                    Four pluggable draft proposers
  tree_fusion/                  Shared DraftTree + fusion algorithms
  suffix_decoding/              ArcticInference SuffixDecodingCache wrapper
  sglang_integration/           SGLang plugin integration
  tracing/                      Per-step instrumentation
  benchmarks/                   Unified benchmark CLIs

simulation/                     — Oracle simulation pipeline (RR-only)
  scripts/
    run_experiment.py           Stage 1 RR collection entry
    measure_*_cost.py           Latency measurement (eagle3 / draft_lm / suffix)
    compile_latency_config.py   Combine measurements → latency_config.json
    experiments/
      remeasure_latency.sh      Wrapper for measure_*_cost + compile
      run_reslice_sweep.py      Stage 3 reslice sweep orchestrator
      analyze_sweep.py          Result analysis
      data_prep/                Dataset prep scripts

  agents/                       Per-benchmark agent runners
    bfcl_agent.py               BFCLv3 (deprecated workload)
    bfcl_v4_agent.py            BFCLv4 (web_search / memory)
    specbench_agent.py          SpecBench / MT-Bench / LongBench
    swebench_agent.py           SWE-Bench (full / sweagent / minisweagent tool styles)
    tools/                      Tool implementations

  oracle/                       SGLang patches
    install_hook.py             Tier-1 disk patches (idempotent)
    oracle_patch.py             Tier-2 runtime monkey-patches

  pipeline/                     Data collection helpers
    save_results.py             Atomic write + checkpoint
    _agent_io.py                Oracle log → per-call tensors
    collect_draft_model.py      Stage 2 (draft LM proposals)
    assemble_records.py         Stage 3 record assembly
    pool_reslicer.py            Captured full pool → (s', k') sub-tree

  evaluation/
    run_tree_oracle_sim.py      Stage 3 main entry
    tree_knapsack.py            Greedy walk
    run_side_suffix_trajectory.py  Side analysis tool

  config/
    rr_qwen3_14b.yaml           Single-GPU RR config (mango3)
    mango1.yaml                 4-shard RR config (mango1)
    latency/                    Per-preset latency_config.json files

  results/                      Output (gitignored)
  docs/                         Detailed Korean docs (00_OVERVIEW + 9 stage docs)
  notebooks/                    Result visualization
```

## Detailed docs

| File | Topic |
|---|---|
| `simulation/docs/00_OVERVIEW.md` | Top-level pipeline overview, env vars, validation checklist |
| `simulation/docs/01_stage1_rr_collection.md` | RR mode internals, shard split, full pool capture |
| `simulation/docs/02_stage1_agents.md` | Per-workload agent details |
| `simulation/docs/03_stage1_tools_and_io.md` | BFCL/SWE tool patches, save_results, oracle log schema |
| `simulation/docs/04_stage2_draft_model.md` | Draft LM collection (`collect_draft_model.py`) |
| `simulation/docs/05_stage3_simulator_core.md` | Stage 3 simulator core, methods, latency model |
| `simulation/docs/06_tree_knapsack.md` | Greedy walk implementation |
| `simulation/docs/07_side_suffix_trajectory.md` | Side suffix trajectory tool |
| `simulation/docs/08_sglang_patches.md` | SGLang patches reference |
| `simulation/docs/09_pool_reslicer.md` | Pool reslicer algorithm |

## Setup

Standard SGLang + ArcticInference + LangChain dependencies. See
`Dockerfile` and `docker-compose.yml` for the canonical environment.
The pipeline assumes execution inside the `sglang-bench` container
unless otherwise noted.

Environment requirements:
- NVIDIA GPU (single or multi). Tested on RTX 4090, H100.
- ≥ 64GB system RAM (more for swebench_verified — capture JSON 가
  ~100GB RSS 로 로딩됨; `feedback_pause_rr_for_big_sims` 메모리 참조).
- ≥ 32GB swap (큰 sim 안정성).

HuggingFace token (`HF_TOKEN`) 필요시 `/workspace/.env` 에 둘 것
(gated 모델 다운로드 용).
