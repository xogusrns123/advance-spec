#!/usr/bin/env bash
# Inside container: measure Qwen3-14B latency across topk × steps × budget.
# Uses 2 GPUs with TP=2. Writes to
# /workspace/simulation/results/latency/qwen3_14b/eagle3_cost.json (resumable).
set -euo pipefail
cd /workspace

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export SGLANG_ORACLE_TIMING_LOG=/tmp/sglang_oracle_timing_14b.jsonl
export SGLANG_ORACLE_LOG=/tmp/sglang_oracle_vanilla_14b.jsonl
export PYTHONPATH=/workspace:${PYTHONPATH:-}

python3 simulation/scripts/measure_eagle3_cost.py \
    --model Qwen/Qwen3-14B \
    --draft-model AngelSlim/Qwen3-14B_eagle3 \
    --workloads specbench,bfcl_v4 \
    --budgets 4,8,16,32,64,128,256,512 \
    --steps 2,4,6,8 \
    --topks 4,8,16 \
    --tp-size 2 \
    --mem-fraction-static 0.85 \
    --port 30001 \
    --output simulation/results/latency/qwen3_14b/eagle3_cost.json
