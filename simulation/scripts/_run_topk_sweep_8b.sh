#!/usr/bin/env bash
# Inside container: measure Qwen3-8B latency across topk × steps × budget.
# Pins to a single GPU via CUDA_VISIBLE_DEVICES. Writes to
# /workspace/simulation/results/latency/qwen3_8b/eagle3_cost.json (resumable).
set -euo pipefail
cd /workspace

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export SGLANG_ORACLE_TIMING_LOG=/tmp/sglang_oracle_timing_8b.jsonl
export SGLANG_ORACLE_LOG=/tmp/sglang_oracle_vanilla_8b.jsonl
export PYTHONPATH=/workspace:${PYTHONPATH:-}

python3 simulation/scripts/measure_eagle3_cost.py \
    --model Qwen/Qwen3-8B \
    --draft-model AngelSlim/Qwen3-8B_eagle3 \
    --workloads specbench,bfcl_v4 \
    --budgets 4,8,16,32,64,128,256,512 \
    --steps 2,4,6,8 \
    --topks 4,8,16 \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --port 30000 \
    --output simulation/results/latency/qwen3_8b/eagle3_cost.json
