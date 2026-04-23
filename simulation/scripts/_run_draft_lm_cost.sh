#!/usr/bin/env bash
# Inside container: measure Qwen3-0.6B draft-LM per-token latency.
# Runs on a single idle GPU (no topk/steps axis — small-model TPOT only).
set -euo pipefail
cd /workspace

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export PYTHONPATH=/workspace:${PYTHONPATH:-}

python3 simulation/scripts/measure_draft_model_cost.py \
    --model Qwen/Qwen3-0.6B \
    --workloads specbench,bfcl_v4 \
    --num-draft-tokens 1,3,5 \
    --port 30010 \
    --output simulation/results/latency/draft_model_cost.json
