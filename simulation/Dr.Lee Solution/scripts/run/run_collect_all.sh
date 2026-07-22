#!/usr/bin/env bash
# ONE-TIME GPU collection pass for the whole reproduction. After this, every slide
# is produced by CPU replay (replay_extension.py / analyze_perpos.py). Run inside
# sglang-bench on GPU0.
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for k in 1 2 4 8; do
  echo "=== [collect] multislot_k${k} $(date -u +%H:%M:%S)UTC ==="
  python3 scripts/capture_perpos.py \
    --dataset scripts/bench_prompts_multislot_k${k}.jsonl \
    --task multislot_k${k} --out results/perpos/multislot_k${k}.jsonl \
    --nwarm 12 --neval 8 --maxtok 96
done

echo "=== [collect] specbench 6-subtask (task-split, per-task=8, no warm) $(date -u +%H:%M:%S)UTC ==="
python3 scripts/capture_perpos.py \
  --dataset /workspace/data/specbench/dataset.jsonl \
  --task specbench --task-field subtask --per-task 8 \
  --out results/perpos/specbench.jsonl --maxtok 96

echo "=== COLLECT_ALL_DONE $(date -u +%H:%M:%S)UTC ==="
