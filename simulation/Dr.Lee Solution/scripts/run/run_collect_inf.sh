#!/usr/bin/env bash
# Re-collect the multislot (Dr. Lee) workload with MAXTOK effectively infinite
# (generate to EOS; cap 4096 as a runaway guard) -> untruncated outputs.
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for k in 0 1 2 4 8; do
  echo "=== [collect-inf] multislot_k${k} maxtok=4096 $(date -u +%H:%M:%S)UTC ==="
  python3 scripts/capture_perpos.py \
    --dataset scripts/bench_prompts_multislot_k${k}.jsonl \
    --task multislot_k${k} --out results/perpos_inf/multislot_k${k}.jsonl \
    --nwarm 12 --neval 8 --maxtok 4096
done
echo "=== COLLECT_INF_DONE $(date -u +%H:%M:%S)UTC ==="
