#!/usr/bin/env bash
# ONE-TIME GPU collection pass for the FULL-TRAJECTORY workloads (bfcl_v4_full_traj
# + swebench_full_traj), then CPU replay + figures. Run inside sglang-bench on GPU0:
#   docker exec -i sglang-bench bash "/workspace/simulation/Dr.Lee Solution/scripts/run_collect_fulltraj.sh"
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [collect] bfcl_v4_full $(date -u +%H:%M:%S)UTC ==="
python3 scripts/capture_traj.py \
  --gt-tokens /workspace/simulation/results/bfcl_v4_full_traj/qwen35_27b_dflash/gt_tokens.jsonl \
  --conv-map results/perpos_bfcl_full/conv_map.json \
  --task bfcl_v4_full \
  --out results/perpos_bfcl_full/bfcl_v4_full.jsonl \
  > results/perpos_bfcl_full/capture.log 2>&1 || { echo "BFCL_CAPTURE_FAILED"; exit 1; }

echo "=== [collect] swebench $(date -u +%H:%M:%S)UTC ==="
python3 scripts/capture_traj.py \
  --gt-tokens /workspace/simulation/results/swebench_full_traj/qwen35_27b_dflash/gt_tokens.jsonl \
  --conv-map results/perpos_swebench/conv_map.json \
  --task swebench \
  --out results/perpos_swebench/swebench.jsonl \
  > results/perpos_swebench/capture.log 2>&1 || { echo "SWE_CAPTURE_FAILED"; exit 1; }

echo "=== [replay+fig] bfcl_v4_full $(date -u +%H:%M:%S)UTC ==="
python3 scripts/plot_mat_traj.py \
  --record results/perpos_bfcl_full/bfcl_v4_full.jsonl \
  --title "BFCLv4 full-traj (memory+web_search, thinking-ON)" \
  --outname mat_bfcl_full.png \
  > results/perpos_bfcl_full/plot.log 2>&1 || echo "BFCL_PLOT_FAILED"

echo "=== [replay+fig] swebench $(date -u +%H:%M:%S)UTC ==="
python3 scripts/plot_mat_traj.py \
  --record results/perpos_swebench/swebench.jsonl \
  --title "SWE-bench Verified (mini-swe-agent, thinking-OFF)" \
  --outname mat_swebench.png \
  > results/perpos_swebench/plot.log 2>&1 || echo "SWE_PLOT_FAILED"

echo "=== COLLECT_FULLTRAJ_DONE $(date -u +%H:%M:%S)UTC ==="
