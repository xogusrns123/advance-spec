#!/bin/bash
# 27B MTP: fit our-Bayes disc from the train-oracle log + serve select1_disc_beta
# PINNED (oracle_hit logged via the patch). raw/calib/oracle already served by the
# perdepth Stage-1 run; this only ADDS the disc (Bayes-ceiling proxy) arm.
# MTP knobs: tail=0, ndt default (steps+1), port 30022.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  [ -x "$c/bin/nvcc" ] && export CUDA_HOME="$c" && break; done
export PATH="$CUDA_HOME/bin:$PATH"
D=simulation/results/chain_hybrid_perdepth/qwen35_27b_ar
TRAIN=simulation/results/chain_hybrid_perdepth/qwen35_27b_ar_train
[ -s "$TRAIN/decisions_select1_oracle.jsonl" ] || { echo "ABORT: no 27B train oracle log"; exit 1; }
[ -s "$D/agent_results_record.json" ] || { echo "ABORT: no eval record (run perdepth first)"; exit 1; }
[ -s "$D/gt_tokens.jsonl" ] || { echo "ABORT: no gt_tokens"; exit 1; }
echo "using CUDA_HOME=$CUDA_HOME"
python3 simulation/scripts/fit_chain_hybrid_discriminator.py \
  --oracle-log "$TRAIN/decisions_select1_oracle.jsonl" --out-dir "$D" || { echo "DISC FIT FAILED"; exit 1; }
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen35_27b_mtp \
  --workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1 \
  --train-n-tasks 30 --n-tasks 20 --port 30022 --tail-max-tokens 0 \
  --replay-existing --skip-calib-fit --pin-trajectory \
  --arms select1_disc_beta --output "$D/run_disc.json" \
  --extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch
echo "DISC27B_RC=$?"
echo "DISC27B_DONE"
