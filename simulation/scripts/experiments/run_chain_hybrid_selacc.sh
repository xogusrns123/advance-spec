#!/bin/bash
# Real-serving SELECTION-ACCURACY measurement: re-serve raw + 4 cond-trained calib
# methods + disc (our-Bayes joint, Bayes-ceiling proxy), all PINNED, with the patch
# now logging oracle_hit on every arm. Each arm's own served picks get GT labels ->
# decisive selection accuracy is read straight from the logs (no estimation).
# Usage: run_chain_hybrid_selacc.sh <CELL> <PORT>   (e.g. qwen3_14b_ar 30021)
set -uo pipefail
CELL="${1:?cell}"; PORT="${2:-30021}"
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  [ -x "$c/bin/nvcc" ] && export CUDA_HOME="$c" && break; done
export PATH="$CUDA_HOME/bin:$PATH"
D="simulation/results/chain_hybrid_perdepth/$CELL"
TRAIN="simulation/results/chain_hybrid_perdepth/qwen3_14b_tp_train"
echo "using CUDA_HOME=$CUDA_HOME ; cell=$D"
# 1. place cond-trained calib maps at top level (the calib arms read $D/calib_pp_*.json)
cp "$D"/calib_cond-trained/calib_pp_*.json "$D"/
# 2. fit our-Bayes disc (joint, accept-conditioned + depth) into the cell -> disc_beta.json
python3 simulation/scripts/fit_chain_hybrid_discriminator.py \
  --oracle-log "$TRAIN/decisions_select1_oracle.jsonl" --out-dir "$D" || { echo "DISC FIT FAILED"; exit 1; }
# 3. serve raw + 4 calibs + disc, pinned (oracle_hit logged on every arm)
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen3_14b \
  --workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1 \
  --train-n-tasks 30 --n-tasks 20 --port "$PORT" --tail-max-tokens 64 \
  --replay-existing --skip-calib-fit --pin-trajectory \
  --arms select1,select1_calib_histogram,select1_calib_isotonic,select1_calib_logistic,select1_calib_beta,select1_disc_beta \
  --output "$D/run_selacc.json" \
  --extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch
echo "SELACC_RC=$?"
echo "SELACC_DONE $CELL"
