#!/bin/bash
# O4 joint-discriminator pipeline (logistic + beta), real serving.
#   STEP D  train-oracle on the 30 TRAIN tasks (offset 0)  -> labeled decisions
#   STEP E  fit disc_logistic.json + disc_beta.json into the eval dir
#   STEP G  eval on the 20 TEST tasks (offset 30), --replay-all, with
#           raw + oracle in the SAME run for a within-run-fair MAT comparison
#
# Waits for GPU0 to free first (so it can queue behind the 27B calib run without
# clobbering it). Run inside the sglang-bench container as root.
#
# Usage:  run_o4_disc.sh <PRESET> <PORT> <DIR>
#   e.g.  run_o4_disc.sh qwen3_14b 30021 qwen3_14b
set -uo pipefail
PRESET="${1:?preset}"; PORT="${2:?port}"; DIR="${3:?dir}"
cd /workspace
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
COMMON="--workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1 --tail-max-tokens 0"
EXTRA="--extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch"
BASE="simulation/results/o4_perdepth"

echo "=== waiting for GPU0 to free (<15GB used) ==="
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  echo "  GPU0 used=${USED} MiB  $(date +%H:%M:%S)"
  [ "${USED:-99999}" -lt 15000 ] && break
  sleep 120
done
sleep 20
echo "GPU0 free; starting $PRESET disc pipeline"

echo "=== STEP D: train-oracle (record+oracle, 30 train tasks, offset 0) ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
  --n-tasks 30 --train-n-tasks 0 --port "$PORT" \
  --arms record,select1_oracle \
  --output "$BASE/${DIR}_disc_train/run.json" $EXTRA
echo "STEP_D_RC=$?"

TRAIN_LOG="$BASE/${DIR}_disc_train/decisions_select1_oracle.jsonl"
if [ ! -s "$TRAIN_LOG" ]; then
  echo "ABORT: train-oracle log missing/empty ($TRAIN_LOG) — STEP D failed"; exit 1
fi
echo "=== STEP E: fit logistic+beta discriminators from train-oracle log ==="
python3 simulation/scripts/fit_chain_hybrid_discriminator.py \
  --oracle-log "$TRAIN_LOG" --out-dir "$BASE/${DIR}_disc"
echo "STEP_E_RC=$?"

if [ ! -s "$BASE/${DIR}_disc/disc_logistic.json" ] || \
   [ ! -s "$BASE/${DIR}_disc/disc_beta.json" ]; then
  echo "ABORT: disc maps missing — STEP E failed"; exit 1
fi
echo "=== STEP G: eval on test (offset 30), replay-all, disc + raw + oracle ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
  --train-n-tasks 30 --n-tasks 20 --port "$PORT" --replay-all \
  --arms record,select1,select1_oracle,select1_disc_logistic,select1_disc_beta \
  --output "$BASE/${DIR}_disc/run.json" $EXTRA
echo "STEP_G_RC=$?"

echo "O4_DISC_${DIR}_DONE"
