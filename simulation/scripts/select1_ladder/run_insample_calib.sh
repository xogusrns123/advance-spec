#!/bin/bash
# Serve in-sample calib arms (maps fit on the EVAL slice) on the SAME eval tasks (replay+pin).
# Usage: PRESET=qwen35_27b_mtp DST=<insample dir> bash run_insample_calib.sh
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
PRESET="${PRESET:?}"; DST="${DST:?}"; PORT="${PORT:-31082}"; TAIL="${TAIL:-0}"
ARMS="${ARMS:-select1_calib_histogram,select1_calib_isotonic,select1_calib_logistic,select1_calib_beta}"
OUT="${OUT:-run_insample_calib.json}"
python3 simulation/scripts/measure_chain_hybrid.py \
  --preset "$PRESET" --workload bfcl_v4 --include-category web_search \
  --steps 16 --tp-size 1 --train-n-tasks 30 --n-tasks 20 --tail-max-tokens "$TAIL" \
  --mem-fraction-static 0.65 --port "$PORT" --replay-existing --pin-trajectory --skip-calib-fit \
  --arms "$ARMS" \
  --output "$DST/$OUT"
echo "INSAMPLE_CALIB_RC=$?"
