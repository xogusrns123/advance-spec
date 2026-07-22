#!/bin/bash
# REALIZED IN-SAMPLE for 27B 2-way bayes/mono: GBM fit on EVAL oracle log (30-49),
# re-served on the SAME eval tasks (replay + pin to the same 89 trajectories as the
# held-out run). Compare realized MAT to held-out bayes 6.303 / mono 6.324.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true   # 2-way, no dflash
DST=/workspace/simulation/results/chain_hybrid_perdepth/qwen35_27b_ar_insample
python3 simulation/scripts/measure_chain_hybrid.py \
  --preset qwen35_27b_mtp --workload bfcl_v4 --include-category web_search \
  --steps 16 --tp-size 1 --train-n-tasks 30 --n-tasks 20 --tail-max-tokens 0 --mem-fraction-static 0.65 \
  --port 31082 --replay-existing --pin-trajectory --skip-calib-fit \
  --arms select1_bayes,select1_mono --output "$DST/run_insample.json"
echo "INSAMPLE_REALIZED_RC=$?"
