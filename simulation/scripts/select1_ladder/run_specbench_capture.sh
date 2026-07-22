#!/bin/bash
# SpecBench select-1 capture (2-way: MTP/EAGLE3 + suffix). record + select1_oracle on the
# interleaved (round-robin by 6 subtasks) dataset -> per-subtask oracle ceiling / raw selacc /
# suffix-win-rate / regime, computed offline from decisions_select1_oracle.jsonl (rid->subtask).
# Env: PRESET (default qwen35_27b_mtp), N (default 120 = 20/subtask), DIRNAME.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true   # 2-way only
PRESET="${PRESET:-qwen35_27b_mtp}"; N="${N:-120}"
DIR="/workspace/simulation/results/chain_hybrid_perdepth/${DIRNAME:-specbench_${PRESET}_2way}"
mkdir -p "$DIR"
echo "SpecBench capture: preset=$PRESET N=$N dir=$DIR CUDA_HOME=$CUDA_HOME"
python3 simulation/scripts/measure_chain_hybrid.py \
  --preset "$PRESET" --workload specbench \
  --steps 16 --tp-size 1 --n-tasks "$N" --tail-max-tokens 0 \
  --arms record,select1_oracle --port 31082 --mem-fraction-static 0.65 \
  --output "$DIR/run.json"
echo "SPECBENCH_CAPTURE_RC=$?"
