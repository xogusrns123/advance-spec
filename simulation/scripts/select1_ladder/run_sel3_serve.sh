#!/bin/bash
# Served 3-way SEL3 (calibrated argmax) arm, replaying the existing 27B 3-way record
# trajectory so accept_length is comparable to the existing raw (5.001)/oracle (7.054).
# Env knobs: METHOD=gbm|logistic  NT=<n-tasks>  OUT=<run json name>
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_CHAIN_HYBRID_DFLASH_AUX=1
DIR=/workspace/simulation/results/chain_hybrid_perdepth/qwen35_27b_3way_sel3
export SGLANG_CHAIN_HYBRID_SEL3="$DIR/sel3_bundle.json"
export SGLANG_CHAIN_HYBRID_SEL3_METHOD="${METHOD:-gbm}"
NT="${NT:-2}"; OUT="${OUT:-run_smoke_${SGLANG_CHAIN_HYBRID_SEL3_METHOD}.json}"
echo "SEL3 method=$SGLANG_CHAIN_HYBRID_SEL3_METHOD NT=$NT OUT=$OUT CUDA_HOME=$CUDA_HOME"
python3 simulation/scripts/measure_chain_hybrid.py \
  --preset qwen35_27b_mtp --workload bfcl_v4 --include-category web_search \
  --steps 16 --tp-size 1 --n-tasks "$NT" --tail-max-tokens 0 \
  --replay-existing --arms select1 --port 31082 --mem-fraction-static 0.65 \
  --output "$DIR/$OUT"
echo "MEASURE_RC=$?"
