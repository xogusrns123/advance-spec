#!/bin/bash
# SWE-Bench(-Verified) select-1 capture (2-way: model + suffix). Agentic code-editing =
# very copy-heavy -> tests whether the model-free(suffix) proposer wins much more and shifts
# the regime. record + select1_oracle; repos auto-clone into --repos-dir at run time (NETWORK
# needed). Heavy + slow (agentic, max-iter 15) -> start with a small N.
# Env: PRESET (default qwen35_27b_mtp), N (default 10 instances).
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
PRESET="${PRESET:-qwen35_27b_mtp}"; N="${N:-5}"
mkdir -p data/swebench/repos
DIR="/workspace/simulation/results/chain_hybrid_perdepth/${DIRNAME:-swebench_${PRESET}_2way}"
mkdir -p "$DIR"
echo "SWE-Bench capture: preset=$PRESET N=$N dir=$DIR (repos auto-clone to data/swebench/repos)"
python3 simulation/scripts/measure_chain_hybrid.py \
  --preset "$PRESET" --workload swebench \
  --steps 16 --tp-size 1 --n-tasks "$N" --tail-max-tokens 0 --max-iterations 15 \
  --arms record,select1,select1_oracle --port 31082 --mem-fraction-static 0.65 \
  --output "$DIR/run.json"
echo "SWEBENCH_CAPTURE_RC=$?"
