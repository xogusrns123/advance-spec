#!/bin/bash
# GSM8K (math) + HumanEval (code) single-turn realized ladder via specbench_agent.
# record(native) + select1(raw) + select1_oracle -> per-subtask native/raw/oracle MAT + suffix-win.
# 2-way (MTP + suffix), no DFLASH_AUX. Env: N (default 80 = 40/subtask).
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }; done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
N="${N:-80}"
DIR=/workspace/simulation/results/chain_hybrid_perdepth/gsm8k_humaneval_2way
mkdir -p "$DIR"
echo "GSM8K+HumanEval ladder: N=$N CUDA_HOME=$CUDA_HOME"
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen35_27b_mtp --workload gsm8k_humaneval \
  --steps 16 --tp-size 1 --n-tasks "$N" --tail-max-tokens 0 \
  --arms record,select1,select1_oracle --port 31082 --mem-fraction-static 0.65 \
  --output "$DIR/run.json"
echo "GSM8K_HUMANEVAL_RC=$?"
