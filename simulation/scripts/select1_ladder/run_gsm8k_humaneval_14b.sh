#!/bin/bash
# GSM8K + HumanEval single-turn realized ladder on the 14B (Qwen3-14B + EAGLE3 = BALANCED regime,
# vs 27B MTP dominant). record(native)+select1(raw)+select1_oracle. 2-way (EAGLE3 + suffix).
# Blackwell GPU0: EAGLE3 AngelSlim draft is head_dim 80 -> force triton attention + pytorch sampling
# (project_blackwell_env), passed via --extra-args (REMAINDER, must be LAST). Env: N (default 80).
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }; done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
N="${N:-80}"
DIR=/workspace/simulation/results/chain_hybrid_perdepth/gsm8k_humaneval_14b_2way
mkdir -p "$DIR"
echo "GSM8K+HumanEval 14B ladder: N=$N CUDA_HOME=$CUDA_HOME"
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen3_14b --workload gsm8k_humaneval \
  --steps 16 --tp-size 1 --n-tasks "$N" --tail-max-tokens 0 \
  --arms record,select1,select1_oracle --port 31082 --mem-fraction-static 0.65 \
  --output "$DIR/run.json" \
  --extra-args --attention-backend triton --sampling-backend pytorch
echo "GSM8K_HUMANEVAL_14B_RC=$?"
