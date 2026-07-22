#!/bin/bash
# Quick SWE-Bench smoke: official mini-swe-agent, record arm, N=1, short max-iter. Validates that
# mini-swe-agent drives multi-turn code-editing generation against the chain-hybrid server.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }; done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
N="${N:-1}"; MI="${MI:-8}"
DIR=/workspace/simulation/results/chain_hybrid_perdepth/swe_mini_smoke
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen35_27b_mtp --workload swebench \
  --steps 16 --tp-size 1 --n-tasks "$N" --tail-max-tokens 0 --max-iterations "$MI" \
  --mem-fraction-static 0.65 --port 31082 --arms record --output "$DIR/run.json"
echo "SWE_SMOKE_RC=$?"
NT=$(python3 -c "import json;d=json.load(open('$DIR/agent_results_record.json'));print(d['questions'][0].get('num_turns',0))" 2>/dev/null || echo 0)
echo "SWE_SMOKE_num_turns=$NT"
