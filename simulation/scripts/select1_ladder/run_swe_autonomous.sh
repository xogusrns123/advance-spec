#!/bin/bash
# AUTONOMOUS overnight SWE-Bench collection via OFFICIAL mini-swe-agent. Goal: basic collection
# done by 10:00 KST. Steps: wait for SpecBench bayes to finish -> kill OUR calib measure (skip
# logistic, free GPU sooner) -> smoke N=1 (validate mini-swe-agent generates multi-turn code edit)
# -> if OK, full N=2 (record + select1 + select1_oracle = realized ladder on real code editing).
# RULES: only kills OUR processes (run_calib measure + our port-31082 server); launches only when
# GPU0 mem<15GB (never onto a foreign jhpark676/um3maru process); all arms realized/served.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }; done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
B=/workspace/simulation/results/chain_hybrid_perdepth
SD=$B/specbench_qwen35_27b_mtp_2way
gpu_free(){ for i in $(seq 1 600); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null|tr -d ' '); [ "${m:-99999}" -lt 15000 ] && return 0; echo "AUTO $(date -u +%H:%MZ) GPU0 busy ${m}MiB, wait"; sleep 30; done; }
meas(){ python3 simulation/scripts/measure_chain_hybrid.py --preset qwen35_27b_mtp --workload swebench --steps 16 --tp-size 1 --tail-max-tokens 0 --mem-fraction-static 0.65 --port 31082 "$@"; }

echo "AUTO $(date -u): waiting for SpecBench bayes to finish (run_calib.json select1_bayes)..."
until python3 -c "import json,sys;a=(json.load(open('$SD/run_calib.json')).get('arms') or {});sys.exit(0 if a.get('select1_bayes',{}).get('accept_length_mean') else 1)" 2>/dev/null; do sleep 30; done
echo "AUTO $(date -u): bayes done -> killing OUR calib measure (skip logistic) to free GPU for SWE-Bench"
pkill -9 -f "measure_chain_hybrid.py.*run_calib.json" 2>/dev/null || true
pkill -9 -f "launch_server.*--port 31082" 2>/dev/null || true
sleep 10; gpu_free; sleep 15

echo "AUTO $(date -u): SWE-Bench SMOKE N=1 (record, max-iter 8) to validate mini-swe-agent generation"
meas --n-tasks 1 --max-iterations 8 --arms record --output $B/swe_mini_smoke/run.json
echo "AUTO_SMOKE_RC=$?"
NT=$(python3 -c "import json;d=json.load(open('$B/swe_mini_smoke/agent_results_record.json'));print(d['questions'][0].get('num_turns',0))" 2>/dev/null || echo 0)
echo "AUTO SMOKE num_turns=$NT $(date -u)"
if [ "${NT:-0}" -ge 3 ]; then
  echo "AUTO $(date -u): SMOKE OK ($NT turns) -> FULL N=2 (record,select1,select1_oracle, max-iter 10)"
  gpu_free; sleep 10
  meas --n-tasks 2 --max-iterations 10 --arms record,select1,select1_oracle --output $B/swe_mini/run.json
  echo "AUTO_FULL_RC=$?"
  echo "AUTO ALL DONE $(date -u)"
else
  echo "AUTO SMOKE FAILED (num_turns=$NT) — halting for debug $(date -u)"
fi
