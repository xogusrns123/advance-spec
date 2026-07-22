#!/bin/bash
# REALIZED chain: after the current SpecBench(record+oracle) finishes, add the served
# select1(raw) arm (realized raw per subtask), then run SWE-Bench(record,select1,oracle).
# All served/realized. GPU0-only: only launches each step when GPU0 mem < 15GB (won't kill
# foreign procs). Runs inside docker (survives host-session restarts).
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }
done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
SD=/workspace/simulation/results/chain_hybrid_perdepth/specbench_qwen35_27b_mtp_2way
gpu_free(){ for i in $(seq 1 240); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  [ "${m:-99999}" -lt 15000 ] && return 0; echo "C2 $(date -u +%H:%MZ) GPU0 busy ${m}MiB, wait"; sleep 30
done; }

echo "C2 $(date -u): waiting for current SpecBench(record+oracle) to finish..."
until ! pgrep -f "measure_chain_hybrid.py.*--workload specbench" >/dev/null 2>&1; do sleep 30; done
gpu_free; sleep 20
echo "C2 $(date -u): SpecBench select1(raw) replay (realized raw)"
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen35_27b_mtp --workload specbench \
  --steps 16 --tp-size 1 --n-tasks 120 --tail-max-tokens 0 --replay-existing --arms select1 \
  --port 31082 --mem-fraction-static 0.65 --output "$SD/run_raw.json"
echo "C2 $(date -u): SpecBench raw done rc=$?"
gpu_free; sleep 20
echo "C2 $(date -u): launching SWE-Bench (record,select1,select1_oracle)"
PRESET=qwen35_27b_mtp N=5 bash simulation/scripts/select1_ladder/run_swebench_capture.sh
echo "C2 $(date -u): ALL DONE"
