#!/bin/bash
# Serve the TRANSPARENT hand-rule arm (select1_handrule: suffix iff suffix_p > A*eagle_p + B)
# as a new realized arm on the 2-way cells, pinned replay (same eval trajectories as the
# existing realized arms). 14B: A=0.6 B=0.25 (tail 64). 27B: A=1.1 B=0.35 (tail 0).
# GPU0-only; waits for GPU0 to be free first.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }; done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
RB=simulation/results/chain_hybrid_perdepth
gpu_free(){ for i in $(seq 1 240); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  [ "${m:-99999}" -lt 15000 ] && return 0; echo "wait GPU0 ${m}MiB $(date -u +%H:%MZ)"; sleep 30
done; }

echo "=== 14B handrule  A=0.6 B=0.25  tail=64 $(date -u) ==="
gpu_free
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen3_14b --workload bfcl_v4 \
  --steps 16 --tp-size 1 --n-tasks 800 --train-n-tasks 0 --tail-max-tokens 64 \
  --replay-existing --pin-trajectory --arms select1_handrule \
  --handrule-a 0.6 --handrule-b 0.25 \
  --port 31083 --mem-fraction-static 0.65 \
  --output "$RB/qwen3_14b_ar/run_handrule.json" \
  --extra-args --attention-backend triton --sampling-backend pytorch
echo "14B_RC=$?  $(date -u)"
gpu_free; sleep 15

echo "=== 27B handrule  A=1.1 B=0.35  tail=0 $(date -u) ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen35_27b_mtp --workload bfcl_v4 \
  --steps 16 --tp-size 1 --n-tasks 800 --train-n-tasks 0 --tail-max-tokens 0 \
  --replay-existing --pin-trajectory --arms select1_handrule \
  --handrule-a 1.1 --handrule-b 0.35 \
  --port 31083 --mem-fraction-static 0.65 \
  --output "$RB/qwen35_27b_ar/run_handrule.json"
echo "27B_RC=$?  $(date -u)"
echo "HANDRULE_DONE $(date -u)"
