#!/bin/bash
# Diagnose 14B EAGLE3 low acceptance: native record on N=8, fp8 KV vs bf16(auto) KV. If bf16 jumps,
# fp8 KV cache is degrading the EAGLE3 draft's hidden-state predictions on Blackwell.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }; done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
DIR=/tmp/diag_14b_kv; mkdir -p "$DIR"
for KV in fp8_e5m2 auto; do
  echo "============ KV=$KV ============"
  python3 simulation/scripts/measure_chain_hybrid.py --preset qwen3_14b --workload gsm8k_humaneval \
    --steps 16 --tp-size 1 --n-tasks 8 --tail-max-tokens 0 --arms record \
    --kv-cache-dtype "$KV" --port 31082 --mem-fraction-static 0.65 \
    --output "$DIR/record_$KV.json" \
    --extra-args --attention-backend triton --sampling-backend pytorch 2>&1 | grep -aE "record: wall|accept_mean" | tail -2
  echo "KV=$KV DONE rc=$?"
done
echo "DIAG_DONE"
