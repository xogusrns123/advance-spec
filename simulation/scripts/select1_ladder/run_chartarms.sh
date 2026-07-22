#!/bin/bash
# Single-proposer arms (baseline=model-alone, suffix=suffix-alone) for the 6-bar per-subtask MAT
# charts, realized. 14B FIRST (user priority), then 27B. 14B also redoes select1_calib_beta (cond-
# trained beta) since its earlier run was killed; 27B calib_beta already done (run_calib_betaCT.json).
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }; done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
RB=simulation/results/chain_hybrid_perdepth

# ---- 14B gsm8k+he: baseline, suffix, calib_beta (cond-trained beta map placed) ----
D=$RB/gsm8k_humaneval_14b_2way
cp "$D/calib_betaCT/calib_pp_beta.json" "$D/calib_pp_beta.json"
echo "================= 14B gsm8k+he: baseline,suffix,calib_beta ================="
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen3_14b --workload gsm8k_humaneval \
  --steps 16 --tp-size 1 --n-tasks 80 --tail-max-tokens 0 --skip-calib-fit \
  --arms baseline,suffix,select1_calib_beta --port 31082 --mem-fraction-static 0.65 \
  --output "$D/run_chartarms.json" \
  --extra-args --attention-backend triton --sampling-backend pytorch
echo "14B_GSM_RC=$?"

# ---- 27B gsm8k+he: baseline, suffix (calib_beta=5.731 already in run_calib_betaCT.json) ----
D=$RB/gsm8k_humaneval_2way
echo "================= 27B gsm8k+he: baseline,suffix ================="
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen35_27b_mtp --workload gsm8k_humaneval \
  --steps 16 --tp-size 1 --n-tasks 80 --tail-max-tokens 0 \
  --arms baseline,suffix --port 31082 --mem-fraction-static 0.65 \
  --output "$D/run_chartarms.json"
echo "27B_GSM_RC=$?"
echo "CHARTARMS_DONE"
