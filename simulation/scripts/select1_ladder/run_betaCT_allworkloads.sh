#!/bin/bash
# Unify calibration on BETA (cond-trained: token==gt + accept-conditioned) across ALL cross-domain
# cells, realized in-sample. For each cell: (1) re-fit beta on the cell's EVAL oracle log into
# calib_betaCT/ (non-destructive), (2) place it at the served calib_pp_beta.json, (3) serve
# select1_calib_beta -> run_calib_betaCT.json. 27B cells first (faster), then 14B.
# Cell fields: dir | preset | measure-workload-args | server-extra(after --extra-args; empty=none)
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }; done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
RB=simulation/results/chain_hybrid_perdepth

CELLS=(
  "gsm8k_humaneval_2way|qwen35_27b_mtp|--workload gsm8k_humaneval --n-tasks 80|"
  "specbench_qwen35_27b_mtp_2way|qwen35_27b_mtp|--workload specbench --n-tasks 120|"
  "qwen35_27b_ar|qwen35_27b_mtp|--workload bfcl_v4 --include-category web_search --train-n-tasks 30 --n-tasks 20|"
  "gsm8k_humaneval_14b_2way|qwen3_14b|--workload gsm8k_humaneval --n-tasks 80|--attention-backend triton --sampling-backend pytorch"
  "qwen3_14b_ar|qwen3_14b|--workload bfcl_v4 --include-category web_search --train-n-tasks 30 --n-tasks 20|--attention-backend triton --sampling-backend pytorch"
)

for cell in "${CELLS[@]}"; do
  IFS='|' read -r dir preset wargs sextra <<< "$cell"
  DIR="$RB/$dir"; OLOG="$DIR/decisions_select1_oracle.jsonl"
  echo "================= CELL $dir ($preset) ================="
  if [ ! -f "$OLOG" ]; then echo "$dir NO_ORACLE_LOG skip"; continue; fi
  python3 simulation/scripts/fit_chain_hybrid_calib_perpos.py --decision-log "$OLOG" \
    --oracle-labels --accept-conditioned --out-dir "$DIR/calib_betaCT" \
    > "$DIR/calib_betaCT_fit.log" 2>&1 || { echo "$dir FIT_FAIL"; continue; }
  cp "$DIR/calib_betaCT/calib_pp_beta.json" "$DIR/calib_pp_beta.json"
  EXTRA=""; [ -n "$sextra" ] && EXTRA="--extra-args $sextra"
  python3 simulation/scripts/measure_chain_hybrid.py --preset "$preset" $wargs \
    --steps 16 --tp-size 1 --tail-max-tokens 0 --skip-calib-fit \
    --arms select1_calib_beta --port 31082 --mem-fraction-static 0.65 \
    --output "$DIR/run_calib_betaCT.json" $EXTRA
  echo "$dir RC=$?"
done
echo "BETA_CT_ALL_DONE"
