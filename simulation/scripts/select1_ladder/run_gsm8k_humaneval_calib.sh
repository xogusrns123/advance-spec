#!/bin/bash
# Complete the GSM8K+HumanEval realized ladder: fit calib(logistic)+bayes(GBM) maps IN-SAMPLE on
# the eval oracle decision log (oracle-trajectory fit = the better calibration per our prior work),
# then serve select1_calib_logistic + select1_bayes LIVE (no replay) on the same 80 tasks -- same
# realized regime as the raw(select1) arm (each arm builds its own draft chain). train_n_tasks=0 so
# no fit-phase; maps are pre-placed.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }; done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
DIR=/workspace/simulation/results/chain_hybrid_perdepth/gsm8k_humaneval_2way
OLOG="$DIR/decisions_select1_oracle.jsonl"
N="${N:-80}"

echo "=== FIT (in-sample on eval oracle log) ==="
python3 simulation/scripts/fit_chain_hybrid_discriminator.py --oracle-log "$OLOG" --out-dir "$DIR" --gbm
echo "disc_fit_rc=$?"
python3 simulation/scripts/fit_chain_hybrid_calib_perpos.py --decision-log "$OLOG" --out-dir "$DIR"
echo "calib_fit_rc=$?"
ls -la "$DIR"/calib_pp_logistic.json "$DIR"/disc_gbm_bayes.json 2>&1 | awk '{print $5,$NF}'

echo "=== SERVE calib_logistic + bayes (live realized, N=$N) ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen35_27b_mtp --workload gsm8k_humaneval \
  --steps 16 --tp-size 1 --n-tasks "$N" --tail-max-tokens 0 --skip-calib-fit \
  --arms select1_calib_logistic,select1_bayes --port 31082 --mem-fraction-static 0.65 \
  --output "$DIR/run_calib.json"
echo "GSM8K_HUMANEVAL_CALIB_RC=$?"
