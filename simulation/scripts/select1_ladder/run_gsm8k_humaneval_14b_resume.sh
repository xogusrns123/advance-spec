#!/bin/bash
# RESUME the 14B GSM8K+HumanEval ladder: record arm already completed (run.json record=1.051,
# gt_tokens.jsonl + agent_results_record.json intact). Run select1(raw,live) + select1_oracle
# (replays the existing record) -> run_sel.json, then fit calib(logistic)+bayes(GBM) in-sample on
# the fresh oracle log and serve them live -> run_calib.json. EAGLE3 + Blackwell triton+pytorch via
# --extra-args (REMAINDER, last). Env N (default 80). 14B native MAT~1 (balanced regime) so each arm
# has many decode steps and runs slow -- expected, not a hang.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }; done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
N="${N:-80}"
DIR=/workspace/simulation/results/chain_hybrid_perdepth/gsm8k_humaneval_14b_2way
OLOG="$DIR/decisions_select1_oracle.jsonl"

echo "=== STEP 1: select1(raw,live) + select1_oracle(replay record) N=$N ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen3_14b --workload gsm8k_humaneval \
  --steps 16 --tp-size 1 --n-tasks "$N" --tail-max-tokens 0 \
  --arms select1,select1_oracle --port 31082 --mem-fraction-static 0.65 \
  --output "$DIR/run_sel.json" \
  --extra-args --attention-backend triton --sampling-backend pytorch
echo "SEL_RC=$?"

echo "=== STEP 2: FIT calib(logistic)+bayes(GBM) in-sample on oracle log ==="
python3 simulation/scripts/fit_chain_hybrid_discriminator.py --oracle-log "$OLOG" --out-dir "$DIR" --gbm
echo "disc_fit_rc=$?"
python3 simulation/scripts/fit_chain_hybrid_calib_perpos.py --decision-log "$OLOG" --out-dir "$DIR"
echo "calib_fit_rc=$?"

echo "=== STEP 3: SERVE calib_logistic + bayes (live realized) N=$N ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen3_14b --workload gsm8k_humaneval \
  --steps 16 --tp-size 1 --n-tasks "$N" --tail-max-tokens 0 --skip-calib-fit \
  --arms select1_calib_logistic,select1_bayes --port 31082 --mem-fraction-static 0.65 \
  --output "$DIR/run_calib.json" \
  --extra-args --attention-backend triton --sampling-backend pytorch
echo "GSM8K_HUMANEVAL_14B_FULL_RC=$?"
