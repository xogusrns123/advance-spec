#!/bin/bash
# DFlash served Panel-B boundary pipeline (sglang-native per-position select-1).
# Mirrors run_chain_hybrid_perdepth.sh + run_boundary_serve.sh, adapted for the
# DFlash block drafter on Blackwell GPU-0. ALL arms are PINNED to the record
# arm's GT-dump trajectory (greedy alone does NOT pin across arms for a block
# drafter — the substituted block flips FP near-ties; see
# project_chain_hybrid_fp_nondeterminism), so MAT/selacc differ only by selection.
#
# Phase A: fit-phase (raw select1 on TRAIN) -> calib_pp maps + decisions_select1_train.jsonl;
#          then record + raw + calib(4) + oracle on EVAL (pinned, replay-all).
# Phase B: fit disc_gbm_{mono,bayes} from the TRAIN select1 log (backfilled oracle_hit).
# Phase C: serve select1_mono,select1_bayes on EVAL (pinned, replay-existing, skip-calib-fit).
# Phase D: boundary figures (analyze_boundary_served.py --cell dflash).
#
# Args: [PORT] [TRAIN_N] [EVAL_N]   (defaults 30057 30 20)
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0 SGLANG_DISABLE_CUDNN_CHECK=1 SGLANG_ENABLE_JIT_DEEPGEMM=0
export PYTHONPATH=/workspace
PORT="${1:-30057}"; TRAIN_N="${2:-30}"; EVAL_N="${3:-20}"
R=simulation/results/chain_hybrid_perdepth
D="$R/qwen3_8b_dflash_ar"
COMMON="--preset qwen3_8b_dflash --workload bfcl_v4 --include-category web_search \
  --steps 16 --tp-size 1 --train-n-tasks $TRAIN_N --n-tasks $EVAL_N \
  --tail-max-tokens 0 --port $PORT --pin-trajectory"

echo "===== PHASE A: fit + eval (raw/calib/oracle), pinned ====="
python3 simulation/scripts/measure_chain_hybrid.py $COMMON --replay-all \
  --arms record,select1,select1_calib_histogram,select1_calib_isotonic,select1_calib_logistic,select1_calib_beta,select1_oracle \
  --output "$D/run_boundary_dflash.json" || { echo "PHASE_A_FAILED"; exit 1; }

echo "===== PHASE B: fit disc_gbm_{mono,bayes} from TRAIN select1 log ====="
[ -s "$D/decisions_select1_train.jsonl" ] || { echo "ABORT: no train log"; exit 1; }
python3 simulation/scripts/fit_chain_hybrid_discriminator.py --gbm \
  --oracle-log "$D/decisions_select1_train.jsonl" --out-dir "$D" \
  || { echo "PHASE_B_FAILED (disc fit)"; exit 1; }

echo "===== PHASE C: serve select1_mono,select1_bayes (pinned, reuse record) ====="
python3 simulation/scripts/measure_chain_hybrid.py $COMMON --replay-existing --skip-calib-fit \
  --arms select1_mono,select1_bayes \
  --output "$D/run_boundary_dflash_gbm.json" || { echo "PHASE_C_FAILED"; exit 1; }

echo "===== PHASE D: boundary figures ====="
python3 simulation/scripts/analyze_boundary_served.py --cell dflash || { echo "PHASE_D_FAILED"; exit 1; }
echo "DFLASH_BOUNDARY_DONE"
