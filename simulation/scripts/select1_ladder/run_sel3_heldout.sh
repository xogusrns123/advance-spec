#!/bin/bash
# HELD-OUT realized 3-way calib/bayes for 27B (matches the 2-way ladder protocol).
#   STEP 1  record + oracle on DISJOINT train tasks [20,50)  -> 3-way train oracle log
#   STEP 2  fit per-proposer SEL3 bundle on that train log   -> sel3_bundle_heldout.json
#   STEP 3  serve SEL3 (gbm, then logistic) on the EVAL trajectory (real_full tasks 0-19,
#           replay-existing) -> realized held-out MAT, comparable to raw 5.001 / oracle 7.054.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_CHAIN_HYBRID_DFLASH_AUX=1
BASE=/workspace/simulation/results/chain_hybrid_perdepth
TRAINDIR="$BASE/qwen35_27b_3way_train"
SEL3DIR="$BASE/qwen35_27b_3way_sel3"
COMMON="--preset qwen35_27b_mtp --workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1 --tail-max-tokens 0 --mem-fraction-static 0.65 --port 31082"
mkdir -p "$TRAINDIR"

echo "===== STEP 1: record+oracle TRAIN tasks [20,50) ====="
unset SGLANG_CHAIN_HYBRID_SEL3 || true
python3 simulation/scripts/measure_chain_hybrid.py $COMMON \
  --train-n-tasks 20 --n-tasks 30 --arms record,select1_oracle \
  --output "$TRAINDIR/run.json"
echo "STEP1_RC=$?"
[ -s "$TRAINDIR/decisions_select1_oracle.jsonl" ] || { echo "ABORT: no train oracle log"; exit 1; }

echo "===== STEP 2: fit held-out SEL3 bundle on train log ====="
python3 simulation/scripts/select1_ladder/fit_3way_selector.py \
  --dir "$TRAINDIR" \
  --props mtp:eagle_token:eagle_p dflash:dflash_token:dflash_p suffix:suffix_token:suffix_p \
  --out "$SEL3DIR/sel3_bundle_heldout.json"
echo "STEP2_RC=$?"
[ -s "$SEL3DIR/sel3_bundle_heldout.json" ] || { echo "ABORT: no held-out bundle"; exit 1; }

export SGLANG_CHAIN_HYBRID_SEL3="$SEL3DIR/sel3_bundle_heldout.json"
for M in gbm logistic; do
  echo "===== STEP 3.$M: serve SEL3=$M on EVAL (real_full tasks 0-19, replay) ====="
  export SGLANG_CHAIN_HYBRID_SEL3_METHOD="$M"
  python3 simulation/scripts/measure_chain_hybrid.py $COMMON \
    --n-tasks 20 --replay-existing --arms select1 \
    --output "$SEL3DIR/run_heldout_${M}.json"
  echo "STEP3_${M}_RC=$?"
done
echo "SEL3_HELDOUT_DONE"
