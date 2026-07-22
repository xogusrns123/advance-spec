#!/bin/bash
# Run basic extension (no selection) for BFCLv4 at the 2 unique best configs,
# force-1 mode + per-step dump enabled.
#
# Output:
#   {STEP_ROOT}/_bfcl_full/bfcl_v4_s2k16B64.jsonl   (basic/dns/topk argmax)
#   {STEP_ROOT}/_bfcl_full/bfcl_v4_s4k16B64.jsonl   (oracle argmax)
set -e
ROOT=/workspace
CAP_DIR=$ROOT/simulation/results/qwen3_14b/bfcl_v4_steps8_topk16_capture
STEP_ROOT=$ROOT/simulation/results/step_dataset/qwen3_14b
OUT_DIR=$STEP_ROOT/_bfcl_full
mkdir -p $OUT_DIR

run_one() {
  local s=$1
  local tag="s${s}k16B64"
  local dump=$OUT_DIR/bfcl_v4_${tag}.jsonl
  local summary=$OUT_DIR/bfcl_v4_${tag}_summary.json
  rm -f $dump
  echo "[run] BFCLv4 (s=$s, k=16, B=64) basic extension, force-1"
  cd $ROOT && SIM_FORCE_ADVANCE_1=1 SIM_PER_STEP_JSONL=$dump \
    python3 -m simulation.evaluation.run_tree_oracle_sim \
      --agent-trajectory $CAP_DIR/agent_results_eagle3.json \
      --latency-data $CAP_DIR/latency_data.json \
      --exclude $STEP_ROOT/_exclude/bfcl_v4.txt \
      --methods 'extension:4.0:0.0' \
      --budgets 64 \
      --capture-steps 8 --capture-topk 16 \
      --reslice-steps $s --reslice-topk 16 \
      --steps $s --topk 16 \
      --model Qwen/Qwen3-14B \
      --output $summary 2>&1 | tail -5
}

run_one 2
run_one 4
echo BFCL_FULL_DUMP_DONE
