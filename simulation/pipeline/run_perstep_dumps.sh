#!/bin/bash
# Run sim per (benchmark, s, k) at B=64 with per-step JSONL dump.
# Covers all best-config combos we need:
#   bfcl_v4:    s=2 (for dns, topk) and s=4 (for basic, oracle)
#   specbench:  s=2 (dns, topk) and s=4 (basic, oracle)
#   swebench:   s=2 (all 4 methods)
set -e
ROOT=/workspace
CAP_ROOT=$ROOT/simulation/results/qwen3_14b
STEP_ROOT=$ROOT/simulation/results/step_dataset/qwen3_14b
DUMP_DIR=$STEP_ROOT/_perstep_dumps
mkdir -p $DUMP_DIR

METHODS="extension:4.0:0.0,extension_oracle:4.0:0.0,extension_dns:0.5:0.8:4.0:0.0,extension_topk:0.5:8:4.0:0.0"

run_one() {
  local bench=$1
  local s=$2
  local dump=$DUMP_DIR/${bench}_s${s}k16_B64.jsonl
  rm -f $dump
  echo "[run] $bench s=$s B=64 → $dump"
  cd $ROOT && SIM_PER_STEP_JSONL=$dump python3 -m simulation.evaluation.run_tree_oracle_sim \
    --agent-results "$CAP_ROOT/${bench}_steps8_topk16_capture/agent_results_eagle3.json" \
    --latency-config "$CAP_ROOT/${bench}_steps8_topk16_capture/latency_config.json" \
    --exclude "$STEP_ROOT/_exclude/${bench}.txt" \
    --methods "$METHODS" \
    --budgets 64 \
    --capture-steps 8 --capture-topk 16 \
    --reslice-steps $s --reslice-topk 16 \
    --steps $s --topk 16 \
    --model Qwen/Qwen3-14B \
    --output $DUMP_DIR/${bench}_s${s}k16_B64_summary.json 2>&1 | tail -5
}

# 5 runs (bfcl s2, bfcl s4, specbench s2, specbench s4, swebench s2)
run_one bfcl_v4 2
run_one bfcl_v4 4
run_one specbench 2
run_one specbench 4
run_one swebench_verified 2

echo PERSTEP_DUMP_DONE
