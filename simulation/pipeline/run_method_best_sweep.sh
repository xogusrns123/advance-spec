#!/bin/bash
# Run run_tree_oracle_sim.py per (benchmark, s, k) combination.
# Output: simulation/results/step_dataset/qwen3_14b/_sims/{bench}_s{s}k{k}.json
#
# Sweep config:
#   (s, k) ∈ {(2,16), (4,16), (6,16), (8,16)}
#   B ∈ {4, 8, 16, 32, 64}
#   Methods: single:eagle3, extension_oracle:4.0:0.0,
#            extension_dns:0.5:0.8:4.0:0.0, extension_topk:0.5:8:4.0:0.0
#   First 10 tasks per benchmark (via _exclude/{bench}.txt)
set -e

ROOT=/workspace
STEP_ROOT=$ROOT/simulation/results/step_dataset/qwen3_14b
CAP_ROOT=$ROOT/simulation/results/qwen3_14b
OUT_DIR=$STEP_ROOT/_sims
mkdir -p $OUT_DIR

METHODS="single:eagle3,extension_oracle:4.0:0.0,extension_dns:0.5:0.8:4.0:0.0,extension_topk:0.5:8:4.0:0.0"
BUDGETS="4,8,16,32,64"

for bench in bfcl_v4 specbench swebench_verified; do
  CAP="$CAP_ROOT/${bench}_steps8_topk16_capture/agent_results_eagle3.json"
  LAT="$CAP_ROOT/${bench}_steps8_topk16_capture/latency_config.json"
  EXC="$STEP_ROOT/_exclude/${bench}.txt"

  for sk in "2 16" "4 16" "6 16" "8 16"; do
    s=$(echo $sk | cut -d' ' -f1)
    k=$(echo $sk | cut -d' ' -f2)
    OUT="$OUT_DIR/${bench}_s${s}k${k}.json"
    if [ -f "$OUT" ]; then
      echo "[skip] $OUT exists"
      continue
    fi
    echo "[run] $bench s=$s k=$k → $OUT"
    cd $ROOT && python3 -m simulation.evaluation.run_tree_oracle_sim \
      --agent-results "$CAP" \
      --latency-config "$LAT" \
      --exclude "$EXC" \
      --methods "$METHODS" \
      --budgets "$BUDGETS" \
      --capture-steps 8 --capture-topk 16 \
      --reslice-steps $s --reslice-topk $k \
      --steps $s --topk $k \
      --model Qwen/Qwen3-14B \
      --output "$OUT" 2>&1 | tail -20
  done
done
echo "SWEEP_DONE"
