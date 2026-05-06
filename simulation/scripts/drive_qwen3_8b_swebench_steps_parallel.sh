#!/usr/bin/env bash
# Parallel swebench steps sweep for Qwen3-8B.
#
# Runs 4 pipelines in parallel (steps=2,4,6,8) — one per GPU — with
# disjoint PORT_OFFSETs and SWE_REPOS_DIRs so they don't race on ports
# or git state. Output dirs follow the `{workload}_steps{N}` convention
# so compare_methods.ipynb auto-discovers them under the "steps" axis.
#
# Prerequisites (done by caller):
#   - data/swebench/dataset.jsonl has ≥4 instances
#   - data/swebench/repos_steps{2,4,6,8}/ each hold the full repo set
#   - simulation/results/qwen3_8b/latency_config.json pre-seeded
#
# Usage:
#   bash simulation/scripts/drive_qwen3_8b_swebench_steps_parallel.sh
set -u

WORKDIR=${WORKDIR:-/workspace}
cd "$WORKDIR"

LOG_DIR=simulation/results/qwen3_8b/_pipeline_logs
mkdir -p "$LOG_DIR"

# Shared config across all 4 pipelines.
export UNION_TRIE=0
export EU_ORACLE=0
export STAGE1_TOPK=16
export STAGE1_NUM_DRAFT_TOKENS=256
export SWE_MAX_ITER=30
export SIM_BUDGETS=1,2,4,8,16,32,64,128

echo "=== Parallel swebench steps sweep (Qwen3-8B) starting at $(date) ==="

PIDS=()
STEPS=(2 4 6 8)
for i in 0 1 2 3; do
  s=${STEPS[$i]}
  gpu=$i
  port_offset=$((i * 100))
  suffix="steps${s}"
  repos_dir="data/swebench/repos_steps${s}"
  out="simulation/results/qwen3_8b/swebench_${suffix}"
  log="$LOG_DIR/swebench_${suffix}.log"

  if [ -f "${out}/tree_oracle_sim.json" ]; then
    echo "[SKIP] steps=$s  (already have ${out}/tree_oracle_sim.json)"
    continue
  fi

  echo "[START] steps=$s gpu=$gpu port_offset=$port_offset repos=$repos_dir"
  (
    GPU_IDS=$gpu \
    NUM_GPUS=1 \
    PORT_OFFSET=$port_offset \
    STAGE1_STEPS=$s \
    OUTPUT_DIR_SUFFIX=$suffix \
    SWE_REPOS_DIR=$repos_dir \
      bash simulation/scripts/run_pipeline.sh swebench qwen3_8b \
        > "$log" 2>&1
    rc=$?
    echo "[DONE] steps=$s rc=$rc $(date)" >> "$LOG_DIR/_master.log"
    if [ $rc -ne 0 ]; then
      echo "[FAIL] steps=$s  last 40 lines of $log:" >> "$LOG_DIR/_master.log"
      tail -40 "$log" | sed 's/^/  /' >> "$LOG_DIR/_master.log"
    fi
  ) &
  PIDS+=($!)
done

echo "Launched ${#PIDS[@]} pipelines. PIDs: ${PIDS[*]}"
echo "Logs: $LOG_DIR/"
echo "Waiting for all to finish..."

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then FAILED=$((FAILED + 1)); fi
done

echo ""
echo "=== Parallel swebench steps sweep done at $(date) ==="
echo "Failures: $FAILED / ${#PIDS[@]}"
exit $FAILED
