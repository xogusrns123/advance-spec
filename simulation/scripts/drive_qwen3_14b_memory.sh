#!/usr/bin/env bash
# Add Qwen3-14B bfcl_v4 memory workload (the gap left by Phase 5 rerun,
# which only covered web_search for bfcl_v4). Runs after Qwen3-8B finishes.
set -u

WORKDIR=${WORKDIR:-/workspace}
cd "$WORKDIR"

echo "Waiting for Qwen3-8B pipeline to exit..."
while pgrep -f drive_qwen3_8b_pipeline.sh > /dev/null \
   || pgrep -f 'run_pipeline.sh.*qwen3_8b' > /dev/null; do
  sleep 60
done
echo "Qwen3-8B done. Starting Qwen3-14B bfcl_v4 memory at $(date)"
sleep 10

export UNION_TRIE=0
export EU_ORACLE=0
export STAGE1_TOPK=16
export STAGE1_NUM_DRAFT_TOKENS=256
export SWE_MAX_ITER=10
export BFCL_V4_INPUT=data/bfcl_agent/dataset_memory_4.jsonl

LOG_DIR=/workspace/simulation/results/qwen3_14b/_pipeline_logs
mkdir -p "$LOG_DIR"

FAILURES=0
for steps in 2 4 6 8; do
  suffix="memory_steps${steps}"
  out="/workspace/simulation/results/qwen3_14b/bfcl_v4_${suffix}"
  if [ -f "${out}/tree_oracle_sim.json" ]; then
    echo "[SKIP] qwen3_14b bfcl_v4_memory steps=$steps"; continue
  fi
  echo "[START] qwen3_14b bfcl_v4_memory steps=$steps  $(date)"
  log="$LOG_DIR/bfcl_v4_memory_${suffix}.log"
  STAGE1_STEPS=$steps OUTPUT_DIR_SUFFIX="$suffix" \
    bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_14b \
      > "$log" 2>&1
  rc=$?
  echo "[DONE ] qwen3_14b bfcl_v4_memory steps=$steps  rc=$rc  $(date)"
  if [ $rc -ne 0 ]; then
    echo "  FAILED"; tail -30 "$log" | sed 's/^/    /'
    FAILURES=$((FAILURES+1))
  fi
done

echo "=========================================================="
echo "Qwen3-14B memory done. Failures: $FAILURES  $(date)"
echo "=========================================================="
exit $FAILURES
