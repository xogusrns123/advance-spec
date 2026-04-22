#!/usr/bin/env bash
# Qwen3-8B full pipeline driver. Mirror of drive_phase5.sh but with
# qwen3_8b preset. Waits for all Qwen3-14B work to clear the GPU
# before starting so there is no contention.
set -u

WORKDIR=${WORKDIR:-/workspace}
cd "$WORKDIR"

echo "Waiting for Qwen3-14B pipelines to exit..."
while pgrep -f 'rerun_swebench.sh\|drive_phase5.sh\|rerun_bfcl_v4.sh\|run_pipeline.sh.*qwen3_14b' > /dev/null; do
  sleep 60
done
echo "Qwen3-14B pipelines done. Starting Qwen3-8B at $(date)"
sleep 10

export UNION_TRIE=0
export EU_ORACLE=0
export STAGE1_TOPK=16
export STAGE1_NUM_DRAFT_TOKENS=256
export MAX_TOKENS_OVERRIDE=1024
export BFCL_V4_INPUT=data/bfcl_agent/dataset_small.jsonl
export SWE_MAX_ITER=10

LOG_DIR=/workspace/simulation/results/qwen3_8b/_pipeline_logs
mkdir -p "$LOG_DIR"
mkdir -p /workspace/data/swebench/repos

declare -A NUM_REQ
NUM_REQ[specbench]=4
NUM_REQ[bfcl_v4]=4
NUM_REQ[swebench]=4

run_one() {
  local workload=$1
  local steps=$2
  local n=${NUM_REQ[$workload]}
  local suffix="steps${steps}"
  local out="/workspace/simulation/results/qwen3_8b/${workload}_${suffix}"

  if [ -f "${out}/tree_oracle_sim.json" ]; then
    echo "[SKIP] $workload steps=$steps — already has Stage 6 output"
    return 0
  fi

  echo "=========================================================="
  echo "[START] qwen3_8b $workload steps=$steps  n=$n  $(date)"
  echo "=========================================================="

  local log="$LOG_DIR/${workload}_${suffix}.log"
  STAGE1_STEPS=$steps OUTPUT_DIR_SUFFIX=$suffix \
    bash simulation/scripts/run_pipeline.sh "$workload" qwen3_8b "$n" \
      > "$log" 2>&1
  local rc=$?
  echo "[DONE ] qwen3_8b $workload steps=$steps  rc=$rc  $(date)"
  if [ $rc -ne 0 ]; then
    echo "  FAILED — log tail:"
    tail -40 "$log" | sed 's/^/    /'
  fi
  return $rc
}

FAILURES=0
for workload in specbench bfcl_v4 swebench; do
  for steps in 2 4 6 8; do
    if ! run_one "$workload" "$steps"; then
      FAILURES=$((FAILURES+1))
    fi
  done
done

echo "=========================================================="
echo "Qwen3-8B pipeline complete. Failures: $FAILURES  $(date)"
echo "=========================================================="
exit $FAILURES
