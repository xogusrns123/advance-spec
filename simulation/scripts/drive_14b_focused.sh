#!/usr/bin/env bash
# Qwen3-14B focused rerun — 3 configs only with the new code
# (by_count cap, draft-side p_t filters, by_count_score combo).
set -u

WORKDIR=${WORKDIR:-/workspace}
cd "$WORKDIR"

export UNION_TRIE=0
export EU_ORACLE=0
export STAGE1_TOPK=16
export STAGE1_NUM_DRAFT_TOKENS=256
export SWE_MAX_ITER=10
export SIM_BUDGETS=1,2,4,8,16,32,64,128

LOG_DIR=/workspace/simulation/results/qwen3_14b/_focused_logs
mkdir -p "$LOG_DIR"

FAILURES=0

run_one() {
  local workload=$1
  local steps=$2
  local suffix="steps${steps}"
  local out="/workspace/simulation/results/qwen3_14b/${workload}_${suffix}_draftpt"

  echo "[START] $workload steps=$steps  $(date)"
  local log="$LOG_DIR/${workload}_${suffix}.log"
  REQ_START=0 REQ_END=4 STAGE1_STEPS=$steps \
    OUTPUT_DIR_SUFFIX="${suffix}_draftpt" \
    bash simulation/scripts/run_pipeline.sh "$workload" qwen3_14b \
      > "$log" 2>&1
  local rc=$?
  echo "[DONE ] $workload steps=$steps  rc=$rc  $(date)"
  if [ $rc -ne 0 ]; then
    echo "  FAILED — tail:"
    tail -40 "$log" | sed 's/^/    /'
    FAILURES=$((FAILURES+1))
  fi
}

run_one specbench 8
run_one bfcl_v4  6
run_one swebench 6

echo "=========================================================="
echo "14B focused rerun complete. Failures: $FAILURES  $(date)"
echo "=========================================================="
exit $FAILURES
