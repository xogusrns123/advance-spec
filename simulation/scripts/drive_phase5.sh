#!/usr/bin/env bash
# Phase 5 rerun: Qwen3-14B × (specbench, bfcl_v4, swebench) × steps={2,4,6,8}.
# 4 samples per workload. Single GPU, sequential.
# bfcl_v4 uses a filtered dataset (web_search_base, no prereqs) to keep the
# effective request count at 4 — the full dataset expands 4 → 34 via
# memory/state deps.
set -u

WORKDIR=${WORKDIR:-/workspace}
cd "$WORKDIR"

export UNION_TRIE=0
export EU_ORACLE=0
export STAGE1_TOPK=16
export STAGE1_NUM_DRAFT_TOKENS=256

# Cap per-turn generation for specbench (writing category otherwise runs to 2048).
export MAX_TOKENS_OVERRIDE=1024

# Filtered bfcl_v4 dataset: 4 no-dep web_search_base entries.
export BFCL_V4_INPUT=data/bfcl_agent/dataset_small.jsonl

# Cap swebench agent loop — full 30 iters × 4 samples × 1GPU would be hours.
export SWE_MAX_ITER=10

LOG_DIR=/workspace/simulation/results/qwen3_14b/_phase5_logs
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
  local out="/workspace/simulation/results/qwen3_14b/${workload}_${suffix}"

  if [ -f "${out}/tree_oracle_sim.json" ]; then
    echo "[SKIP] $workload steps=$steps — already has Stage 6 output"
    return 0
  fi

  echo "=========================================================="
  echo "[START] $workload steps=$steps  n=$n  $(date)"
  echo "  OUTPUT_DIR_SUFFIX=$suffix  STAGE1_STEPS=$steps"
  echo "=========================================================="

  local log="$LOG_DIR/${workload}_${suffix}.log"
  STAGE1_STEPS=$steps OUTPUT_DIR_SUFFIX=$suffix \
    bash simulation/scripts/run_pipeline.sh "$workload" qwen3_14b "$n" \
      > "$log" 2>&1
  local rc=$?
  echo "[DONE ] $workload steps=$steps  rc=$rc  $(date)"
  if [ $rc -ne 0 ]; then
    echo "  FAILED — log tail:"
    tail -40 "$log" | sed 's/^/    /'
  fi
  return $rc
}

# Order: specbench (fastest), bfcl_v4 (network), swebench (repo clone + agentic)
FAILURES=0
for workload in specbench bfcl_v4 swebench; do
  for steps in 2 4 6 8; do
    if ! run_one "$workload" "$steps"; then
      FAILURES=$((FAILURES+1))
    fi
  done
done

echo "=========================================================="
echo "Phase 5 complete. Failures: $FAILURES"
echo "=========================================================="
exit $FAILURES
