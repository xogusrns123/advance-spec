#!/usr/bin/env bash
# Qwen3-8B full pipeline driver — 4 workloads × 4 steps × 4 samples each.
# Wait for any GPU-heavy Qwen3-14B work to exit so we do not contend.
set -u

WORKDIR=${WORKDIR:-/workspace}
cd "$WORKDIR"

echo "Waiting for Qwen3-14B pipelines to exit..."
# pgrep uses POSIX ERE; '\|' is literal, not alternation. Use separate calls.
while pgrep -f rerun_swebench.sh > /dev/null \
   || pgrep -f drive_phase5.sh > /dev/null \
   || pgrep -f rerun_bfcl_v4.sh > /dev/null \
   || pgrep -f 'run_pipeline.sh .* qwen3_14b' > /dev/null \
   || pgrep -f 'run_pipeline.sh.*qwen3_14b' > /dev/null; do
  sleep 60
done
echo "Qwen3-14B pipelines done. Starting Qwen3-8B at $(date)"
sleep 10

export UNION_TRIE=0
export EU_ORACLE=0
export STAGE1_TOPK=16
export STAGE1_NUM_DRAFT_TOKENS=256
export MAX_TOKENS_OVERRIDE=1024
export SWE_MAX_ITER=10

LOG_DIR=/workspace/simulation/results/qwen3_8b/_pipeline_logs
mkdir -p "$LOG_DIR"
mkdir -p /workspace/data/swebench/repos

# --- Workloads / inputs ---
# bfcl_v4 memory: first 4 memory_kv_prereq items (no cross-file deps).
# bfcl_v4 web_search: same 4 no-dep web_search_base entries used before.
# specbench / swebench: first 4 dataset entries via REQ_START/REQ_END.

run_specbench() {
  local steps=$1
  local suffix="steps${steps}"
  local out="/workspace/simulation/results/qwen3_8b/specbench_${suffix}"
  if [ -f "${out}/tree_oracle_sim.json" ]; then
    echo "[SKIP] specbench steps=$steps"; return 0
  fi
  echo "[START] qwen3_8b specbench steps=$steps  $(date)"
  local log="$LOG_DIR/specbench_${suffix}.log"
  REQ_START=0 REQ_END=4 STAGE1_STEPS=$steps OUTPUT_DIR_SUFFIX="$suffix" \
    bash simulation/scripts/run_pipeline.sh specbench qwen3_8b \
      > "$log" 2>&1
  local rc=$?
  echo "[DONE ] qwen3_8b specbench steps=$steps  rc=$rc  $(date)"
  if [ $rc -ne 0 ]; then echo "  FAILED"; tail -30 "$log" | sed 's/^/    /'; fi
  # Move the req-suffixed output to a cleaner path
  # run_pipeline appends both _req and _steps{N}; normalize.
  return $rc
}

run_bfcl_v4_memory() {
  local steps=$1
  local suffix="memory_steps${steps}"
  local out="/workspace/simulation/results/qwen3_8b/bfcl_v4_${suffix}"
  if [ -f "${out}/tree_oracle_sim.json" ]; then
    echo "[SKIP] bfcl_v4_memory steps=$steps"; return 0
  fi
  echo "[START] qwen3_8b bfcl_v4_memory steps=$steps  $(date)"
  local log="$LOG_DIR/bfcl_v4_memory_${suffix}.log"
  BFCL_V4_INPUT=data/bfcl_agent/dataset_memory_4.jsonl \
    STAGE1_STEPS=$steps OUTPUT_DIR_SUFFIX="$suffix" \
    bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b \
      > "$log" 2>&1
  local rc=$?
  echo "[DONE ] qwen3_8b bfcl_v4_memory steps=$steps  rc=$rc  $(date)"
  if [ $rc -ne 0 ]; then echo "  FAILED"; tail -30 "$log" | sed 's/^/    /'; fi
  return $rc
}

run_bfcl_v4_websearch() {
  local steps=$1
  local suffix="websearch_steps${steps}"
  local out="/workspace/simulation/results/qwen3_8b/bfcl_v4_${suffix}"
  if [ -f "${out}/tree_oracle_sim.json" ]; then
    echo "[SKIP] bfcl_v4_websearch steps=$steps"; return 0
  fi
  echo "[START] qwen3_8b bfcl_v4_websearch steps=$steps  $(date)"
  local log="$LOG_DIR/bfcl_v4_websearch_${suffix}.log"
  BFCL_V4_INPUT=data/bfcl_agent/dataset_small.jsonl \
    STAGE1_STEPS=$steps OUTPUT_DIR_SUFFIX="$suffix" \
    bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b \
      > "$log" 2>&1
  local rc=$?
  echo "[DONE ] qwen3_8b bfcl_v4_websearch steps=$steps  rc=$rc  $(date)"
  if [ $rc -ne 0 ]; then echo "  FAILED"; tail -30 "$log" | sed 's/^/    /'; fi
  return $rc
}

run_swebench() {
  local steps=$1
  local suffix="steps${steps}"
  local out="/workspace/simulation/results/qwen3_8b/swebench_${suffix}"
  if [ -f "${out}/tree_oracle_sim.json" ]; then
    echo "[SKIP] swebench steps=$steps"; return 0
  fi
  echo "[START] qwen3_8b swebench steps=$steps  $(date)"
  local log="$LOG_DIR/swebench_${suffix}.log"
  REQ_START=0 REQ_END=4 STAGE1_STEPS=$steps OUTPUT_DIR_SUFFIX="$suffix" \
    bash simulation/scripts/run_pipeline.sh swebench qwen3_8b \
      > "$log" 2>&1
  local rc=$?
  echo "[DONE ] qwen3_8b swebench steps=$steps  rc=$rc  $(date)"
  if [ $rc -ne 0 ]; then echo "  FAILED"; tail -30 "$log" | sed 's/^/    /'; fi
  return $rc
}

FAILURES=0
# Run 4 workloads × 4 steps. Order: specbench (shortest/fastest) first,
# then bfcl_v4 categories, then swebench (longest due to repo clones).
for steps in 2 4 6 8; do
  run_specbench $steps       || FAILURES=$((FAILURES+1))
done
for steps in 2 4 6 8; do
  run_bfcl_v4_websearch $steps || FAILURES=$((FAILURES+1))
done
for steps in 2 4 6 8; do
  run_bfcl_v4_memory $steps  || FAILURES=$((FAILURES+1))
done
for steps in 2 4 6 8; do
  run_swebench $steps        || FAILURES=$((FAILURES+1))
done

echo "=========================================================="
echo "Qwen3-8B pipeline complete. Failures: $FAILURES  $(date)"
echo "=========================================================="
exit $FAILURES
