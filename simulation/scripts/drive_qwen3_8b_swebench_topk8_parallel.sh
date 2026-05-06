#!/usr/bin/env bash
# Queued follow-up to drive_qwen3_8b_swebench_steps_parallel.sh:
# same swebench × steps={2,4,6,8} sweep but with STAGE1_TOPK=8 (branching
# dimension) instead of the earlier topk=16 run.
#
# Output dir pattern: swebench_topk8_steps{N}  → compare_methods.ipynb
# treats these as a distinct workload ("swebench_topk8") with steps axis.
#
# Waits for the topk=16 driver to finish before launching so GPUs and
# repo copies (repos_steps{N}) are released for reuse.
set -u

WORKDIR=${WORKDIR:-/workspace}
cd "$WORKDIR"

echo "=== Queued topk=8 sweep waiting for topk=16 driver to exit..."
while pgrep -f drive_qwen3_8b_swebench_steps_parallel.sh > /dev/null \
   || pgrep -f 'run_pipeline.sh.*swebench.*qwen3_8b' > /dev/null; do
  sleep 60
done
echo "=== topk=16 driver done. Starting topk=8 sweep at $(date) ==="
sleep 10

LOG_DIR=simulation/results/qwen3_8b/_pipeline_logs
mkdir -p "$LOG_DIR"

export UNION_TRIE=0
export EU_ORACLE=0
export STAGE1_TOPK=8
export STAGE1_NUM_DRAFT_TOKENS=256
export SWE_MAX_ITER=30
export SIM_BUDGETS=1,2,4,8,16,32,64,128

PIDS=()
STEPS=(2 4 6 8)
for i in 0 1 2 3; do
  s=${STEPS[$i]}
  gpu=$i
  port_offset=$((i * 100))
  # workload="swebench_topk8" so compare_methods.ipynb groups them separately
  # from the topk=16 runs under the _steps axis.
  suffix="topk8_steps${s}"
  # Reuse the repo copies from the topk=16 run — they were reset to
  # base_commit at the end of each pipeline by _cleanup_repos().
  repos_dir="data/swebench/repos_steps${s}"
  out="simulation/results/qwen3_8b/swebench_${suffix}"
  log="$LOG_DIR/swebench_${suffix}.log"

  if [ -f "${out}/tree_oracle_sim.json" ]; then
    echo "[SKIP] topk=8 steps=$s  (already have ${out}/tree_oracle_sim.json)"
    continue
  fi

  echo "[START] topk=8 steps=$s gpu=$gpu port_offset=$port_offset repos=$repos_dir"
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
    echo "[DONE] topk=8 steps=$s rc=$rc $(date)" >> "$LOG_DIR/_master.log"
    if [ $rc -ne 0 ]; then
      echo "[FAIL] topk=8 steps=$s  last 40 lines of $log:" >> "$LOG_DIR/_master.log"
      tail -40 "$log" | sed 's/^/  /' >> "$LOG_DIR/_master.log"
    fi
  ) &
  PIDS+=($!)
done

echo "Launched ${#PIDS[@]} topk=8 pipelines. PIDs: ${PIDS[*]}"
echo "Logs: $LOG_DIR/"

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then FAILED=$((FAILED + 1)); fi
done

echo ""
echo "=== topk=8 sweep done at $(date) ==="
echo "Failures: $FAILED / ${#PIDS[@]}"
exit $FAILED
