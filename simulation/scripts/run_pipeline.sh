#!/usr/bin/env bash
# Unified oracle simulation pipeline for all benchmarks.
#
# Usage:
#   bash simulation/scripts/run_pipeline.sh <benchmark> <model_preset> [num_requests]
#
# Benchmarks: bfcl_v3, bfcl_v4, specbench, swebench
# Model presets: glm4_flash, qwen3_8b
#
# Execution toggles:
#   UNION_TRIE=0 (default)  skip Stages 4 and 5 entirely. Stage 6 assembles
#                           per-proposer records on the fly and skips both
#                           union_trie_* and EU_oracle methods.
#   UNION_TRIE=1            build union trie (Stage 4) and run union_trie_*
#                           methods in Stage 6.
#   EU_ORACLE=0 (default)   skip Stage 5 (p_t collection) and disable the
#                           EU oracle in Stage 6. Uses p_t_oracle derived
#                           from ground truth if any method needs p_t.
#   EU_ORACLE=1             run Stage 5 (real target p_t) and enable the
#                           EU oracle in Stage 6. Requires UNION_TRIE=1.
#
# Stage 1 EAGLE3 tree shape (override to sweep draft-tree configurations):
#   STAGE1_TOPK=8           tree branching factor per level (default 8)
#   STAGE1_STEPS=5          tree max depth (default 5)
#   STAGE1_NUM_DRAFT_TOKENS=256  total draft budget (default 256)
#   OUTPUT_DIR_SUFFIX=<str> appended to OUTPUT_DIR when sweeping so each
#                           run (e.g. steps=2/4/6/8) writes to a distinct dir.
#
# Request range (for parallel execution across machines):
#   REQ_START=0  REQ_END=50  bash simulation/scripts/run_pipeline.sh bfcl_v4 glm4_flash  # Machine A
#   REQ_START=50 REQ_END=100 bash simulation/scripts/run_pipeline.sh bfcl_v4 glm4_flash  # Machine B
#   # Then merge: simulation/scripts/merge_shards.sh simulation/results/glm4_flash/bfcl_v4
#
# Examples:
#   bash simulation/scripts/run_pipeline.sh bfcl_v3 glm4_flash 10   # first 10 requests
#   bash simulation/scripts/run_pipeline.sh bfcl_v4 glm4_flash 5    # first 5 requests
#   REQ_START=5 REQ_END=10 bash simulation/scripts/run_pipeline.sh bfcl_v4 glm4_flash  # requests 5-9
#   UNION_TRIE=1 bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b  # enable Stage 4
#   UNION_TRIE=1 EU_ORACLE=1 bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b  # full
set -euo pipefail

BENCHMARK=${1:?Usage: $0 <benchmark> <model_preset> [num_requests]}
MODEL_PRESET=${2:?Usage: $0 <benchmark> <model_preset> [num_requests]}
NUM_REQUESTS=${3:-}
PORT=${PORT:-30000}
REQ_START=${REQ_START:-}
REQ_END=${REQ_END:-}
NUM_WORKERS=${NUM_WORKERS:-1}
EU_ORACLE=${EU_ORACLE:-0}
UNION_TRIE=${UNION_TRIE:-0}

# EU_ORACLE=1 requires UNION_TRIE=1 (EU relies on the union trie).
if [ "$EU_ORACLE" = "1" ] && [ "$UNION_TRIE" = "0" ]; then
  echo "ERROR: EU_ORACLE=1 requires UNION_TRIE=1; EU oracle depends on the union trie."
  exit 1
fi

# --- Model preset ---
# TP_SIZE is only consumed by Stage 3c (MTP server, GLM only). Stage 1/5
# shard one server per GPU with TP=1, so new presets default to TP_SIZE=1.
case $MODEL_PRESET in
  glm4_flash)
    MODEL="zai-org/GLM-4.7-Flash"
    DRAFT_MODEL="thoughtworks/GLM-4.7-Flash-Eagle3"
    DRAFT_LM=""
    TP_SIZE=4
    MEM_FRAC=0.8
    ;;
  qwen3_8b)
    MODEL="Qwen/Qwen3-8B"
    DRAFT_MODEL="Tengyunw/qwen3_8b_eagle3"
    DRAFT_LM="Qwen/Qwen3-0.6B"
    TP_SIZE=1
    MEM_FRAC=0.85
    ;;
  qwen3_14b)
    MODEL="Qwen/Qwen3-14B"
    DRAFT_MODEL="AngelSlim/Qwen3-14B_eagle3"
    DRAFT_LM="Qwen/Qwen3-0.6B"
    TP_SIZE=1
    MEM_FRAC=0.85
    ;;
  qwen3_32b)
    MODEL="Qwen/Qwen3-32B"
    DRAFT_MODEL="Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3"
    DRAFT_LM="Qwen/Qwen3-0.6B"
    TP_SIZE=1
    MEM_FRAC=0.85
    ;;
  *)
    echo "Unknown model preset: $MODEL_PRESET (use glm4_flash, qwen3_8b, qwen3_14b, or qwen3_32b)"
    exit 1
    ;;
esac

# --- Benchmark config ---
case $BENCHMARK in
  bfcl_v3)
    AGENT_MODULE="simulation.agents.bfcl_agent"
    INPUT_FILE="data/bfcl_multi_turn/dataset.jsonl"
    DATASET_FLAG="--model $MODEL"
    MAX_ITER_FLAG="--max-iterations 5"
    TEMP_FLAG="--temperature 0.0"
    ;;
  bfcl_v4)
    AGENT_MODULE="simulation.agents.bfcl_v4_agent"
    INPUT_FILE="data/bfcl_agent/dataset.jsonl"
    DATASET_FLAG="--model $MODEL"
    MAX_ITER_FLAG="--max-iterations 5"
    TEMP_FLAG=""
    ;;
  specbench)
    AGENT_MODULE="simulation.agents.specbench_agent"
    INPUT_FILE="data/specbench/dataset.jsonl"
    DATASET_FLAG="--dataset $INPUT_FILE --model $MODEL"
    MAX_ITER_FLAG=""
    TEMP_FLAG="--temperature 0.0"
    ;;
  swebench)
    AGENT_MODULE="simulation.agents.swebench_agent"
    INPUT_FILE="data/swebench/dataset.jsonl"
    DATASET_FLAG="--model $MODEL"
    MAX_ITER_FLAG="--max-iterations 30 --repos-dir data/swebench/repos"
    TEMP_FLAG="--temperature 0.0"
    ;;
  *)
    echo "Unknown benchmark: $BENCHMARK (use bfcl_v3, bfcl_v4, specbench, or swebench)"
    exit 1
    ;;
esac

MODEL_SHORT=$(echo $MODEL_PRESET | tr '[:upper:]' '[:lower:]')
OUTPUT_DIR="simulation/results/${MODEL_SHORT}/${BENCHMARK}"

# Stage 1 EAGLE3 tree shape (passed through to run_parallel_stage1.sh).
# Sweep steps={2,4,6,8} by running this script multiple times with
# STAGE1_STEPS=<N> and distinct OUTPUT_DIR_SUFFIX to avoid clobbering.
export STAGE1_TOPK=${STAGE1_TOPK:-8}
export STAGE1_STEPS=${STAGE1_STEPS:-5}
export STAGE1_NUM_DRAFT_TOKENS=${STAGE1_NUM_DRAFT_TOKENS:-256}
OUTPUT_DIR_SUFFIX=${OUTPUT_DIR_SUFFIX:-}
if [ -n "$OUTPUT_DIR_SUFFIX" ]; then
  OUTPUT_DIR="${OUTPUT_DIR}_${OUTPUT_DIR_SUFFIX}"
fi

# --- Request range: slice input dataset for parallel execution ---
IS_PARTIAL=""
if [ -n "$REQ_START" ] && [ -n "$REQ_END" ]; then
  IS_PARTIAL=1
  OUTPUT_DIR="${OUTPUT_DIR}_req${REQ_START}-${REQ_END}"
fi
mkdir -p "$OUTPUT_DIR"

# Verify input file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "ERROR: Input file not found: $INPUT_FILE"
  echo "Run the appropriate prepare script first:"
  echo "  bfcl_v3:   python3 simulation/scripts/prepare_bfcl_data.py --benchmark v3"
  echo "  bfcl_v4:   python3 simulation/scripts/prepare_bfcl_data.py --benchmark v4"
  echo "  specbench: python3 simulation/scripts/prepare_specbench_data.py"
  echo "  swebench:  Collect trajectories to data/swebench/trajectories.jsonl"
  exit 1
fi

# Slice the input dataset if range specified
ORIGINAL_INPUT_FILE="$INPUT_FILE"
if [ -n "$IS_PARTIAL" ]; then
  SLICED_INPUT="$OUTPUT_DIR/input_slice.jsonl"
  python3 -c "
import sys
lines = open('$INPUT_FILE').readlines()
start, end = $REQ_START, $REQ_END
sliced = lines[start:end]
with open('$SLICED_INPUT', 'w') as f:
    f.writelines(sliced)
print(f'Requests [{start}:{end}]: {len(sliced)}/{len(lines)} selected', file=sys.stderr)
"
  INPUT_FILE="$SLICED_INPUT"
fi

echo "======================================"
echo "Oracle Pipeline: $BENCHMARK + $MODEL_PRESET"
echo "======================================"
echo "Model: $MODEL"
echo "Draft: $DRAFT_MODEL"
echo "TP: $TP_SIZE"
echo "Input: $INPUT_FILE"
if [ -n "$IS_PARTIAL" ]; then
  echo "Range: requests [$REQ_START:$REQ_END)"
fi
echo "Output: $OUTPUT_DIR"
echo ""

NUM_REQ_FLAG=""
if [ -n "$NUM_REQUESTS" ]; then
  NUM_REQ_FLAG="--num-requests $NUM_REQUESTS"
fi

wait_for_server() {
  echo "Waiting for server on port $PORT..."
  for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
      echo "Server ready!"
      return 0
    fi
    sleep 3
  done
  echo "ERROR: Server failed to start within 360s"
  return 1
}

SERVER_PID=""
kill_server() {
  if [ -n "$SERVER_PID" ]; then
    echo "Stopping server (PID=$SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    SERVER_PID=""
  fi
  sleep 3
}

# ============================================================
# Stage 1: EAGLE3 Oracle Vanilla (multi-GPU parallel)
# ============================================================
echo ""
echo "=== Stage 1: EAGLE3 Oracle Vanilla ==="

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
bash simulation/scripts/run_parallel_stage1.sh \
  "$INPUT_FILE" "$OUTPUT_DIR" "$MODEL" "$DRAFT_MODEL" \
  "$AGENT_MODULE" "$NUM_GPUS" \
  $TEMP_FLAG $NUM_REQ_FLAG $MAX_ITER_FLAG

# ============================================================
# Stage 2: Extract Trajectory
# ============================================================
echo ""
echo "=== Stage 2: Extract Trajectory ==="

python3 -m simulation.pipeline.extract_trajectory \
  --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
  --output "$OUTPUT_DIR/trajectory.json"

# ============================================================
# Stage 3: Draft Token Collection (3a: Suffix, 3b: Draft Model, 3c: MTP)
# ============================================================
# Stage 3b semantics: --model is the draft LM, so target-model flag is
# carried separately. Mirror DATASET_FLAG structure (specbench includes
# --dataset, others don't).
DM_DATASET_FLAG="--target-model $MODEL"
if [ "$BENCHMARK" = "specbench" ]; then
  DM_DATASET_FLAG="--dataset $INPUT_FILE $DM_DATASET_FLAG"
fi

# ---- Stage 3a: Suffix decoding (common) ----
echo ""
echo "=== Stage 3a: Suffix Decoding ==="
python3 -m simulation.pipeline.collect_suffix_drafts \
  --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
  --output "$OUTPUT_DIR/suffix_drafts.jsonl" \
  $DATASET_FLAG

# ---- Stage 3b: Draft model (if DRAFT_LM is set) ----
DRAFT_LM=${DRAFT_LM:-}
DM_FLAG=""
if [ -n "$DRAFT_LM" ]; then
  echo ""
  echo "=== Stage 3b: Draft Model ($DRAFT_LM) ==="
  NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
  bash simulation/scripts/run_parallel_draft_model.sh \
    "$OUTPUT_DIR/agent_results_eagle3.json" \
    "$OUTPUT_DIR/draft_model_drafts.jsonl" \
    "$DRAFT_LM" \
    "$NUM_GPUS" 16 \
    $DM_DATASET_FLAG
  DM_FLAG="--draft-model-drafts $OUTPUT_DIR/draft_model_drafts.jsonl"
else
  echo ""
  echo "=== Stage 3b: SKIPPED (DRAFT_LM unset) ==="
fi

# ---- Stage 3c: MTP replay (glm4_flash only; skip specbench) ----
MTP_FLAG=""
if [ "$BENCHMARK" != "specbench" ]; then
  python3 simulation/scripts/replay_oracle.py \
    --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
    --output /dev/null \
    --trajectory-output "$OUTPUT_DIR/replay_trajectory.json" 2>/dev/null

  if [ "$MODEL_PRESET" = "glm4_flash" ]; then
    echo ""
    echo "=== Stage 3c: MTP Oracle Replay ==="

    export SGLANG_ORACLE_REPLAY="$OUTPUT_DIR/replay_trajectory.json"
    python3 -m simulation.oracle.install_hook

    python3 -m sglang.launch_server \
      --model-path "$MODEL" \
      --tp-size $TP_SIZE \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 3 \
      --speculative-eagle-topk 4 \
      --speculative-num-draft-tokens 16 \
      --mem-fraction-static $MEM_FRAC \
      --disable-cuda-graph \
      --watchdog-timeout 600 \
      --host 0.0.0.0 --port $PORT \
      > /tmp/sglang_pipeline.log 2>&1 &
    SERVER_PID=$!

    wait_for_server || exit 1

    python3 simulation/scripts/replay_oracle.py \
      --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
      --output "$OUTPUT_DIR/agent_results_mtp.json" \
      --server-url "http://localhost:$PORT" \
      --model "$MODEL" \
      --num-workers 1

    kill_server
    unset SGLANG_ORACLE_REPLAY
    MTP_FLAG="--mtp-agent-results $OUTPUT_DIR/agent_results_mtp.json"
  else
    echo ""
    echo "=== Stage 3c: SKIPPED (model has no MTP heads) ==="
  fi
fi

# ============================================================
# Stage 4: Collect Union Trie (merge EAGLE3 + Suffix + [DM] + [MTP])
# ============================================================
if [ "$UNION_TRIE" = "0" ]; then
  echo ""
  echo "=== Stage 4: SKIPPED (UNION_TRIE=0) ==="
else
  echo ""
  echo "=== Stage 4: Collect Union Trie ==="
  python3 -m simulation.pipeline.collect_union_trie \
    --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
    --suffix-drafts "$OUTPUT_DIR/suffix_drafts.jsonl" \
    $DM_FLAG \
    $MTP_FLAG \
    --output "$OUTPUT_DIR/union_trie_data.jsonl" \
    $DATASET_FLAG
fi

# ============================================================
# Stage 5: Collect Target Model p_t (needs both UNION_TRIE=1 and EU_ORACLE=1)
# ============================================================
if [ "$UNION_TRIE" = "0" ]; then
  echo ""
  echo "=== Stage 5: SKIPPED (UNION_TRIE=0) ==="
elif [ "$EU_ORACLE" = "0" ]; then
  echo ""
  echo "=== Stage 5: SKIPPED (EU_ORACLE=0) ==="
else
  echo ""
  echo "=== Stage 5: Collect p_t ==="
  NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
  bash simulation/scripts/run_parallel_p_t.sh \
    "$OUTPUT_DIR/union_trie_data.jsonl" \
    "$OUTPUT_DIR/union_trie_data_with_pt.jsonl" \
    "$MODEL" \
    "$NUM_GPUS"
fi

# ============================================================
# Stage 6: Oracle Simulation
# ============================================================
echo ""
echo "=== Stage 6: Oracle Simulation ==="

LATENCY_FLAG=""
if [ ! -f "$OUTPUT_DIR/latency_config.json" ] && [ -f "simulation/results/${MODEL_SHORT}/latency_config.json" ]; then
  cp "simulation/results/${MODEL_SHORT}/latency_config.json" "$OUTPUT_DIR/latency_config.json"
fi
if [ -f "$OUTPUT_DIR/latency_config.json" ]; then
  LATENCY_FLAG="--latency-config $OUTPUT_DIR/latency_config.json"
fi

# Stage 6 input + method flags depend on UNION_TRIE / EU_ORACLE.
#  UNION_TRIE=0               → assemble per-proposer on the fly from
#                               Stage 1/3 artifacts, skip union_trie_* + EU
#  UNION_TRIE=1, EU_ORACLE=0  → read union_trie_data.jsonl, skip EU
#  UNION_TRIE=1, EU_ORACLE=1  → read union_trie_data_with_pt.jsonl, run EU
if [ "$UNION_TRIE" = "0" ]; then
  SIM_INPUT_FLAGS=(
    --agent-results "$OUTPUT_DIR/agent_results_eagle3.json"
    --suffix-drafts "$OUTPUT_DIR/suffix_drafts.jsonl"
  )
  if [ -n "$DRAFT_LM" ]; then
    SIM_INPUT_FLAGS+=(--draft-model-drafts "$OUTPUT_DIR/draft_model_drafts.jsonl")
  fi
  if [ -n "$MTP_FLAG" ]; then
    SIM_INPUT_FLAGS+=(--mtp-agent-results "$OUTPUT_DIR/agent_results_mtp.json")
  fi
  # Forward the same $DATASET_FLAG (--model + optional --dataset) for
  # on-the-fly BFCL/SpecBench prompt reconstruction.
  SIM_INPUT_FLAGS+=($DATASET_FLAG)
  METHOD_FLAGS=(--no-union-trie)
  PT_KEY="p_t_oracle"
elif [ "$EU_ORACLE" = "0" ]; then
  SIM_INPUT_FLAGS=(--union-trie-data "$OUTPUT_DIR/union_trie_data.jsonl")
  METHOD_FLAGS=()
  PT_KEY="p_t_oracle"
else
  SIM_INPUT_FLAGS=(--union-trie-data "$OUTPUT_DIR/union_trie_data_with_pt.jsonl")
  METHOD_FLAGS=(--enable-eu)
  PT_KEY="p_t"
fi

python3 -m simulation.evaluation.run_tree_oracle_sim \
  "${SIM_INPUT_FLAGS[@]}" \
  --budgets 1,2,4,8,16,32,64,128,256,512 \
  --p-t-key "$PT_KEY" \
  --output "$OUTPUT_DIR/tree_oracle_sim.json" \
  "${METHOD_FLAGS[@]}" \
  --print-summary \
  $LATENCY_FLAG

echo ""
echo "======================================"
echo "Pipeline complete: $BENCHMARK + $MODEL_PRESET"
echo "Results: $OUTPUT_DIR/"
echo "======================================"
