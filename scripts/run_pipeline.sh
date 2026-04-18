#!/usr/bin/env bash
# Unified oracle simulation pipeline for all benchmarks.
#
# Usage:
#   bash scripts/run_pipeline.sh <benchmark> <model_preset> [num_requests]
#
# Benchmarks: bfcl_v3, bfcl_v4, specbench, swebench
# Model presets: glm4_flash, qwen3_8b
#
# Request range (for parallel execution across machines):
#   REQ_START=0  REQ_END=50  bash scripts/run_pipeline.sh bfcl_v4 glm4_flash  # Machine A
#   REQ_START=50 REQ_END=100 bash scripts/run_pipeline.sh bfcl_v4 glm4_flash  # Machine B
#   # Then merge: scripts/merge_shards.sh results/glm4_flash/bfcl_v4
#
# Examples:
#   bash scripts/run_pipeline.sh bfcl_v3 glm4_flash 10   # first 10 requests
#   bash scripts/run_pipeline.sh bfcl_v4 glm4_flash 5    # first 5 requests
#   REQ_START=5 REQ_END=10 bash scripts/run_pipeline.sh bfcl_v4 glm4_flash  # requests 5-9
set -euo pipefail

BENCHMARK=${1:?Usage: $0 <benchmark> <model_preset> [num_requests]}
MODEL_PRESET=${2:?Usage: $0 <benchmark> <model_preset> [num_requests]}
NUM_REQUESTS=${3:-}
PORT=${PORT:-30000}
REQ_START=${REQ_START:-}
REQ_END=${REQ_END:-}
NUM_WORKERS=${NUM_WORKERS:-1}
SKIP_PT=${SKIP_PT:-}
ENABLE_EU=${ENABLE_EU:-}

# --- Model preset ---
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
  *)
    echo "Unknown model preset: $MODEL_PRESET (use glm4_flash or qwen3_8b)"
    exit 1
    ;;
esac

# --- Benchmark config ---
case $BENCHMARK in
  bfcl_v3)
    AGENT_MODULE="hybrid_spec_decoding.analysis.bfcl_agent"
    INPUT_FILE="data/bfcl_multi_turn/dataset.jsonl"
    DATASET_FLAG="--model $MODEL"
    MAX_ITER_FLAG="--max-iterations 5"
    TEMP_FLAG="--temperature 0.0"
    ;;
  bfcl_v4)
    AGENT_MODULE="hybrid_spec_decoding.analysis.bfcl_v4_agent"
    INPUT_FILE="data/bfcl_agent/dataset.jsonl"
    DATASET_FLAG="--model $MODEL"
    MAX_ITER_FLAG="--max-iterations 5"
    TEMP_FLAG=""
    ;;
  specbench)
    AGENT_MODULE="hybrid_spec_decoding.analysis.specbench_agent"
    INPUT_FILE="data/specbench/dataset.jsonl"
    DATASET_FLAG="--dataset $INPUT_FILE --model $MODEL"
    MAX_ITER_FLAG=""
    TEMP_FLAG="--temperature 0.0"
    ;;
  swebench)
    AGENT_MODULE="hybrid_spec_decoding.analysis.swebench_agent"
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
OUTPUT_DIR="results/${MODEL_SHORT}/${BENCHMARK}"

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
  echo "  bfcl_v3:   python3 scripts/prepare_bfcl_data.py --benchmark v3"
  echo "  bfcl_v4:   python3 scripts/prepare_bfcl_data.py --benchmark v4"
  echo "  specbench: python3 scripts/prepare_specbench_data.py"
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
bash scripts/run_parallel_stage1.sh \
  "$INPUT_FILE" "$OUTPUT_DIR" "$MODEL" "$DRAFT_MODEL" \
  "$AGENT_MODULE" "$NUM_GPUS" \
  $TEMP_FLAG $NUM_REQ_FLAG $MAX_ITER_FLAG

# ============================================================
# Stage 2: Extract Trajectory
# ============================================================
echo ""
echo "=== Stage 2: Extract Trajectory ==="

python3 -m hybrid_spec_decoding.analysis.extract_trajectory \
  --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
  --output "$OUTPUT_DIR/trajectory.json"

# ============================================================
# Stage 3: MTP Oracle Replay (skip if model has no MTP heads)
# ============================================================
MTP_FLAG=""
if [ "$BENCHMARK" != "specbench" ]; then
  # Check if model supports MTP by trying to build trajectory for replay
  python3 scripts/replay_oracle.py \
    --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
    --output /dev/null \
    --trajectory-output "$OUTPUT_DIR/replay_trajectory.json" 2>/dev/null

  # Only run MTP if model supports NEXTN/EAGLE algorithm
  # (Qwen3-8B has no MTP heads, GLM-4.7-Flash does)
  if [ "$MODEL_PRESET" = "glm4_flash" ]; then
    echo ""
    echo "=== Stage 3: MTP Oracle Replay ==="

    export SGLANG_ORACLE_REPLAY="$OUTPUT_DIR/replay_trajectory.json"
    python3 -m hybrid_spec_decoding.sglang_integration.install_hook

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

    python3 scripts/replay_oracle.py \
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
    echo "=== Stage 3: SKIPPED (model has no MTP heads) ==="
  fi
fi

# ============================================================
# Stage 4: Collect Union Trie (EAGLE3 + Suffix [+ MTP])
# ============================================================
echo ""
echo "=== Stage 4: Collect Union Trie ==="

python3 -m hybrid_spec_decoding.analysis.collect_union_trie \
  --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
  $MTP_FLAG \
  --output "$OUTPUT_DIR/union_trie_data.jsonl" \
  $DATASET_FLAG

# ============================================================
# Stage 4b: Collect Draft Model proposals (if DRAFT_LM is set)
# ============================================================
DRAFT_LM=${DRAFT_LM:-}
PT_INPUT="$OUTPUT_DIR/union_trie_data.jsonl"

if [ -n "$DRAFT_LM" ]; then
  echo ""
  echo "=== Stage 4b: Draft Model ($DRAFT_LM) ==="
  NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
  bash scripts/run_parallel_draft_model.sh \
    "$OUTPUT_DIR/union_trie_data.jsonl" \
    "$OUTPUT_DIR/union_trie_data_with_dm.jsonl" \
    "$DRAFT_LM" \
    "$NUM_GPUS"
  PT_INPUT="$OUTPUT_DIR/union_trie_data_with_dm.jsonl"
fi

# ============================================================
# Stage 5: Collect Target Model p_t (skip with SKIP_PT=1)
# ============================================================
SIM_INPUT="$PT_INPUT"
if [ -n "$SKIP_PT" ]; then
  echo ""
  echo "=== Stage 5: SKIPPED (SKIP_PT=1) ==="
else
  echo ""
  echo "=== Stage 5: Collect p_t ==="
  NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
  bash scripts/run_parallel_p_t.sh \
    "$PT_INPUT" \
    "$OUTPUT_DIR/union_trie_data_with_pt.jsonl" \
    "$MODEL" \
    "$NUM_GPUS"
  SIM_INPUT="$OUTPUT_DIR/union_trie_data_with_pt.jsonl"
fi

# ============================================================
# Stage 6: Oracle Simulation
# ============================================================
echo ""
echo "=== Stage 6: Oracle Simulation ==="

LATENCY_FLAG=""
if [ ! -f "$OUTPUT_DIR/latency_config.json" ] && [ -f "results/${MODEL_SHORT}/latency_config.json" ]; then
  cp "results/${MODEL_SHORT}/latency_config.json" "$OUTPUT_DIR/latency_config.json"
fi
if [ -f "$OUTPUT_DIR/latency_config.json" ]; then
  LATENCY_FLAG="--latency-config $OUTPUT_DIR/latency_config.json"
fi

EU_FLAG=""
if [ -n "$ENABLE_EU" ]; then
  EU_FLAG="--enable-eu"
fi

python3 -m hybrid_spec_decoding.analysis.run_tree_oracle_sim \
  --union-trie-data "$SIM_INPUT" \
  --budgets 1,2,4,8,16,32,64,128,256,512 \
  --p-t-key p_t \
  --output "$OUTPUT_DIR/tree_oracle_sim.json" \
  $EU_FLAG \
  --print-summary \
  $LATENCY_FLAG

echo ""
echo "======================================"
echo "Pipeline complete: $BENCHMARK + $MODEL_PRESET"
echo "Results: $OUTPUT_DIR/"
echo "======================================"
