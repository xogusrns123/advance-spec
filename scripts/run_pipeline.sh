#!/usr/bin/env bash
# Unified oracle simulation pipeline for all benchmarks.
#
# Usage:
#   bash scripts/run_pipeline.sh <benchmark> <model_preset> [num_requests]
#
# Benchmarks: bfcl, specbench, swebench
# Model presets: glm4_flash, qwen3_8b
#
# Examples:
#   bash scripts/run_pipeline.sh specbench qwen3_8b 5
#   bash scripts/run_pipeline.sh bfcl glm4_flash 10
#   bash scripts/run_pipeline.sh swebench qwen3_8b
set -euo pipefail

BENCHMARK=${1:?Usage: $0 <benchmark> <model_preset> [num_requests]}
MODEL_PRESET=${2:?Usage: $0 <benchmark> <model_preset> [num_requests]}
NUM_REQUESTS=${3:-}
PORT=${PORT:-30000}

# --- Model preset ---
case $MODEL_PRESET in
  glm4_flash)
    MODEL="zai-org/GLM-4.7-Flash"
    DRAFT_MODEL="thoughtworks/GLM-4.7-Flash-Eagle3"
    TP_SIZE=4
    MEM_FRAC=0.8
    ;;
  qwen3_8b)
    MODEL="Qwen/Qwen3-8B"
    DRAFT_MODEL="Tengyunw/qwen3_8b_eagle3"
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
  bfcl)
    AGENT_MODULE="hybrid_spec_decoding.analysis.bfcl_agent"
    INPUT_FILE="data/bfcl_multi_turn/dataset.jsonl"
    DATASET_FLAG="--model $MODEL"
    MAX_ITER_FLAG="--max-iterations 20"
    ;;
  specbench)
    AGENT_MODULE="hybrid_spec_decoding.analysis.specbench_agent"
    INPUT_FILE="data/specbench/dataset.jsonl"
    DATASET_FLAG="--dataset $INPUT_FILE --model $MODEL"
    MAX_ITER_FLAG=""
    ;;
  swebench)
    AGENT_MODULE="hybrid_spec_decoding.analysis.swebench_agent"
    INPUT_FILE="data/swebench/dataset.jsonl"
    DATASET_FLAG="--model $MODEL"
    MAX_ITER_FLAG="--max-iterations 15 --repos-dir data/swebench/repos"
    ;;
  *)
    echo "Unknown benchmark: $BENCHMARK (use bfcl, specbench, or swebench)"
    exit 1
    ;;
esac

MODEL_SHORT=$(echo $MODEL_PRESET | tr '[:upper:]' '[:lower:]')
OUTPUT_DIR="results/${MODEL_SHORT}/${BENCHMARK}"
mkdir -p "$OUTPUT_DIR"

echo "======================================"
echo "Oracle Pipeline: $BENCHMARK + $MODEL_PRESET"
echo "======================================"
echo "Model: $MODEL"
echo "Draft: $DRAFT_MODEL"
echo "TP: $TP_SIZE"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

NUM_REQ_FLAG=""
if [ -n "$NUM_REQUESTS" ]; then
  NUM_REQ_FLAG="--num-requests $NUM_REQUESTS"
fi

# Verify input file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "ERROR: Input file not found: $INPUT_FILE"
  echo "Run the appropriate prepare script first:"
  echo "  bfcl:      python3 scripts/prepare_bfcl_data.py"
  echo "  specbench: python3 scripts/prepare_specbench_data.py"
  echo "  swebench:  Collect trajectories to data/swebench/trajectories.jsonl"
  exit 1
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

kill_server() {
  echo "Stopping server..."
  pkill -f "sglang.launch_server" 2>/dev/null || true
  sleep 5
}

# ============================================================
# Stage 1: EAGLE3 Oracle Vanilla (Round 1)
# ============================================================
echo ""
echo "=== Stage 1: EAGLE3 Oracle Vanilla ==="

kill_server

export SGLANG_ORACLE_VANILLA=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size $TP_SIZE \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16 \
  --mem-fraction-static $MEM_FRAC \
  --disable-cuda-graph \
  --host 0.0.0.0 --port $PORT \
  > /tmp/sglang_pipeline.log 2>&1 &

wait_for_server || exit 1

python3 -m $AGENT_MODULE \
  --url http://localhost:$PORT/v1 \
  --model "$MODEL" \
  --input-file "$INPUT_FILE" \
  --output-file "$OUTPUT_DIR/agent_results_eagle3.json" \
  --temperature 0.0 \
  $NUM_REQ_FLAG $MAX_ITER_FLAG

kill_server

# ============================================================
# Stage 2: Extract Trajectory
# ============================================================
echo ""
echo "=== Stage 2: Extract Trajectory ==="

python3 -m hybrid_spec_decoding.analysis.extract_trajectory \
  --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
  --output "$OUTPUT_DIR/trajectory.json"

# ============================================================
# Stage 3: MTP Oracle Replay (Round 2)
# ============================================================
echo ""
echo "=== Stage 3: MTP Oracle Replay ==="

export SGLANG_ORACLE_REPLAY="$OUTPUT_DIR/trajectory.json"

python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size $TP_SIZE \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16 \
  --mem-fraction-static $MEM_FRAC \
  --disable-cuda-graph \
  --host 0.0.0.0 --port $PORT \
  > /tmp/sglang_pipeline.log 2>&1 &

wait_for_server || exit 1

python3 -m $AGENT_MODULE \
  --url http://localhost:$PORT/v1 \
  --model "$MODEL" \
  --input-file "$INPUT_FILE" \
  --output-file "$OUTPUT_DIR/agent_results_mtp.json" \
  --temperature 0.0 \
  --replay "$OUTPUT_DIR/agent_results_eagle3.json" \
  $NUM_REQ_FLAG

kill_server
unset SGLANG_ORACLE_REPLAY

# ============================================================
# Stage 4: Collect Union Trie (EAGLE3 + Suffix + MTP)
# ============================================================
echo ""
echo "=== Stage 4: Collect Union Trie ==="

python3 -m hybrid_spec_decoding.analysis.collect_union_trie \
  --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
  --mtp-agent-results "$OUTPUT_DIR/agent_results_mtp.json" \
  --output "$OUTPUT_DIR/union_trie_data.jsonl" \
  $DATASET_FLAG

# ============================================================
# Stage 5: Collect Target Model p_t
# ============================================================
echo ""
echo "=== Stage 5: Collect p_t ==="

python3 -m hybrid_spec_decoding.analysis.collect_target_probs \
  --union-trie-data "$OUTPUT_DIR/union_trie_data.jsonl" \
  --output "$OUTPUT_DIR/union_trie_data_with_pt.jsonl" \
  --model "$MODEL"

# ============================================================
# Stage 6: Oracle Simulation
# ============================================================
echo ""
echo "=== Stage 6: Oracle Simulation ==="

# Use existing latency config if available, otherwise oracle-only
LATENCY_FLAG=""
if [ -f "$OUTPUT_DIR/latency_config.json" ]; then
  LATENCY_FLAG="--latency-config $OUTPUT_DIR/latency_config.json"
fi

python3 -m hybrid_spec_decoding.analysis.run_tree_oracle_sim \
  --union-trie-data "$OUTPUT_DIR/union_trie_data_with_pt.jsonl" \
  --budgets 1,2,4,8,16 \
  --p-t-key p_t \
  --output "$OUTPUT_DIR/tree_oracle_sim.json" \
  --print-summary \
  $LATENCY_FLAG

echo ""
echo "======================================"
echo "Pipeline complete: $BENCHMARK + $MODEL_PRESET"
echo "Results: $OUTPUT_DIR/"
echo "======================================"
