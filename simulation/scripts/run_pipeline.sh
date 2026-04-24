#!/usr/bin/env bash
# Unified oracle simulation pipeline for all benchmarks.
#
# Usage:
#   bash simulation/scripts/run_pipeline.sh <benchmark> <model_preset> [num_requests]
#
# Benchmarks: bfcl_v3, bfcl_v4, specbench, swebench
# Model presets: glm4_flash, qwen3_8b, qwen3_14b, qwen3_32b, llama3_8b
#
# Pipeline stages (all mandatory):
#   Stage 1  EAGLE3 Oracle Vanilla  (multi-GPU, writes agent_results_eagle3.json)
#   Stage 2  Draft Model Collection (if DRAFT_LM set, writes draft_model_drafts.jsonl)
#   Stage 3  Oracle Simulation      (writes tree_oracle_sim.json; suffix is
#                                    drawn live inside the simulator)
#
# Stage 1 EAGLE3 tree shape (override to sweep draft-tree configurations):
#   STAGE1_TOPK=8           tree branching factor per level (default 8)
#   STAGE1_STEPS=5          tree max depth (default 5)
#   STAGE1_NUM_DRAFT_TOKENS=256  total draft budget (default 256)
#   OUTPUT_DIR_SUFFIX=<str> appended to OUTPUT_DIR when sweeping so each
#                           run (e.g. steps=2/4/6/8) writes to a distinct dir.
#
# Request range (for parallel execution across machines):
#   REQ_START=0  REQ_END=50  bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b   # Machine A
#   REQ_START=50 REQ_END=100 bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b   # Machine B
#   # Then merge: simulation/scripts/merge_shards.sh simulation/results/qwen3_8b/bfcl_v4
#
# Examples:
#   bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b 5                        # first 5 requests
#   REQ_START=5 REQ_END=10 bash simulation/scripts/run_pipeline.sh bfcl_v4 qwen3_8b   # requests 5-9
set -euo pipefail

BENCHMARK=${1:?Usage: $0 <benchmark> <model_preset> [num_requests]}
MODEL_PRESET=${2:?Usage: $0 <benchmark> <model_preset> [num_requests]}
NUM_REQUESTS=${3:-}
PORT=${PORT:-30000}
REQ_START=${REQ_START:-}
REQ_END=${REQ_END:-}
NUM_WORKERS=${NUM_WORKERS:-1}

# --- Model preset ---
case $MODEL_PRESET in
  glm4_flash)
    MODEL="${MODEL:-zai-org/GLM-4.7-Flash}"
    DRAFT_MODEL="${DRAFT_MODEL:-thoughtworks/GLM-4.7-Flash-Eagle3}"
    DRAFT_LM="${DRAFT_LM:-}"
    ;;
  qwen3_8b)
    MODEL="${MODEL:-Qwen/Qwen3-8B}"
    DRAFT_MODEL="${DRAFT_MODEL:-AngelSlim/Qwen3-8B_eagle3}"
    DRAFT_LM="${DRAFT_LM:-Qwen/Qwen3-0.6B}"
    ;;
  qwen3_14b)
    MODEL="${MODEL:-Qwen/Qwen3-14B}"
    DRAFT_MODEL="${DRAFT_MODEL:-AngelSlim/Qwen3-14B_eagle3}"
    DRAFT_LM="${DRAFT_LM:-Qwen/Qwen3-0.6B}"
    ;;
  qwen3_32b)
    MODEL="${MODEL:-Qwen/Qwen3-32B}"
    DRAFT_MODEL="${DRAFT_MODEL:-Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3}"
    DRAFT_LM="${DRAFT_LM:-Qwen/Qwen3-0.6B}"
    ;;
  llama3_8b)
    MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
    DRAFT_MODEL="${DRAFT_MODEL:-yuhuili/EAGLE3-LLaMA3.1-Instruct-8B}"
    DRAFT_LM="${DRAFT_LM:-meta-llama/Llama-3.2-1B-Instruct}"
    export TOOL_CALL_PARSER=llama3
    ;;
  *)
    echo "Unknown model preset: $MODEL_PRESET (use glm4_flash, qwen3_8b, qwen3_14b, qwen3_32b, or llama3_8b)"
    exit 1
    ;;
esac
# Qwen models use qwen25 parser; override via preset for non-Qwen.
export TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen25}

# --- Benchmark config ---
case $BENCHMARK in
  bfcl_v3)
    AGENT_MODULE="simulation.agents.bfcl_agent"
    INPUT_FILE="data/bfcl_multi_turn/dataset.jsonl"
    DATASET_FLAG="--model $MODEL"
    MAX_ITER_FLAG="--max-iterations ${BFCL_MAX_ITER:-5}"
    TEMP_FLAG="--temperature 0.0"
    ;;
  bfcl_v4)
    AGENT_MODULE="simulation.agents.bfcl_v4_agent"
    # BFCL_V4_INPUT lets callers pre-filter the agent dataset (e.g. to web_search
    # only) to avoid the prereq-dependency expansion that balloons num_requests.
    INPUT_FILE="${BFCL_V4_INPUT:-data/bfcl_agent/dataset.jsonl}"
    DATASET_FLAG="--model $MODEL"
    MAX_ITER_FLAG="--max-iterations ${BFCL_MAX_ITER:-5}"
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
    # SWE_MAX_ITER overrides the agent loop cap; 30 is the full SWE-Bench budget
    # but an order of magnitude cheaper caps still produce enough trajectory
    # data for oracle simulation at a fraction of the wall time.
    MAX_ITER_FLAG="--max-iterations ${SWE_MAX_ITER:-30} --repos-dir data/swebench/repos"
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

# ============================================================
# Stage 1: EAGLE3 Oracle Vanilla (multi-GPU parallel)
# ============================================================
echo ""
echo "=== Stage 1: EAGLE3 Oracle Vanilla ==="

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
MAX_TOKENS_FLAG=""
# Only specbench_agent exposes --max-tokens as a CLI flag; bfcl/swebench agents
# hard-code their own caps. Keep override scoped to specbench until agents
# accept a unified flag.
if [ -n "${MAX_TOKENS_OVERRIDE:-}" ] && [ "$BENCHMARK" = "specbench" ]; then
  MAX_TOKENS_FLAG="--max-tokens $MAX_TOKENS_OVERRIDE"
fi
bash simulation/scripts/run_parallel_stage1.sh \
  "$INPUT_FILE" "$OUTPUT_DIR" "$MODEL" "$DRAFT_MODEL" \
  "$AGENT_MODULE" "$NUM_GPUS" \
  $TEMP_FLAG $NUM_REQ_FLAG $MAX_ITER_FLAG $MAX_TOKENS_FLAG

# ============================================================
# Stage 2: Draft Model Collection (if DRAFT_LM is set)
# ============================================================
# --model is the draft LM (target-model flag carried separately). Mirror
# DATASET_FLAG structure (specbench includes --dataset, others don't).
DM_DATASET_FLAG="--target-model $MODEL"
if [ "$BENCHMARK" = "specbench" ]; then
  DM_DATASET_FLAG="--dataset $INPUT_FILE $DM_DATASET_FLAG"
fi

DM_FLAG=""
if [ -n "$DRAFT_LM" ]; then
  echo ""
  echo "=== Stage 2: Draft Model ($DRAFT_LM) ==="
  NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
  bash simulation/scripts/run_parallel_draft_model.sh \
    "$OUTPUT_DIR/agent_results_eagle3.json" \
    "$OUTPUT_DIR/draft_model_drafts.jsonl" \
    "$DRAFT_LM" \
    "$NUM_GPUS" "${STAGE2_MAX_TOKENS:-${STAGE3B_MAX_TOKENS:-16}}" \
    $DM_DATASET_FLAG
  DM_FLAG="--draft-model-drafts $OUTPUT_DIR/draft_model_drafts.jsonl"
else
  echo ""
  echo "=== Stage 2: SKIPPED (DRAFT_LM unset) ==="
fi

# ============================================================
# Stage 3: Oracle Simulation
# ============================================================
echo ""
echo "=== Stage 3: Oracle Simulation ==="

LATENCY_FLAG=""
# Preferred location: simulation/config/latency/<preset>.json (committed).
# Fallback: simulation/results/<preset>/latency_config.json (local-only,
# from legacy measurement runs).
if [ ! -f "$OUTPUT_DIR/latency_config.json" ]; then
  if [ -f "simulation/config/latency/${MODEL_SHORT}.json" ]; then
    cp "simulation/config/latency/${MODEL_SHORT}.json" "$OUTPUT_DIR/latency_config.json"
  elif [ -f "simulation/results/${MODEL_SHORT}/latency_config.json" ]; then
    cp "simulation/results/${MODEL_SHORT}/latency_config.json" "$OUTPUT_DIR/latency_config.json"
  fi
fi
if [ -f "$OUTPUT_DIR/latency_config.json" ]; then
  LATENCY_FLAG="--latency-config $OUTPUT_DIR/latency_config.json"
fi

python3 -m simulation.evaluation.run_tree_oracle_sim \
  --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
  $DM_FLAG \
  $DATASET_FLAG \
  --budgets "${SIM_BUDGETS:-1,2,4,8,16,32,64,128,256,512}" \
  --output "$OUTPUT_DIR/tree_oracle_sim.json" \
  --print-summary \
  $LATENCY_FLAG

echo ""
echo "======================================"
echo "Pipeline complete: $BENCHMARK + $MODEL_PRESET"
echo "Results: $OUTPUT_DIR/"
echo "======================================"
