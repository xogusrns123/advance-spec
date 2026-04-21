#!/usr/bin/env bash
# Re-run Stages 3a + 4 + 6 for a completed pipeline directory. Stage 3b
# (draft model) and Stage 3c (MTP) outputs are reused as-is since they
# don't depend on suffix params. This is the common case when tuning
# suffix-decoding hyperparameters.
#
# Usage: bash simulation/scripts/rerun_from_stage4.sh <output_dir> <model_preset>
#
# Honors the same EU_ORACLE / UNION_TRIE toggles as run_pipeline.sh:
#   UNION_TRIE=0    (default) skip Stage 4 build; Stage 6 assembles
#                   per-proposer on-the-fly and skips union_trie_* + EU
#   UNION_TRIE=1    run Stage 4, enable union_trie_* methods in Stage 6
#   EU_ORACLE=0     (default) skip EU in Stage 6; reuse with_pt.jsonl if
#                   present; otherwise p_t_oracle
#   EU_ORACLE=1     require union_trie_data_with_pt.jsonl and enable EU
set -euo pipefail

OUTPUT_DIR=${1:?}
MODEL_PRESET=${2:?}
EU_ORACLE=${EU_ORACLE:-0}
UNION_TRIE=${UNION_TRIE:-0}

if [ "$EU_ORACLE" = "1" ] && [ "$UNION_TRIE" = "0" ]; then
  echo "ERROR: EU_ORACLE=1 requires UNION_TRIE=1."
  exit 1
fi

case $MODEL_PRESET in
  qwen3_8b)
    MODEL="Qwen/Qwen3-8B"
    ;;
  qwen3_14b)
    MODEL="Qwen/Qwen3-14B"
    ;;
  qwen3_32b)
    MODEL="Qwen/Qwen3-32B"
    ;;
  glm4_flash)
    MODEL="zai-org/GLM-4.7-Flash"
    ;;
  *)
    echo "Unknown model preset: $MODEL_PRESET"; exit 1 ;;
esac

if [[ "$OUTPUT_DIR" == *specbench* ]]; then
  DATASET_FLAG="--dataset data/specbench/dataset.jsonl --model $MODEL"
else
  DATASET_FLAG="--model $MODEL"
fi

echo "====================================="
echo "Re-run: $OUTPUT_DIR"
echo "UNION_TRIE=$UNION_TRIE  EU_ORACLE=$EU_ORACLE"
echo "====================================="

rm -f "$OUTPUT_DIR/suffix_drafts.jsonl"
rm -f "$OUTPUT_DIR/union_trie_data.jsonl"
rm -f "$OUTPUT_DIR/tree_oracle_sim.json"
if [ "$EU_ORACLE" = "1" ]; then
  rm -f "$OUTPUT_DIR/union_trie_data_with_pt.jsonl"
fi

if [ ! -f "$OUTPUT_DIR/latency_config.json" ] && [ -f "simulation/results/${MODEL_PRESET}/latency_config.json" ]; then
  cp "simulation/results/${MODEL_PRESET}/latency_config.json" "$OUTPUT_DIR/latency_config.json"
fi

# Stage 3a: Suffix (always)
echo "=== Stage 3a: Suffix Decoding ==="
python3 -m simulation.pipeline.collect_suffix_drafts \
  --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
  --output "$OUTPUT_DIR/suffix_drafts.jsonl" \
  $DATASET_FLAG

# Stage 4: Union Trie — skipped when UNION_TRIE=0
DM_FLAG=""
if [ -f "$OUTPUT_DIR/draft_model_drafts.jsonl" ]; then
  DM_FLAG="--draft-model-drafts $OUTPUT_DIR/draft_model_drafts.jsonl"
fi
MTP_FLAG=""
if [ -f "$OUTPUT_DIR/agent_results_mtp.json" ]; then
  MTP_FLAG="--mtp-agent-results $OUTPUT_DIR/agent_results_mtp.json"
fi

if [ "$UNION_TRIE" = "0" ]; then
  echo "=== Stage 4: SKIPPED (UNION_TRIE=0) ==="
else
  echo "=== Stage 4: Collect Union Trie ==="
  python3 -m simulation.pipeline.collect_union_trie \
    --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
    --suffix-drafts "$OUTPUT_DIR/suffix_drafts.jsonl" \
    $DM_FLAG \
    $MTP_FLAG \
    --output "$OUTPUT_DIR/union_trie_data.jsonl" \
    $DATASET_FLAG
fi

# Stage 5: reuse or re-collect p_t
if [ "$EU_ORACLE" = "1" ]; then
  PT_FILE="$OUTPUT_DIR/union_trie_data_with_pt.jsonl"
  if [ ! -f "$PT_FILE" ]; then
    echo "=== Stage 5: Collect p_t ==="
    NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
    bash simulation/scripts/run_parallel_p_t.sh \
      "$OUTPUT_DIR/union_trie_data.jsonl" \
      "$PT_FILE" \
      "$MODEL" \
      "$NUM_GPUS"
  else
    echo "=== Stage 5: SKIPPED (reusing $PT_FILE) ==="
  fi
fi

# Stage 6: Oracle sim
echo "=== Stage 6: Oracle Simulation ==="
if [ "$UNION_TRIE" = "0" ]; then
  SIM_INPUT_FLAGS=(
    --agent-results "$OUTPUT_DIR/agent_results_eagle3.json"
    --suffix-drafts "$OUTPUT_DIR/suffix_drafts.jsonl"
  )
  [ -n "$DM_FLAG" ] && SIM_INPUT_FLAGS+=($DM_FLAG)
  [ -n "$MTP_FLAG" ] && SIM_INPUT_FLAGS+=($MTP_FLAG)
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
  --latency-config "$OUTPUT_DIR/latency_config.json" \
  "${METHOD_FLAGS[@]}" \
  --print-summary

echo "DONE: $OUTPUT_DIR"
