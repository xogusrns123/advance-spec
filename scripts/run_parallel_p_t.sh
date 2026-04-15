#!/usr/bin/env bash
# Run collect_target_probs on 4 GPUs in parallel.
#
# Each GPU loads the model independently and processes 1/4 of the records.
# Per-request checkpointing prevents RAM OOM.
#
# Usage:
#   bash scripts/run_parallel_p_t.sh \
#       results/.../union_trie_data.jsonl \
#       results/.../union_trie_data_with_pt.jsonl \
#       Qwen/Qwen3-8B \
#       [num_gpus=4]
set -euo pipefail

INPUT=${1:?Usage: $0 <input.jsonl> <output.jsonl> <model> [num_gpus]}
OUTPUT=${2:?Usage: $0 <input.jsonl> <output.jsonl> <model> [num_gpus]}
MODEL=${3:?Usage: $0 <input.jsonl> <output.jsonl> <model> [num_gpus]}
NUM_GPUS=${4:-4}

OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"

# Cap NUM_GPUS to number of unique requests
N_REQS=$(python3 -c "
import json
rids = set()
with open('$INPUT') as f:
    for line in f:
        if line.strip():
            rids.add(json.loads(line).get('request_id',''))
print(len(rids))
")
if [ "$N_REQS" -lt "$NUM_GPUS" ]; then
  NUM_GPUS=$N_REQS
fi

echo "======================================"
echo "Parallel p_t collection"
echo "  Input: $INPUT"
echo "  Model: $MODEL"
echo "  GPUs: $NUM_GPUS (requests: $N_REQS)"
echo "======================================"

# Launch one process per GPU
PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  SHARD_OUTPUT="${OUTPUT%.jsonl}_shard${GPU_ID}.jsonl"
  echo "Starting shard $GPU_ID/$NUM_GPUS on GPU $GPU_ID → $SHARD_OUTPUT"

  CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m hybrid_spec_decoding.analysis.collect_target_probs \
    --union-trie-data "$INPUT" \
    --output "$SHARD_OUTPUT" \
    --model "$MODEL" \
    --device cuda \
    --shard "$GPU_ID/$NUM_GPUS" \
    --checkpoint-every 1 &
  PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} processes..."

# Wait for all
FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait $PID; then
    echo "Process $PID failed"
    FAILED=$((FAILED + 1))
  fi
done

if [ $FAILED -gt 0 ]; then
  echo "ERROR: $FAILED processes failed"
  exit 1
fi

# Merge shards
echo ""
echo "Merging shards..."
> "$OUTPUT"
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  SHARD_OUTPUT="${OUTPUT%.jsonl}_shard${GPU_ID}.jsonl"
  cat "$SHARD_OUTPUT" >> "$OUTPUT"
  rm -f "$SHARD_OUTPUT"
done

TOTAL=$(wc -l < "$OUTPUT")
echo "Done: $TOTAL records → $OUTPUT"
