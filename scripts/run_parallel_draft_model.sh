#!/usr/bin/env bash
# Run collect_draft_model on multiple GPUs in parallel.
#
# Usage:
#   bash scripts/run_parallel_draft_model.sh \
#       results/.../union_trie_data.jsonl \
#       results/.../union_trie_data_with_dm.jsonl \
#       Qwen/Qwen3-0.6B \
#       [num_gpus=4] [max_draft_tokens=16]
set -euo pipefail

INPUT=${1:?Usage: $0 <input.jsonl> <output.jsonl> <model> [num_gpus] [max_tokens]}
OUTPUT=${2:?}
MODEL=${3:?}
NUM_GPUS=${4:-4}
MAX_TOKENS=${5:-16}

# Cap to actual request count
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
echo "Parallel draft model collection"
echo "  Input: $INPUT"
echo "  Draft model: $MODEL"
echo "  Max tokens: $MAX_TOKENS"
echo "  GPUs: $NUM_GPUS (requests: $N_REQS)"
echo "======================================"

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  SHARD_OUTPUT="${OUTPUT%.jsonl}_shard${GPU_ID}.jsonl"
  echo "Starting shard $GPU_ID/$NUM_GPUS on GPU $GPU_ID"

  CUDA_VISIBLE_DEVICES=$GPU_ID python3 scripts/collect_draft_model.py \
    --union-trie-data "$INPUT" \
    --output "$SHARD_OUTPUT" \
    --model "$MODEL" \
    --max-draft-tokens "$MAX_TOKENS" \
    --device cuda \
    --shard "$GPU_ID/$NUM_GPUS" \
    --checkpoint-every 1 &
  PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} processes..."
FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait $PID; then FAILED=$((FAILED + 1)); fi
done
if [ $FAILED -gt 0 ]; then echo "ERROR: $FAILED failed"; exit 1; fi

echo "Merging..."
> "$OUTPUT"
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  cat "${OUTPUT%.jsonl}_shard${GPU_ID}.jsonl" >> "$OUTPUT"
  rm -f "${OUTPUT%.jsonl}_shard${GPU_ID}.jsonl"
done
echo "Done: $(wc -l < "$OUTPUT") records → $OUTPUT"
