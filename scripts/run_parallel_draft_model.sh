#!/usr/bin/env bash
# Run collect_draft_model on multiple GPUs in parallel using SGLang servers.
# Each GPU runs its own SGLang server with the draft model + prefix caching.
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
echo "Parallel draft model (SGLang)"
echo "  Input: $INPUT"
echo "  Draft model: $MODEL"
echo "  Max tokens: $MAX_TOKENS"
echo "  GPUs: $NUM_GPUS (requests: $N_REQS)"
echo "======================================"

# Start SGLang servers (one per GPU)
SRV_PIDS=()
BASE_PORT=31000
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((BASE_PORT + GPU_ID))
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m sglang.launch_server \
    --model-path "$MODEL" --tp-size 1 \
    --mem-fraction-static 0.85 --disable-cuda-graph \
    --host 0.0.0.0 --port $PORT \
    > /tmp/draft_model_server_${GPU_ID}.log 2>&1 &
  SRV_PIDS+=($!)
  echo "Server $GPU_ID starting on port $PORT (PID ${SRV_PIDS[-1]})"
done

# Wait for all servers
echo "Waiting for servers..."
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((BASE_PORT + GPU_ID))
  for i in $(seq 1 120); do
    curl -s "http://localhost:$PORT/health" > /dev/null 2>&1 && break
    sleep 3
  done
done
echo "All servers ready"

# Launch collection processes
PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((BASE_PORT + GPU_ID))
  SHARD_OUTPUT="${OUTPUT%.jsonl}_shard${GPU_ID}.jsonl"

  python3 scripts/collect_draft_model.py \
    --union-trie-data "$INPUT" \
    --output "$SHARD_OUTPUT" \
    --model "$MODEL" \
    --max-draft-tokens "$MAX_TOKENS" \
    --server-url "http://localhost:$PORT" \
    --shard "$GPU_ID/$NUM_GPUS" &
  PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} collection processes..."
FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait $PID; then FAILED=$((FAILED + 1)); fi
done

# Kill servers
for PID in "${SRV_PIDS[@]}"; do
  kill $PID 2>/dev/null || true
done
wait "${SRV_PIDS[@]}" 2>/dev/null || true

if [ $FAILED -gt 0 ]; then echo "ERROR: $FAILED processes failed"; exit 1; fi

# Merge shards
echo "Merging..."
> "$OUTPUT"
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  cat "${OUTPUT%.jsonl}_shard${GPU_ID}.jsonl" >> "$OUTPUT"
  rm -f "${OUTPUT%.jsonl}_shard${GPU_ID}.jsonl"
done
echo "Done: $(wc -l < "$OUTPUT") records → $OUTPUT"
