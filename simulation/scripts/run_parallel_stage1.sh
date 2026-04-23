#!/usr/bin/env bash
# Run Stage 1 (EAGLE3 oracle vanilla) on multiple GPUs in parallel.
# Each GPU runs its own SGLang server on a different port.
#
# Usage:
#   bash simulation/scripts/run_parallel_stage1.sh \
#       <input.jsonl> <output_dir> <model> <draft_model> \
#       <agent_module> [num_gpus=4] [extra_agent_args...]
#
# Example:
#   bash simulation/scripts/run_parallel_stage1.sh \
#       data/bfcl_agent/dataset.jsonl results/test \
#       Qwen/Qwen3-8B Tengyunw/qwen3_8b_eagle3 \
#       simulation.agents.bfcl_v4_agent 4 \
#       --max-iterations 5
set -euo pipefail

INPUT=${1:?}; OUTPUT_DIR=${2:?}; MODEL=${3:?}; DRAFT_MODEL=${4:?}; AGENT_MODULE=${5:?}
NUM_GPUS=${6:-4}; shift 6 || shift $#
EXTRA_ARGS="$*"

# EAGLE3 speculation tree shape — override via env to sweep steps/topk/budget
# without editing this script. Defaults reproduce the legacy setting.
STAGE1_TOPK=${STAGE1_TOPK:-8}
STAGE1_STEPS=${STAGE1_STEPS:-5}
STAGE1_NUM_DRAFT_TOKENS=${STAGE1_NUM_DRAFT_TOKENS:-256}

# GPU_IDS: comma-separated list of GPU indices (e.g. "0,2,3" to skip GPU 1)
# If not set, uses 0..NUM_GPUS-1
if [ -n "${GPU_IDS:-}" ]; then
  IFS=',' read -ra GPU_LIST <<< "$GPU_IDS"
  NUM_GPUS=${#GPU_LIST[@]}
else
  GPU_LIST=()
  for i in $(seq 0 $((NUM_GPUS - 1))); do GPU_LIST+=($i); done
fi

# Count input lines
TOTAL=$(wc -l < "$INPUT")
if [ "$TOTAL" -lt "$NUM_GPUS" ]; then
  NUM_GPUS=$TOTAL
  GPU_LIST=("${GPU_LIST[@]:0:$NUM_GPUS}")
fi

echo "======================================"
echo "Parallel Stage 1: $NUM_GPUS GPUs (${GPU_LIST[*]}), $TOTAL requests"
echo "======================================"

export SGLANG_ORACLE_VANILLA=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export TORCHINDUCTOR_COMPILE_THREADS=1
unset SGLANG_ORACLE_REPLAY SGLANG_ORACLE_VERIFY_TRIES

python3 -m simulation.oracle.install_hook

PIDS=()
for SHARD_IDX in $(seq 0 $((NUM_GPUS - 1))); do
  GPU_ID=${GPU_LIST[$SHARD_IDX]}
  PORT=$((${STAGE1_BASE_PORT:-30000} + SHARD_IDX))
  SHARD_DIR="$OUTPUT_DIR/_stage1_shard${SHARD_IDX}"
  mkdir -p "$SHARD_DIR"

  # Slice input for this shard
  python3 -c "
lines = open('$INPUT').readlines()
shard = [lines[i] for i in range(len(lines)) if i % $NUM_GPUS == $SHARD_IDX]
open('$SHARD_DIR/input.jsonl','w').writelines(shard)
print(f'Shard $SHARD_IDX (GPU $GPU_ID): {len(shard)} requests')
import sys; sys.stdout.flush()
"

  # Launch server + agent in background.
  # --tool-call-parser qwen25: required for LangChain-based agents
  # (swebench_agent) to receive OpenAI-format tool_calls. Without this,
  # Qwen3 emits <tool_call>…</tool_call> XML that bind_tools() can't
  # recognize, so the agent bails after 1 turn with no actions taken.
  # bfcl_v4 has its own native-syntax parser and isn't affected, but the
  # flag is harmless for it.
  # Source .env so HF_TOKEN (for gated repos like Llama) is available
  # to sglang's download path.
  if [ -f /workspace/.env ]; then set -a; source /workspace/.env; set +a; fi
  (
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m sglang.launch_server \
      --model-path "$MODEL" --tp-size 1 \
      --speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT_MODEL" \
      --speculative-num-steps $STAGE1_STEPS \
      --speculative-eagle-topk $STAGE1_TOPK \
      --speculative-num-draft-tokens $STAGE1_NUM_DRAFT_TOKENS \
      --tool-call-parser ${TOOL_CALL_PARSER:-qwen25} \
      --mem-fraction-static 0.85 --disable-cuda-graph --watchdog-timeout 600 \
      --host 0.0.0.0 --port $PORT > "$SHARD_DIR/server.log" 2>&1 &
    SRV_PID=$!

    # Wait for server
    for i in $(seq 1 120); do
      curl -s "http://localhost:$PORT/health" > /dev/null 2>&1 && break; sleep 3
    done

    python3 -m $AGENT_MODULE \
      --url "http://localhost:$PORT/v1" --model "$MODEL" \
      --input-file "$SHARD_DIR/input.jsonl" \
      --output-file "$SHARD_DIR/agent_results.json" \
      --num-workers 1 $EXTRA_ARGS > "$SHARD_DIR/agent.log" 2>&1

    kill $SRV_PID 2>/dev/null || true
    wait $SRV_PID 2>/dev/null || true
  ) &
  PIDS+=($!)
done

echo "Waiting for $NUM_GPUS shards..."
FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait $PID; then FAILED=$((FAILED + 1)); fi
done
if [ $FAILED -gt 0 ]; then
  echo "WARNING: $FAILED shards failed — server logs preserved under $OUTPUT_DIR/_stage1_shard*/"
fi

# Merge results
echo "Merging..."
python3 -c "
import json, sys

output_dir = '$OUTPUT_DIR'
num_gpus = $NUM_GPUS
merged_questions = []
total_tokens = 0
total_oracle = 0

for shard_idx in range(num_gpus):
    shard_path = f'{output_dir}/_stage1_shard{shard_idx}/agent_results.json'
    try:
        d = json.load(open(shard_path))
        merged_questions.extend(d.get('questions', []))
        m = d.get('metadata', {})
        total_tokens += m.get('total_tokens', 0)
        total_oracle += m.get('total_oracle_entries', 0)
    except Exception as e:
        print(f'WARN: shard {shard_idx} failed: {e}', file=sys.stderr)

output = {
    'metadata': {
        'model': '$MODEL',
        'num_requests': len(merged_questions),
        'total_tokens': total_tokens,
        'total_oracle_entries': total_oracle,
        'oracle_enabled': True,
    },
    'questions': merged_questions,
}
with open(f'{output_dir}/agent_results_eagle3.json', 'w') as f:
    json.dump(output, f)
print(f'Merged: {len(merged_questions)} requests, {total_tokens} tokens, {total_oracle} oracle')
"

# Cleanup shard dirs only if all shards succeeded (preserve server.log / agent.log
# on failure so the user can diagnose).
if [ "$FAILED" -eq 0 ]; then
  for SHARD_IDX in $(seq 0 $((NUM_GPUS - 1))); do
    rm -rf "$OUTPUT_DIR/_stage1_shard${SHARD_IDX}"
  done
  echo "Stage 1 parallel done"
else
  echo "Stage 1 parallel done ($FAILED failures — shard dirs kept for inspection)"
  exit 1
fi
