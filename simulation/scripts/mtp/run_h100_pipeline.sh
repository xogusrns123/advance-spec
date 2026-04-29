#!/usr/bin/env bash
# End-to-end H100 MTP draft collection for the 3 production workloads.
# Launches one server per workload sequentially (single H100), runs the
# client, then post-processes into per-position posacc JSONs.
#
# Required env: MODEL=<hf-id-or-path>
# Optional:     MTP_STEPS, MTP_TOPK, MTP_BUDGET (defaults 8/8/128),
#               MAX_QUESTIONS (default 80), MAX_NEW_TOKENS (default 512),
#               PORT (default 31010), GPU_ID (default 0).
#
# Output:
#   simulation/results/qwen35_mtp/mtp_drafts_<wl>.jsonl  (raw collection)
#   simulation/results/qwen35_mtp/posacc_mtp_<wl>.json   (per-position metric)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

MODEL="${MODEL:?set MODEL=<hf-id-or-path>}"
MTP_STEPS="${MTP_STEPS:-8}"
MTP_TOPK="${MTP_TOPK:-8}"
MTP_BUDGET="${MTP_BUDGET:-128}"
MAX_QUESTIONS="${MAX_QUESTIONS:-80}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
PORT="${PORT:-31010}"
GPU_ID="${GPU_ID:-0}"

OUT_DIR=simulation/results/qwen35_mtp
mkdir -p "$OUT_DIR"
LOG_DIR=/tmp/mtp_h100_logs
mkdir -p "$LOG_DIR"

declare -A WL_DATASET=(
  [specbench]=data/specbench/dataset.jsonl
  [bfcl_v4]=data/bfcl_agent/dataset_stratified_interleaved.jsonl
  [swebench_verified]=data/swebench_verified/dataset_interleaved.jsonl
)
WORKLOADS=(${WORKLOADS:-specbench bfcl_v4 swebench_verified})

ts() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST'; }

for wl in "${WORKLOADS[@]}"; do
  ds="${WL_DATASET[$wl]:-}"
  [ -n "$ds" ] || { echo "no dataset for $wl"; continue; }
  echo "[$(ts)] === $wl ==="

  ORACLE_LOG=/tmp/sglang_oracle_vanilla_${wl}.jsonl
  TIMING_LOG=/tmp/sglang_oracle_timing_${wl}.jsonl
  SERVER_LOG=$LOG_DIR/server_${wl}.log

  # Launch server in background.
  echo "[$(ts)] launching server..."
  ORACLE_LOG="$ORACLE_LOG" TIMING_LOG="$TIMING_LOG" \
  MODEL="$MODEL" MTP_STEPS="$MTP_STEPS" MTP_TOPK="$MTP_TOPK" \
  MTP_BUDGET="$MTP_BUDGET" PORT="$PORT" GPU_ID="$GPU_ID" \
    bash simulation/scripts/mtp/launch_h100_server.sh \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  # Wait for /get_model_info.
  for i in {1..120}; do
    sleep 5
    if curl -sf "http://localhost:${PORT}/get_model_info" > /dev/null 2>&1; then
      echo "[$(ts)] server ready (PID=$SERVER_PID, ${i}× 5s wait)"
      break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "[$(ts)] server died — see $SERVER_LOG"
      tail -50 "$SERVER_LOG"
      exit 1
    fi
  done

  if ! curl -sf "http://localhost:${PORT}/get_model_info" > /dev/null 2>&1; then
    echo "[$(ts)] server didn't come up in 600s — see $SERVER_LOG"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
  fi

  # Run client.
  RAW_OUT=$OUT_DIR/mtp_drafts_${wl}.jsonl
  CLIENT_LOG=$LOG_DIR/client_${wl}.log
  echo "[$(ts)] running client → $RAW_OUT"
  python3 -m simulation.scripts.mtp.collect_mtp_drafts \
    --workload "$wl" \
    --dataset "$ds" \
    --model "$MODEL" \
    --server-url "http://localhost:${PORT}" \
    --oracle-log "$ORACLE_LOG" \
    --max-questions "$MAX_QUESTIONS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output "$RAW_OUT" \
    > "$CLIENT_LOG" 2>&1

  # Stop server before next workload (single GPU).
  echo "[$(ts)] stopping server..."
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
  sleep 5

  # Post-process.
  POSACC_OUT=$OUT_DIR/posacc_mtp_${wl}.json
  echo "[$(ts)] post-processing → $POSACC_OUT"
  python3 -m simulation.scripts.mtp.compute_mtp_position_accepts \
    --input "$RAW_OUT" \
    --output "$POSACC_OUT" \
    --max-position 64 \
    --proposer-name mtp

  echo "[$(ts)] $wl done"
done

echo "[$(ts)] === ALL workloads done ==="
echo "Results in $OUT_DIR/posacc_mtp_*.json"
