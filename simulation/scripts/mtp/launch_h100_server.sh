#!/usr/bin/env bash
# Launch SGLang server on a single H100 (80 GB) with MTP/NEXTN spec
# decoding + Oracle-Vanilla draft logging.
#
# Per-step draft trees (token_ids, parents, etc.) are appended to
# $ORACLE_LOG (default /tmp/sglang_oracle_vanilla.jsonl) by the runtime
# patch in simulation/oracle/oracle_patch.py. Force-accept is on:
# every verify step commits exactly 1 bonus token, so output_token_ids
# are the greedy decode and each step's ground-truth-future is just the
# subsequent bonus tokens.
#
# Default config matches the user-requested grid:
#   speculative-num-steps         = 8
#   speculative-eagle-topk        = 8
#   speculative-num-draft-tokens  = 128 (the budget B)
#
# Override with env vars before invoking:
#   MODEL          (e.g. Qwen/Qwen3.5-9B or kakaocorp/kanana-2-1.5b)
#   MTP_STEPS      (default 8)
#   MTP_TOPK       (default 8)
#   MTP_BUDGET     (default 128)
#   PORT           (default 31010)
#   GPU_ID         (default 0)
#   ORACLE_LOG     (default /tmp/sglang_oracle_vanilla.jsonl)
#   TIMING_LOG     (default /tmp/sglang_oracle_timing.jsonl)
#   MEM_FRACTION   (default 0.85)
#
# Usage:
#   MODEL=Qwen/Qwen3.5-9B bash launch_h100_server.sh
#
# Server is launched in the foreground; ctrl-c to stop.
set -euo pipefail

MODEL="${MODEL:?set MODEL=<hf-id-or-path>}"
MTP_STEPS="${MTP_STEPS:-8}"
MTP_TOPK="${MTP_TOPK:-8}"
MTP_BUDGET="${MTP_BUDGET:-128}"
PORT="${PORT:-31010}"
GPU_ID="${GPU_ID:-0}"
ORACLE_LOG="${ORACLE_LOG:-/tmp/sglang_oracle_vanilla.jsonl}"
TIMING_LOG="${TIMING_LOG:-/tmp/sglang_oracle_timing.jsonl}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# Truncate logs so the byte-offset reader in collect_mtp_drafts.py
# starts at 0.
: > "$ORACLE_LOG"
: > "$TIMING_LOG"

echo "[$(date '+%H:%M:%S')] Launching MTP server"
echo "  model:    $MODEL"
echo "  spec:     steps=$MTP_STEPS topk=$MTP_TOPK budget=$MTP_BUDGET"
echo "  port:     $PORT  GPU: $GPU_ID"
echo "  oracle:   $ORACLE_LOG"
echo "  project:  $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Required env wiring:
#  - SGLANG_ORACLE_VANILLA=1 activates the runtime patch (force-accept
#    + per-step draft tree logging).
#  - SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 lets the EAGLE/MTP draft
#    head's max-position-embeddings be < target's.
#  - TORCHINDUCTOR_COMPILE_THREADS=1 prevents fork-bomb on torch.compile.
exec env \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  SGLANG_ORACLE_VANILLA=1 \
  SGLANG_ORACLE_LOG="$ORACLE_LOG" \
  SGLANG_ORACLE_TIMING_LOG="$TIMING_LOG" \
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  python3 -m simulation.oracle.install_hook \
    --model-path "$MODEL" \
    --speculative-algorithm NEXTN \
    --speculative-num-steps "$MTP_STEPS" \
    --speculative-eagle-topk "$MTP_TOPK" \
    --speculative-num-draft-tokens "$MTP_BUDGET" \
    --mem-fraction-static "$MEM_FRACTION" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --tp 1
