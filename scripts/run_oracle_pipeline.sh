#!/usr/bin/env bash
# Oracle Simulation Pipeline for GLM-4.7-Flash
#
# Two-round oracle vanilla collection + offline simulation:
#   Round 1: EAGLE3 oracle vanilla → trajectory + EAGLE3 drafts
#   Round 2: MTP oracle vanilla → same trajectory + MTP drafts (TODO)
#   Offline: Oracle simulation (88+ method comparison)
#
# Usage:
#   bash scripts/run_oracle_pipeline.sh [num_requests]
#
# Prerequisites:
#   - Run inside Docker container (sglang-bench)
#   - BFCL dataset at data/bfcl_multi_turn/dataset.jsonl
#   - SGLANG_ORACLE_VANILLA=1 (set automatically by this script)

set -euo pipefail
cd "$(dirname "$0")/.."

NUM_REQUESTS="${1:-10}"
MODEL="zai-org/GLM-4.7-Flash"
DRAFT_MODEL="thoughtworks/GLM-4.7-Flash-Eagle3"
OUTPUT_DIR="results/glm4_flash/oracle_vanilla"
PORT=30000

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " Oracle Pipeline: GLM-4.7-Flash"
echo " Requests: $NUM_REQUESTS"
echo " Output: $OUTPUT_DIR"
echo "=============================================="

# ── Round 1: EAGLE3 Oracle Vanilla ──────────────────────────────────

echo ""
echo "[Round 1] EAGLE3 oracle vanilla collection"
echo ""

# Install patches
export SGLANG_ORACLE_VANILLA=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python3 -m hybrid_spec_decoding.sglang_integration.install_hook

# Launch EAGLE3 server
echo "Starting EAGLE3 server..."
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 4 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.8 \
    --disable-cuda-graph \
    --host 0.0.0.0 --port $PORT &

SERVER_PID=$!

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died!" && exit 1
    fi
    sleep 5
done

# Run BFCL agent
echo ""
echo "Running BFCL agent ($NUM_REQUESTS requests)..."
python3 -m hybrid_spec_decoding.analysis.bfcl_agent \
    --url "http://localhost:$PORT/v1" \
    --model "$MODEL" \
    --input-file data/bfcl_multi_turn/dataset.jsonl \
    --output-file "$OUTPUT_DIR/agent_results.json" \
    --num-requests "$NUM_REQUESTS"

# Stop server
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# ── Round 2: MTP Oracle Replay ─────────────────────────────────────

echo ""
echo "[Round 2] MTP oracle replay"
echo ""

# Extract trajectory from Round 1
echo "Extracting trajectory from Round 1..."
python3 -m hybrid_spec_decoding.analysis.extract_trajectory \
    --agent-results "$OUTPUT_DIR/agent_results.json" \
    --output "$OUTPUT_DIR/trajectory.json"

# Launch MTP server with replay
echo "Starting MTP server (replay mode)..."
export SGLANG_ORACLE_REPLAY="$OUTPUT_DIR/trajectory.json"
python3 -m hybrid_spec_decoding.sglang_integration.install_hook
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 4 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.8 \
    --disable-cuda-graph \
    --host 0.0.0.0 --port $PORT &

SERVER_PID=$!

# Wait for server
echo "Waiting for MTP server..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died!" && exit 1
    fi
    sleep 5
done

# Run BFCL agent (replay same requests)
echo ""
echo "Running BFCL agent (MTP replay, $NUM_REQUESTS requests)..."
python3 -m hybrid_spec_decoding.analysis.bfcl_agent \
    --url "http://localhost:$PORT/v1" \
    --model "$MODEL" \
    --input-file data/bfcl_multi_turn/dataset.jsonl \
    --output-file "$OUTPUT_DIR/agent_results_mtp.json" \
    --num-requests "$NUM_REQUESTS" \
    --replay "$OUTPUT_DIR/agent_results.json"

# Stop server
echo "Stopping MTP server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
unset SGLANG_ORACLE_REPLAY

# ── Offline: Oracle Simulation ──────────────────────────────────────

echo ""
echo "[Offline] Running oracle simulation..."
python3 -m hybrid_spec_decoding.analysis.run_oracle_sim \
    --agent-results "$OUTPUT_DIR/agent_results.json" \
    --output "$OUTPUT_DIR/oracle_sim.json" \
    --model "$MODEL" \
    --print-summary

echo ""
echo "=============================================="
echo " Done! Results in $OUTPUT_DIR"
echo "=============================================="
echo " - agent_results.json     : EAGLE3 oracle trajectories"
echo " - agent_results_mtp.json : MTP oracle trajectories (replay)"
echo " - trajectory.json        : Shared token trajectory"
echo " - oracle_sim.json        : 88+ method simulation"
echo "=============================================="
