#!/usr/bin/env bash
# Full pipeline integration test
# Round 1: EAGLE3 → Round 2: MTP replay → Union trie → p_t → Simulation
#
# Usage: docker exec -w /workspace sglang-bench bash scripts/test_full_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="zai-org/GLM-4.7-Flash"
DRAFT_MODEL="thoughtworks/GLM-4.7-Flash-Eagle3"
OUTPUT_DIR="results/glm4_flash/pipeline_test"
PORT=30000
NUM_REQUESTS=1
MAX_ITERATIONS=1

mkdir -p "$OUTPUT_DIR"

cleanup() {
    echo "Cleaning up..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    unset SGLANG_ORACLE_VANILLA SGLANG_ORACLE_REPLAY SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN
}
trap cleanup EXIT

echo "=============================================="
echo " FULL PIPELINE INTEGRATION TEST"
echo " 1 request, max 1 iteration"
echo "=============================================="

# ── Round 1: EAGLE3 ────────────────────────────────────────────────

echo ""
echo "[Round 1] EAGLE3 oracle vanilla"

export SGLANG_ORACLE_VANILLA=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python3 -m hybrid_spec_decoding.sglang_integration.install_hook

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

echo "Waiting for EAGLE3 server..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "EAGLE3 server died!" && exit 1
    fi
    sleep 5
done

echo "Running BFCL agent (Round 1)..."
python3 -m hybrid_spec_decoding.analysis.bfcl_agent \
    --url "http://localhost:$PORT/v1" \
    --model "$MODEL" \
    --input-file data/bfcl_multi_turn/dataset.jsonl \
    --output-file "$OUTPUT_DIR/agent_results_eagle3.json" \
    --num-requests "$NUM_REQUESTS" \
    --max-iterations "$MAX_ITERATIONS"

echo "Stopping EAGLE3 server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Verify Round 1
python3 -c "
import json
with open('$OUTPUT_DIR/agent_results_eagle3.json') as f:
    d = json.load(f)
q = d['questions'][0]
entries = sum(len(s.get('spec_decode',{}).get('oracle_vanilla_entries',[])) for s in q['agent_metrics']['steps'])
has_tree = any(
    e.get('eagle3_tree') is not None
    for s in q['agent_metrics']['steps']
    for e in s.get('spec_decode',{}).get('oracle_vanilla_entries',[])
)
print(f'Round 1: {entries} entries, has_tree={has_tree}')
"

# ── Extract trajectory ─────────────────────────────────────────────

echo ""
echo "Extracting trajectory..."
python3 -m hybrid_spec_decoding.analysis.extract_trajectory \
    --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
    --output "$OUTPUT_DIR/trajectory.json"

# ── Round 2: MTP ───────────────────────────────────────────────────

echo ""
echo "[Round 2] MTP oracle replay"

export SGLANG_ORACLE_REPLAY="$OUTPUT_DIR/trajectory.json"
python3 -m hybrid_spec_decoding.sglang_integration.install_hook

echo "Starting MTP server..."
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 4 \
    --speculative-algorithm EAGLE \
    --enable-multi-layer-eagle \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.8 \
    --disable-cuda-graph \
    --host 0.0.0.0 --port $PORT &
SERVER_PID=$!

echo "Waiting for MTP server..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "MTP server died!" && exit 1
    fi
    sleep 5
done

echo "Running BFCL agent (Round 2 replay)..."
python3 -m hybrid_spec_decoding.analysis.bfcl_agent \
    --url "http://localhost:$PORT/v1" \
    --model "$MODEL" \
    --input-file data/bfcl_multi_turn/dataset.jsonl \
    --output-file "$OUTPUT_DIR/agent_results_mtp.json" \
    --num-requests "$NUM_REQUESTS" \
    --max-iterations "$MAX_ITERATIONS" \
    --replay "$OUTPUT_DIR/agent_results_eagle3.json"

echo "Stopping MTP server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
unset SGLANG_ORACLE_REPLAY

# Verify Round 2
python3 -c "
import json
with open('$OUTPUT_DIR/agent_results_mtp.json') as f:
    d = json.load(f)
q = d['questions'][0]
entries = sum(len(s.get('spec_decode',{}).get('oracle_vanilla_entries',[])) for s in q['agent_metrics']['steps'])
has_tree = any(
    e.get('eagle3_tree') is not None
    for s in q['agent_metrics']['steps']
    for e in s.get('spec_decode',{}).get('oracle_vanilla_entries',[])
)
proposer = None
for s in q['agent_metrics']['steps']:
    for e in s.get('spec_decode',{}).get('oracle_vanilla_entries',[]):
        if e.get('proposer'):
            proposer = e['proposer']
            break
    if proposer: break
print(f'Round 2: {entries} entries, has_tree={has_tree}, proposer={proposer}')
"

# ── Offline: Union trie + p_t + Simulation ─────────────────────────

echo ""
echo "[Offline] Collecting union tries (EAGLE3 + MTP + Suffix)..."
python3 -m hybrid_spec_decoding.analysis.collect_union_trie \
    --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
    --mtp-agent-results "$OUTPUT_DIR/agent_results_mtp.json" \
    --output "$OUTPUT_DIR/union_trie_data.jsonl"

echo ""
echo "[Offline] Collecting target model p_t..."
python3 -m hybrid_spec_decoding.analysis.collect_target_probs \
    --union-trie-data "$OUTPUT_DIR/union_trie_data.jsonl" \
    --output "$OUTPUT_DIR/union_trie_data_with_pt.jsonl" \
    --model "$MODEL"

echo ""
echo "[Offline] Running tree oracle simulation..."
python3 -m hybrid_spec_decoding.analysis.run_tree_oracle_sim \
    --union-trie-data "$OUTPUT_DIR/union_trie_data_with_pt.jsonl" \
    --budgets "1,2,3,4,5,8,16" \
    --p-t-key p_t \
    --output "$OUTPUT_DIR/tree_oracle_sim.json" \
    --print-summary

# Also run with oracle p_t
python3 -m hybrid_spec_decoding.analysis.run_tree_oracle_sim \
    --union-trie-data "$OUTPUT_DIR/union_trie_data_with_pt.jsonl" \
    --budgets "1,2,3,4,5,8,16" \
    --p-t-key p_t_oracle \
    --output "$OUTPUT_DIR/tree_oracle_sim_oracle.json" \
    --print-summary

echo ""
echo "=============================================="
echo " PIPELINE TEST COMPLETE"
echo "=============================================="
echo " Output files:"
for f in "$OUTPUT_DIR"/*; do
    echo "   $(basename $f): $(wc -c < $f) bytes"
done
echo "=============================================="
