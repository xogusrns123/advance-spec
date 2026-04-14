#!/usr/bin/env bash
# Qwen3-8B full pipeline test (EAGLE3 full tree + Suffix)
# Faster than GLM-4.7-Flash: 8B model, TP=1
#
# Usage: docker exec -w /workspace sglang-bench bash scripts/test_qwen3_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="Qwen/Qwen3-8B"
DRAFT_MODEL="Tengyunw/qwen3_8b_eagle3"
OUTPUT_DIR="results/qwen3_8b/pipeline_test"
PORT=30000
NUM_REQUESTS=1
MAX_ITERATIONS=1

mkdir -p "$OUTPUT_DIR"

cleanup() {
    echo "Cleaning up..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "=============================================="
echo " Qwen3-8B Pipeline Test"
echo " EAGLE3 full tree + Suffix, TP=1"
echo "=============================================="

# ── Round 1: EAGLE3 ────────────────────────────────────────────────

echo ""
echo "[Round 1] EAGLE3 oracle vanilla"

export SGLANG_ORACLE_VANILLA=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python3 -m hybrid_spec_decoding.sglang_integration.install_hook

echo "Starting EAGLE3 server (TP=1)..."
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 1 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.85 \
    --disable-cuda-graph \
    --host 0.0.0.0 --port $PORT &
SERVER_PID=$!

echo "Waiting for server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died!" && exit 1
    fi
    sleep 3
done

echo "Running BFCL agent..."
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

# ── Offline: Build union tries ─────────────────────────────────────

echo ""
echo "[Offline] Building union tries..."
python3 -c "
import json
from hybrid_spec_decoding.analysis.run_oracle_sim import extract_requests
from hybrid_spec_decoding.analysis.collect_union_trie import collect_union_tries
from arctic_inference.suffix_decoding import SuffixDecodingCache

with open('$OUTPUT_DIR/agent_results_eagle3.json') as f:
    data = json.load(f)
requests = extract_requests(data, set())
cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
records = collect_union_tries(requests, cache)
with open('$OUTPUT_DIR/union_trie_data.jsonl', 'w') as f:
    for rec in records:
        f.write(json.dumps(rec) + '\n')
print(f'Union tries: {len(records)} steps')
"

# ── Round 2: Verify union tries via SUFFIX server ──────────────────

echo ""
echo "[Round 2] Verifying union tries (SUFFIX server)"

# Extract trajectory for replay
python3 -m hybrid_spec_decoding.analysis.extract_trajectory \
    --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
    --output "$OUTPUT_DIR/trajectory.json"

export SGLANG_ORACLE_REPLAY="$OUTPUT_DIR/trajectory.json"
export SGLANG_ORACLE_VERIFY_TRIES="$OUTPUT_DIR/union_trie_data.jsonl"
> /tmp/sglang_oracle_verify_p_t.jsonl

echo "Starting SUFFIX server (verify mode)..."
python3 -m sglang.launch_server \
    --model-path "$MODEL" --tp-size 1 \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.85 \
    --disable-cuda-graph \
    --host 0.0.0.0 --port $PORT &
SERVER_PID=$!

echo "Waiting for server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died!" && exit 1
    fi
    sleep 3
done

echo "Running BFCL agent (replay + verify)..."
python3 -m hybrid_spec_decoding.analysis.bfcl_agent \
    --url "http://localhost:$PORT/v1" \
    --model "$MODEL" \
    --input-file data/bfcl_multi_turn/dataset.jsonl \
    --output-file "$OUTPUT_DIR/agent_results_verify.json" \
    --num-requests "$NUM_REQUESTS" \
    --max-iterations "$MAX_ITERATIONS" \
    --replay "$OUTPUT_DIR/agent_results_eagle3.json"

echo "Stopping SUFFIX server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
unset SGLANG_ORACLE_REPLAY SGLANG_ORACLE_VERIFY_TRIES

# ── Offline: Merge p_t + Simulation ────────────────────────────────

echo ""
echo "[Offline] Merge p_t + Simulation"
python3 -c "
import json, os

OUTPUT_DIR = '$OUTPUT_DIR'

# Load union trie records
records = []
with open(f'{OUTPUT_DIR}/union_trie_data.jsonl') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
print(f'Union trie records: {len(records)}')

# Load verification p_t
p_t_map = {}
verify_path = '/tmp/sglang_oracle_verify_p_t.jsonl'
if os.path.exists(verify_path):
    with open(verify_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                key = (entry['request_id'], entry['call_idx'], entry['step_idx'])
                p_t_map[key] = entry['p_t']
print(f'Verification p_t entries: {len(p_t_map)}')

# Merge p_t into records
n_merged = 0
for rec in records:
    key = (rec['request_id'], rec.get('call_idx', 0), rec.get('step_idx', 0))
    pt = p_t_map.get(key)
    if pt is not None:
        rec['p_t'] = pt
        n_merged += 1
print(f'Merged p_t: {n_merged}/{len(records)}')

# Add oracle p_t
from hybrid_spec_decoding.analysis.collect_target_probs import enrich_with_ground_truth_p_t
enrich_with_ground_truth_p_t(records)

# Simulation
from hybrid_spec_decoding.analysis.run_tree_oracle_sim import (
    evaluate_choose_one, evaluate_expected_utility,
    evaluate_choose_one_at_budget, print_summary,
)

choose_one = evaluate_choose_one(records)
budgets = [1, 2, 3, 4, 5, 8, 10, 15]
c1b = evaluate_choose_one_at_budget(records, budgets)

# Real p_t (from verification)
has_real_pt = any('p_t' in r for r in records)
if has_real_pt:
    eu_real = evaluate_expected_utility(records, budgets, 'p_t')
    print('\n=== Real p_t (verification logits) ===')
    print_summary(choose_one, eu_real, c1b, budgets, 'p_t')

# Oracle p_t
eu_oracle = evaluate_expected_utility(records, budgets, 'p_t_oracle')
print('\n=== Oracle p_t ===')
print_summary(choose_one, eu_oracle, c1b, budgets, 'p_t_oracle')

# Save enriched data
with open(f'{OUTPUT_DIR}/union_trie_data_with_pt.jsonl', 'w') as f:
    for rec in records:
        f.write(json.dumps(rec) + '\n')

output = {
    'metadata': {
        'model': '$MODEL',
        'draft_model': '$DRAFT_MODEL',
        'n_steps': len(records),
        'n_merged_p_t': n_merged,
    },
    'choose_one': {'aggregate': choose_one['aggregate']},
    'expected_utility_oracle': {
        'budget_sweep': [
            {'budget': B, 'eu_actual': eu_oracle[B]['avg_actual_acc'],
             'choose_one': c1b[B]['avg_acc'],
             'gap': eu_oracle[B]['avg_actual_acc'] - c1b[B]['avg_acc']}
            for B in budgets
        ],
    },
}
with open(f'{OUTPUT_DIR}/tree_oracle_sim.json', 'w') as f:
    json.dump(output, f, indent=2)

for fn in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(f'{OUTPUT_DIR}/{fn}')
    print(f'  {fn}: {size:,} bytes')
"

echo ""
echo "=============================================="
echo " DONE! Results in $OUTPUT_DIR"
echo "=============================================="
