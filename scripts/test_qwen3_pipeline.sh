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

echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# ── Offline: Union trie + Simulation ───────────────────────────────

echo ""
echo "[Offline] Full pipeline"
python3 -c "
import json, time, os
from collections import Counter

OUTPUT_DIR = '$OUTPUT_DIR'

# Load data
with open(f'{OUTPUT_DIR}/agent_results_eagle3.json') as f:
    data = json.load(f)

from hybrid_spec_decoding.analysis.run_oracle_sim import extract_requests
from hybrid_spec_decoding.analysis.collect_union_trie import collect_union_tries
from arctic_inference.suffix_decoding import SuffixDecodingCache
from hybrid_spec_decoding.analysis.collect_target_probs import enrich_with_ground_truth_p_t
from hybrid_spec_decoding.analysis.run_tree_oracle_sim import (
    evaluate_choose_one, evaluate_expected_utility,
    evaluate_choose_one_at_budget, print_summary,
)

# Extract
requests = extract_requests(data, set())
q = data['questions'][0]
entries = sum(len(s.get('spec_decode',{}).get('oracle_vanilla_entries',[])) for s in q['agent_metrics']['steps'])
n_tree = sum(1 for s in q['agent_metrics']['steps']
             for e in s.get('spec_decode',{}).get('oracle_vanilla_entries',[])
             if e.get('eagle3_tree'))
n_branch = sum(1 for s in q['agent_metrics']['steps']
               for e in s.get('spec_decode',{}).get('oracle_vanilla_entries',[])
               if e.get('eagle3_tree') and any(c > 1 for c in Counter(e['eagle3_tree']['parents']).values()))
print(f'Entries: {entries}, with tree: {n_tree}, with branching: {n_branch}')

# Union trie
cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
records = collect_union_tries(requests, cache)
total_nodes = sum(len(r['union_trie']['token_ids']) for r in records)
proposer_counts = {}
for r in records:
    for name in r['per_proposer']:
        proposer_counts[name] = proposer_counts.get(name, 0) + 1
print(f'Steps: {len(records)}, union nodes: {total_nodes} (avg {total_nodes/len(records):.1f})')
print(f'Proposers: {proposer_counts}')

# Oracle simulation
enrich_with_ground_truth_p_t(records)
choose_one = evaluate_choose_one(records)
budgets = [1, 2, 3, 4, 5, 8, 10, 15]
eu_oracle = evaluate_expected_utility(records, budgets, 'p_t_oracle')
c1b = evaluate_choose_one_at_budget(records, budgets)
print_summary(choose_one, eu_oracle, c1b, budgets, 'p_t_oracle')

# Save
with open(f'{OUTPUT_DIR}/union_trie_data.jsonl', 'w') as f:
    for rec in records:
        f.write(json.dumps(rec) + '\n')

output = {
    'metadata': {
        'model': '$MODEL',
        'draft_model': '$DRAFT_MODEL',
        'n_steps': len(records),
        'n_entries': entries,
        'n_branching': n_branch,
        'proposer_counts': proposer_counts,
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
