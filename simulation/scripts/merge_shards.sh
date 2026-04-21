#!/usr/bin/env bash
# Merge partial pipeline results and run Stage 6 (oracle simulation).
#
# Usage:
#   bash simulation/scripts/merge_shards.sh simulation/results/glm4_flash/bfcl_v4
#
# Expects partial-run directories like:
#   simulation/results/glm4_flash/bfcl_v4_req0-50/
#   simulation/results/glm4_flash/bfcl_v4_req50-100/
#
# Produces merged output in:
#   simulation/results/glm4_flash/bfcl_v4/

set -euo pipefail

OUTPUT_DIR=${1:?Usage: $0 <output_dir>  (e.g. simulation/results/glm4_flash/bfcl_v4)}

# Find partial-run directories
SHARD_DIRS=$(ls -d "${OUTPUT_DIR}"_req*/ 2>/dev/null | sort)
if [ -z "$SHARD_DIRS" ]; then
  echo "ERROR: No partial-run directories found matching ${OUTPUT_DIR}_req*/"
  exit 1
fi

NUM_SHARDS=$(echo "$SHARD_DIRS" | wc -l)
echo "Found $NUM_SHARDS partial runs:"
echo "$SHARD_DIRS" | sed 's/^/  /'
echo ""

mkdir -p "$OUTPUT_DIR"

# --- Merge union_trie_data_with_pt.jsonl ---
MERGED_PT="$OUTPUT_DIR/union_trie_data_with_pt.jsonl"
echo "Merging union_trie_data_with_pt.jsonl..."
> "$MERGED_PT"
for d in $SHARD_DIRS; do
  f="$d/union_trie_data_with_pt.jsonl"
  if [ -f "$f" ]; then
    cat "$f" >> "$MERGED_PT"
  else
    echo "  WARN: $f not found, skipping"
  fi
done
echo "  $(wc -l < "$MERGED_PT") total records"

# --- Merge agent_results_eagle3.json (questions arrays) ---
echo "Merging agent_results_eagle3.json..."
python3 -c "
import json, sys, glob

files = sorted(glob.glob('${OUTPUT_DIR}_req*/agent_results_eagle3.json'))
if not files:
    print('WARN: No agent_results_eagle3.json found', file=sys.stderr)
    sys.exit(0)

merged = json.load(open(files[0]))
for f in files[1:]:
    data = json.load(open(f))
    merged['questions'].extend(data.get('questions', []))
merged['metadata']['num_requests'] = len(merged['questions'])

with open('$OUTPUT_DIR/agent_results_eagle3.json', 'w') as f:
    json.dump(merged, f)
print(f'  {len(merged[\"questions\"])} questions from {len(files)} shards')
"

# --- Merge agent_results_mtp.json ---
echo "Merging agent_results_mtp.json..."
python3 -c "
import json, sys, glob

files = sorted(glob.glob('${OUTPUT_DIR}_req*/agent_results_mtp.json'))
if not files:
    print('WARN: No agent_results_mtp.json found', file=sys.stderr)
    sys.exit(0)

merged = json.load(open(files[0]))
for f in files[1:]:
    data = json.load(open(f))
    merged['questions'].extend(data.get('questions', []))
merged['metadata']['num_requests'] = len(merged['questions'])

with open('$OUTPUT_DIR/agent_results_mtp.json', 'w') as f:
    json.dump(merged, f)
print(f'  {len(merged[\"questions\"])} questions from {len(files)} shards')
"

# --- Run Stage 6: Oracle Simulation ---
echo ""
echo "=== Stage 6: Oracle Simulation (merged) ==="

LATENCY_FLAG=""
if [ -f "$OUTPUT_DIR/latency_config.json" ]; then
  LATENCY_FLAG="--latency-config $OUTPUT_DIR/latency_config.json"
fi

python3 -m simulation.evaluation.run_tree_oracle_sim \
  --union-trie-data "$MERGED_PT" \
  --budgets 1,2,4,8,16 \
  --p-t-key p_t \
  --output "$OUTPUT_DIR/tree_oracle_sim.json" \
  --print-summary \
  $LATENCY_FLAG

echo ""
echo "======================================"
echo "Merge complete!"
echo "Shards: $NUM_SHARDS"
echo "Results: $OUTPUT_DIR/"
echo "======================================"
