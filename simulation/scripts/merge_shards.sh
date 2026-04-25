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

merge_jsonl() {
  local name=$1
  local out="$OUTPUT_DIR/$name"
  echo "Merging $name..."
  > "$out"
  local found=0
  for d in $SHARD_DIRS; do
    local f="$d/$name"
    if [ -f "$f" ]; then
      cat "$f" >> "$out"
      found=$((found + 1))
    fi
  done
  if [ "$found" -eq 0 ]; then
    rm -f "$out"
    echo "  SKIP: no shard had $name"
    return 1
  fi
  echo "  $(wc -l < "$out") total records from $found shards"
  return 0
}

# --- Merge Stage 2 per-step draft artifacts ---
merge_jsonl draft_model_drafts.jsonl || true

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

# --- Run Stage 3: Oracle Simulation ---
echo ""
echo "=== Stage 3: Oracle Simulation (merged) ==="

LATENCY_FLAG=""
if [ -f "$OUTPUT_DIR/latency_config.json" ]; then
  LATENCY_FLAG="--latency-config $OUTPUT_DIR/latency_config.json"
fi

DM_FLAG=""
if [ -f "$OUTPUT_DIR/draft_model_drafts.jsonl" ]; then
  DM_FLAG="--draft-model-drafts $OUTPUT_DIR/draft_model_drafts.jsonl"
fi

python3 -m simulation.evaluation.run_tree_oracle_sim \
  --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
  $DM_FLAG \
  --budgets "${SIM_BUDGETS:-1,2,4,8,16,32,64,128,256,512}" \
  --output "$OUTPUT_DIR/tree_oracle_sim.json" \
  --print-summary \
  $LATENCY_FLAG

echo ""
echo "======================================"
echo "Merge complete!"
echo "Shards: $NUM_SHARDS"
echo "Results: $OUTPUT_DIR/"
echo "======================================"
