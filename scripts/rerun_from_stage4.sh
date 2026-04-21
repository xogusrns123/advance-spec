#!/usr/bin/env bash
# Re-run Stage 4 with new suffix params, MERGE existing draft_model, then Stage 6.
# Draft model data doesn't depend on suffix, so we reuse it from the old
# union_trie_data_with_dm.jsonl (backup) instead of re-collecting on GPU.
#
# Usage: bash scripts/rerun_from_stage4.sh <output_dir> <model_preset>
set -euo pipefail

OUTPUT_DIR=${1:?}
MODEL_PRESET=${2:?}

case $MODEL_PRESET in
  qwen3_8b)
    MODEL="Qwen/Qwen3-8B"
    ;;
  *)
    echo "Unknown model preset: $MODEL_PRESET"; exit 1 ;;
esac

if [[ "$OUTPUT_DIR" == *specbench* ]]; then
  DATASET_FLAG="--dataset data/specbench/dataset.jsonl --model $MODEL"
else
  DATASET_FLAG="--model $MODEL"
fi

echo "====================================="
echo "Re-run Stage 4-6 (merge DM): $OUTPUT_DIR"
echo "====================================="

# Preserve old with_dm for draft_model merge
OLD_DM="$OUTPUT_DIR/union_trie_data_with_dm_OLD.jsonl"
if [ -f "$OUTPUT_DIR/union_trie_data_with_dm.jsonl" ] && [ ! -f "$OLD_DM" ]; then
  mv "$OUTPUT_DIR/union_trie_data_with_dm.jsonl" "$OLD_DM"
fi

rm -f "$OUTPUT_DIR/union_trie_data.jsonl"
rm -f "$OUTPUT_DIR/tree_oracle_sim.json"

# Ensure latency_config
if [ ! -f "$OUTPUT_DIR/latency_config.json" ] && [ -f "results/${MODEL_PRESET}/latency_config.json" ]; then
  cp "results/${MODEL_PRESET}/latency_config.json" "$OUTPUT_DIR/latency_config.json"
fi

# Stage 4: Collect Union Trie (with new suffix params)
echo "=== Stage 4 ==="
python3 -m simulation.pipeline.collect_union_trie \
  --agent-results "$OUTPUT_DIR/agent_results_eagle3.json" \
  --output "$OUTPUT_DIR/union_trie_data.jsonl" \
  $DATASET_FLAG

# Stage 4b-merge: copy draft_model from OLD into new records
echo "=== Stage 4b (merge from backup) ==="
python3 -c "
import json
import sys

old_dm = '$OLD_DM'
new_data = '$OUTPUT_DIR/union_trie_data.jsonl'
out = '$OUTPUT_DIR/union_trie_data_with_dm.jsonl'

# Build (req_id, call_idx, step_idx) -> draft_model dict from old
dm_by_key = {}
with open(old_dm) as f:
    for line in f:
        r = json.loads(line)
        key = (r.get('request_id',''), r.get('call_idx', 0), r.get('step_idx', 0))
        dm = r.get('per_proposer', {}).get('draft_model')
        if dm:
            dm_by_key[key] = dm

merged = 0
missing = 0
with open(new_data) as fin, open(out, 'w') as fout:
    for line in fin:
        r = json.loads(line)
        key = (r.get('request_id',''), r.get('call_idx', 0), r.get('step_idx', 0))
        dm = dm_by_key.get(key)
        if dm:
            r.setdefault('per_proposer', {})['draft_model'] = dm
            # Rebuild union_trie to include draft_model tokens
            merged += 1
        else:
            missing += 1
        fout.write(json.dumps(r) + '\n')

print(f'Merged draft_model into {merged} records, missing in {missing}', file=sys.stderr)
"

# Stage 6: Oracle sim
echo "=== Stage 6 ==="
python3 -m simulation.evaluation.run_tree_oracle_sim \
  --union-trie-data "$OUTPUT_DIR/union_trie_data_with_dm.jsonl" \
  --budgets 1,2,4,8,16,32,64,128,256,512 \
  --p-t-key p_t \
  --output "$OUTPUT_DIR/tree_oracle_sim.json" \
  --latency-config "$OUTPUT_DIR/latency_config.json" \
  --print-summary

echo "DONE: $OUTPUT_DIR"
