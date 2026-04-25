#!/usr/bin/env bash
# Rerun Stage 3 (oracle simulation) on an existing pipeline output dir,
# sharding the budget list into parallel processes so the large budgets
# (which dominate wall time) can run alongside the small-budget shard.
# Merges partial outputs into a single tree_oracle_sim.json.
#
# Usage: bash simulation/scripts/rerun_stage6_sharded.sh <output_dir>
set -euo pipefail
D=${1:?Usage: $0 <output_dir>}

AGENT_RESULTS="$D/agent_results_eagle3.json"
if [ ! -f "$AGENT_RESULTS" ]; then
  echo "ERROR: $AGENT_RESULTS not found"
  exit 1
fi

DM_FLAG=""
if [ -f "$D/draft_model_drafts.jsonl" ]; then
  DM_FLAG="--draft-model-drafts $D/draft_model_drafts.jsonl"
fi

LATENCY_FLAG=""
if [ -f "$D/latency_config.json" ]; then
  LATENCY_FLAG="--latency-config $D/latency_config.json"
fi

# Budget shards: small group together, large each get own shard
BUDGETS_A="1,2,4,8,16,32,64"
BUDGETS_B="128"
BUDGETS_C="256"
BUDGETS_D="512"

SAFE_D=${D//\//_}

PIDS=()
for SH in A B C D; do
  case $SH in
    A) B=$BUDGETS_A ;;
    B) B=$BUDGETS_B ;;
    C) B=$BUDGETS_C ;;
    D) B=$BUDGETS_D ;;
  esac
  (
    python3 -m simulation.evaluation.run_tree_oracle_sim \
      --agent-results "$AGENT_RESULTS" \
      $DM_FLAG \
      --budgets "$B" \
      --output "$D/tree_oracle_sim_part_${SH}.json" \
      $LATENCY_FLAG \
      --print-summary > "/tmp/stage3_${SAFE_D}_${SH}.log" 2>&1
  ) &
  PIDS+=($!)
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait $PID; then FAILED=$((FAILED + 1)); fi
done
if [ $FAILED -gt 0 ]; then
  echo "WARN: $FAILED shard(s) failed for $D"
fi

# Merge partials
python3 -c "
import json, os
parts = []
for i in ['A', 'B', 'C', 'D']:
    p = '$D/tree_oracle_sim_part_' + i + '.json'
    if os.path.exists(p):
        with open(p) as f:
            parts.append(json.load(f))
if not parts:
    raise SystemExit('No partial outputs to merge')
merged = parts[0]
all_budgets = []
for p in parts:
    all_budgets.extend(p['latency']['budget_sweep'])
all_budgets.sort(key=lambda x: x['budget'])
merged['latency']['budget_sweep'] = all_budgets
merged['metadata']['budgets'] = sorted(set(x['budget'] for x in all_budgets))
with open('$D/tree_oracle_sim.json', 'w') as f:
    json.dump(merged, f)
print(f'Merged {len(all_budgets)} budget entries')
"

rm -f "$D/tree_oracle_sim_part_"*.json
echo "DONE: $D"
