#!/usr/bin/env bash
# Full latency re-measurement for extension cost model.
#
# Re-measures everything (target_forward, eagle3_draft, suffix, draft_lm)
# with extended budgets to remove the linear-extrapolation regime above
# B=512. Compiles into simulation/config/latency/<preset>.json.
#
# Constraints:
#   * Each (K, B, S) eagle3 measurement boots a fresh SGLang server.
#     Cannot share a GPU/port with another running SGLang server.
#   * Budgets > capacity(K, S) = K + (S-1)*K^2 + 1 are auto-skipped.
#
# Run inside docker, on a GPU NOT used by _stage1_round_robin.sh:
#   docker exec -u root -d sglang-bench bash -c \
#     "cd /workspace && CUDA_VISIBLE_DEVICES=1 PORT=30001 PRESET=qwen3_14b \
#      bash _remeasure_latency.sh > /tmp/latency_14b.log 2>&1"
#
# Estimated wallclock: ~3-4h per preset (K=8,16 × S=2..10 × B=4..2048).
#
# Resume: each measure_*_cost.py supports incremental output (cached entries
# preserved). Re-running this script picks up where it left off.
set -euo pipefail

PRESET=${PRESET:-qwen3_14b}
case $PRESET in
  qwen3_8b)
    MODEL=${MODEL:-Qwen/Qwen3-8B}
    DRAFT_MODEL=${DRAFT_MODEL:-AngelSlim/Qwen3-8B_eagle3}
    DRAFT_LM=${DRAFT_LM:-Qwen/Qwen3-0.6B}
    ;;
  qwen3_14b)
    MODEL=${MODEL:-Qwen/Qwen3-14B}
    DRAFT_MODEL=${DRAFT_MODEL:-AngelSlim/Qwen3-14B_eagle3}
    DRAFT_LM=${DRAFT_LM:-Qwen/Qwen3-0.6B}
    ;;
  *)
    echo "Unknown PRESET=$PRESET (use qwen3_8b or qwen3_14b)" >&2
    exit 1
    ;;
esac

PORT=${PORT:-30001}
WORKLOADS=${WORKLOADS:-"specbench,bfcl_v4,swebench"}

# Sweep grid. Defaults add (1024, 2048) over the legacy table and steps=10
# so K=16 can reach B=2048 (capacity 2321).
BUDGETS=${BUDGETS:-"4,8,16,32,64,128,256,512,1024,2048"}
STEPS=${STEPS:-"2,4,6,8,10"}
TOPKS=${TOPKS:-"8,16"}

# Draft LM TPOT samples (extension's draft_lm proposer never used at
# inference time but kept in the config for completeness).
DRAFT_LM_NS=${DRAFT_LM_NS:-"1,2,3,5,8,16"}

# Suffix cost is computed off existing agent_results — points at the
# current results dir for this preset.
AGENT_RESULTS_DIR=${AGENT_RESULTS_DIR:-/workspace/simulation/results/${PRESET}}

OUTPUT_DIR=/workspace/results/latency/${PRESET}
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "Re-measuring latency for preset=$PRESET on port=$PORT"
echo "  budgets:   $BUDGETS"
echo "  steps:     $STEPS"
echo "  topks:     $TOPKS"
echo "  workloads: $WORKLOADS"
echo "  output:    $OUTPUT_DIR"
echo "================================================================"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export TORCHINDUCTOR_COMPILE_THREADS=1

if [ -f /workspace/.env ]; then set -a; source /workspace/.env; set +a; fi

# 1. EAGLE3 cost (target_forward_ms + eagle3_draft_ms)
echo ""
echo "=== [1/4] EAGLE3 cost: target_forward_ms + eagle3_draft_ms ==="
python3 simulation/scripts/measure_eagle3_cost.py \
  --model "$MODEL" --draft-model "$DRAFT_MODEL" \
  --workloads "$WORKLOADS" \
  --budgets "$BUDGETS" --steps "$STEPS" --topks "$TOPKS" \
  --port "$PORT" \
  --output "$OUTPUT_DIR/eagle3_cost.json"

# 2. Draft LM cost (per_token_ms by num_draft_tokens)
echo ""
echo "=== [2/4] Draft-LM TPOT: $DRAFT_LM ==="
python3 simulation/scripts/measure_draft_model_cost.py \
  --model "$DRAFT_LM" --workloads "$WORKLOADS" \
  --num-draft-tokens "$DRAFT_LM_NS" \
  --port "$PORT" \
  --output "$OUTPUT_DIR/draft_model_cost.json"

# 3. Suffix speculate cost (no server; reads agent_results)
echo ""
echo "=== [3/4] Suffix speculate cost ==="
if [ -d "$AGENT_RESULTS_DIR" ]; then
  python3 simulation/scripts/measure_suffix_cost.py \
    --workloads "$WORKLOADS" --model "$MODEL" \
    --agent-results-dir "$AGENT_RESULTS_DIR" \
    --output "$OUTPUT_DIR/suffix_cost.json"
else
  echo "WARN: agent results dir not found ($AGENT_RESULTS_DIR); skipping suffix cost." >&2
  echo "  -> reuse previous suffix_cost.json or rerun once Stage 1 has output." >&2
fi

# 4. Compile into latency_config.json
echo ""
echo "=== [4/4] Compile latency_config.json ==="
COMPILE_ARGS=( --eagle3-cost "$OUTPUT_DIR/eagle3_cost.json" )
[ -f "$OUTPUT_DIR/draft_model_cost.json" ] && COMPILE_ARGS+=( --draft-cost "$OUTPUT_DIR/draft_model_cost.json" )
[ -f "$OUTPUT_DIR/suffix_cost.json" ]      && COMPILE_ARGS+=( --suffix-cost "$OUTPUT_DIR/suffix_cost.json" )

python3 simulation/scripts/compile_latency_config.py \
  "${COMPILE_ARGS[@]}" \
  --canonical-steps 8 --canonical-topk 16 \
  --draft-ref-n 3 \
  --output "simulation/config/latency/${PRESET}.json"

echo ""
echo "================================================================"
echo "Latency config refreshed: simulation/config/latency/${PRESET}.json"
echo "================================================================"
