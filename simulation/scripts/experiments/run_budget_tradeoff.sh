#!/bin/bash
# Node-budget tradeoff: per (model, method) real-serving budget sweep, then the
# dual-axis (ms/tok vs tau) figure. REAL SGLang serving on Blackwell GPU0.
#
# Run INSIDE the sglang-bench container as root (figures dir is root-owned):
#   docker exec -u root sglang-bench bash -lc \
#     ". /opt/venv/bin/activate && cd /workspace && \
#      simulation/scripts/experiments/run_budget_tradeoff.sh qwen3_8b"
#
# MODEL_KEY in {qwen3_8b, qwen35_27b}. ~7 budgets x 4 cells = ~28 server boots
# per model — run one model at a time. Override BUDGETS/PORT/MAX_TOKENS via env.
set -uo pipefail
MODEL="${1:?model key: qwen3_8b | qwen35_27b}"
cd /workspace

BUDGETS="${BUDGETS:-16,32,64,128,256,512,1024}"
PORT="${PORT:-30066}"
MAX_TOKENS="${MAX_TOKENS:-128}"
OUTROOT="simulation/results/budget_tradeoff"

# 27b_small (Qwen3.5-0.8B draft) has no valid draft checkpoint — dropped.
# 27B uses the cell default mem (0.70): higher boot-OOMs (model+draft+workspace
# need ~29GB free), so 27B tree methods are memory-capped (~B<=96) and the
# harness early-stops higher budgets. Suffix (model-free) reaches 1024.
case "$MODEL" in
  qwen3_8b)   CELLS="8b_eagle3,8b_small,8b_dflash,8b_suffix"; MEM="${MEM:-}" ;;
  qwen35_27b) CELLS="27b_mtp,27b_dflash,27b_suffix";          MEM="${MEM:-}" ;;
  *) echo "unknown model key '$MODEL' (use qwen3_8b | qwen35_27b)"; exit 1 ;;
esac
MEMARG=""; [ -n "$MEM" ] && MEMARG="--mem-fraction $MEM"

echo "=== STEP 1: budget sweep | model=$MODEL cells=$CELLS budgets=$BUDGETS mem=$MEM ==="
python3 simulation/scripts/experiments/measure_budget_sweep.py \
  --cells "$CELLS" --budgets "$BUDGETS" --port "$PORT" \
  --max-tokens "$MAX_TOKENS" $MEMARG --out "$OUTROOT"
echo "SWEEP_RC=$?"

echo "=== STEP 2: plot ==="
python3 simulation/scripts/plot_budget_tradeoff.py \
  --glob "$OUTROOT/${MODEL}/*.json"

echo "BUDGET_TRADEOFF_${MODEL}_DONE"
