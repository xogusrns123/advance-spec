#!/usr/bin/env bash
# Run the two hybrid baselines (offline, no server required).
#   1. Suffix Decoding + EAGLE-3 (simple merge)
#   2. RASD-style tree pruning + longest-prefix fusion
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-results/hybrid}"

echo "=== Hybrid baselines (dummy prompts) ==="
python -m hybrid_spec_decoding.benchmarks.run_hybrid \
    --baselines suffix_eagle_simple rasd_fusion \
    --dummy \
    --max-tokens 128 \
    --max-tree-tokens 64 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Output files ==="
ls -la "$OUTPUT_DIR"
echo ""
echo "Comparison (JSON):"
cat "$OUTPUT_DIR/comparison.json" 2>/dev/null || echo "(no comparison file)"
echo ""
echo "Comparison (CSV):"
cat "$OUTPUT_DIR/comparison.csv" 2>/dev/null || echo "(no comparison csv)"
