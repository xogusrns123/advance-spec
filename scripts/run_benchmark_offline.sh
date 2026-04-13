#!/usr/bin/env bash
# Run the unified benchmark with dummy data (no server required).
# Good for verifying the pipeline end-to-end.
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-results/offline_benchmark}"

echo "=== Offline benchmark (dummy prompts) ==="
python -m hybrid_spec_decoding.benchmarks.run_benchmark \
    --proposer mtp draft_model \
    --dummy \
    --max-tokens 64 \
    --max-tree-tokens 16 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Results ==="
ls -la "$OUTPUT_DIR"
echo ""
echo "Comparison (JSON):"
cat "$OUTPUT_DIR/comparison.json" 2>/dev/null || echo "(no comparison file)"
