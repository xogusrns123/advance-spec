#!/usr/bin/env bash
# Run all tests (offline-safe, no GPU / server required).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Running core tests ==="
python -m pytest tests/ -v --tb=short "$@"

echo ""
echo "=== Running simulation tests ==="
python -m pytest simulation/tests/ -v --tb=short "$@"
