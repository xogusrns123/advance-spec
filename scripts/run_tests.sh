#!/usr/bin/env bash
# Run all tests (offline-safe, no GPU / server required).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Running tests ==="
python -m pytest tests/ -v --tb=short "$@"
