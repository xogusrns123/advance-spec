#!/usr/bin/env bash
# Run everything: tests, offline benchmarks, hybrid baselines.
# No GPU or server required -- safe on any machine.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=============================================="
echo " 1/3  Tests"
echo "=============================================="
bash scripts/run_tests.sh

echo ""
echo "=============================================="
echo " 2/3  Offline benchmark (MTP + DraftModel)"
echo "=============================================="
bash scripts/run_benchmark_offline.sh results/offline_benchmark

echo ""
echo "=============================================="
echo " 3/3  Hybrid baselines"
echo "=============================================="
bash scripts/run_hybrid_baselines.sh results/hybrid

echo ""
echo "=============================================="
echo " Done!  All results in results/"
echo "=============================================="
