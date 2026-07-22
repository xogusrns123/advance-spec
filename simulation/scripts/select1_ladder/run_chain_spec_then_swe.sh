#!/bin/bash
# Chain: wait for the running SpecBench capture to finish, then (once GPU0 is genuinely free
# — won't kill any foreign process) launch the SWE-Bench capture. Runs inside docker so it
# survives host-session restarts. GPU0-only safety: only launches when GPU0 mem < 15GB.
set -uo pipefail
echo "CHAIN: $(date -u) waiting for SpecBench measure to finish..."
until ! pgrep -f "measure_chain_hybrid.py.*--workload specbench" >/dev/null 2>&1; do sleep 30; done
echo "CHAIN: $(date -u) SpecBench measure gone. waiting for GPU0 free (<15GB, no foreign proc)..."
for i in $(seq 1 240); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  [ "${m:-99999}" -lt 15000 ] && break
  echo "CHAIN: GPU0 busy (${m} MiB) $(date -u); waiting"; sleep 30
done
sleep 20
echo "CHAIN: $(date -u) launching SWE-Bench capture"
PRESET="${PRESET:-qwen35_27b_mtp}" N="${N:-10}" bash /workspace/simulation/scripts/select1_ladder/run_swebench_capture.sh
echo "CHAIN: $(date -u) SWE-Bench capture returned rc, chain done"
