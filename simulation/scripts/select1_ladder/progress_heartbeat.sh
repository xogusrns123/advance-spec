#!/bin/bash
# Periodic progress heartbeat for the SpecBench->SWE-Bench pipeline. Emits one compact
# line every ~5 min: current phase + latest per-task tqdm + GPU0. Exits when the pipeline
# is done (no measure process AND the SWE-Bench run.json exists). Runs inside docker.
SD=/workspace/simulation/results/chain_hybrid_perdepth/specbench_qwen35_27b_mtp_2way
WD=/workspace/simulation/results/chain_hybrid_perdepth/swebench_qwen35_27b_mtp_2way
for i in $(seq 1 120); do
  phase="boot/replay"
  pgrep -f "workload specbench" >/dev/null 2>&1 && phase="SpecBench"
  pgrep -f "workload swebench" >/dev/null 2>&1 && phase="SWE-Bench"
  latest=$(ls -t $SD/*_agent.log $WD/*_agent.log 2>/dev/null | head -1)
  prog=$(grep -haoE "[0-9]+/[0-9]+ \[[0-9][0-9:]*<[^]]*\]" "$latest" 2>/dev/null | tail -1)
  arm=$(ls -t $SD $WD 2>/dev/null | grep -m1 -aoE "" ; ls -t $SD/server_*.log $WD/server_*.log 2>/dev/null | head -1 | sed 's#.*/server_##;s#\.log##')
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 0 2>/dev/null | tr '\n' ' ')
  err=$(grep -haE "Traceback|CUDA error|out of memory|Failed to setup repo" $SD/*_agent.log $WD/*_agent.log 2>/dev/null | tail -1)
  echo "$(date -u +%H:%MZ) ${phase}[arm=${arm:-?}] | ${prog:-booting/replay} | GPU0 ${gpu}${err:+ | ERR: $err}"
  if ! pgrep -f "measure_chain_hybrid" >/dev/null 2>&1 && [ -f "$WD/run.json" ]; then echo "PIPELINE DONE (both captures finished)"; break; fi
  sleep 300
done
