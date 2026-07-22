#!/bin/bash
# Serve realized calib(logistic) + bayes(GBM) select-1 arms on the SpecBench eval (replay the
# existing record). Maps were fit in-sample on the eval oracle log. Completes the
# single/raw/calib/bayes/oracle realized ladder for SpecBench.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }
done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
SD=/workspace/simulation/results/chain_hybrid_perdepth/specbench_qwen35_27b_mtp_2way
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen35_27b_mtp --workload specbench \
  --steps 16 --tp-size 1 --n-tasks 120 --tail-max-tokens 0 --replay-existing --skip-calib-fit \
  --arms select1_calib_logistic,select1_bayes --port 31082 --mem-fraction-static 0.65 \
  --output "$SD/run_calib.json"
echo "SPECBENCH_CALIB_RC=$?"
