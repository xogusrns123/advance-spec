#!/bin/bash
# select1_branch served validation ladder: single(record) / raw / BRANCH / oracle,
# all PINNED to one standalone trajectory, on the 14B bfcl eval slice.
#
# Validates the no-GPU replay prediction (simulation/results/select1_branch_sim/
# REPORT.md §0): branch(chain-side, band<0.3, m=2) served-est 1.597 vs raw 1.328 /
# disc_beta 1.385 / oracle 1.783, at +11 verify nodes/step (measured node price
# on this cell: 1.7us/node target fwd, 38.7us/node step total).
#
# The branch arm is NEW CODE (chain_hybrid_patch.py: _branch_append tree surgery;
# CPU tensor tests pass — see select1_branch_sim/test_branch_tensors.py). First
# served run: watch server_select1_branch.log for [branch-*] warn_once lines and
# compare decisions_select1_branch.jsonl branch_meta n_side_nodes/step (~11)
# against the replay prediction.
#
# RUN INSIDE sglang-bench, as root:
#   docker exec sglang-bench bash /workspace/simulation/scripts/experiments/run_branch_arm.sh
# Self-gates on GPU0 being free.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
: "${CUDA_HOME:?no CUDA toolkit with nvcc found under /usr/local}"
export PATH="$CUDA_HOME/bin:$PATH"

PRESET="${PRESET:-qwen3_14b}"
PORT="${PORT:-30021}"
OUTDIR="${OUTDIR:-simulation/results/chain_hybrid_perdepth/qwen3_14b_branch}"
TRAIN_N="${TRAIN_N:-30}"
EVAL_N="${EVAL_N:-20}"
TAIL="${TAIL:-64}"
BAND="${BAND:-0.3}"; M="${M:-2}"; BLEN="${BLEN:-16}"
mkdir -p "$OUTDIR"

echo "=== waiting for GPU0 (<15GB) ==="
while true; do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  echo "  GPU0 used=${U} MiB $(date +%H:%M:%S)"; [ "${U:-99999}" -lt 15000 ] && break; sleep 120
done
sleep 20

echo "=== branch ladder: record,select1,select1_branch,select1_oracle (pinned) ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" \
  --workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1 \
  --train-n-tasks "$TRAIN_N" --n-tasks "$EVAL_N" --port "$PORT" \
  --tail-max-tokens "$TAIL" --replay-all --skip-calib-fit --pin-trajectory \
  --branch-band "$BAND" --branch-m "$M" --branch-len "$BLEN" \
  --arms record,select1,select1_branch,select1_oracle \
  --output "$OUTDIR/run.json" \
  --extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch
echo "BRANCH_LADDER_RC=$?"
python3 - <<'EOF'
import json
j = json.load(open("simulation/results/chain_hybrid_perdepth/qwen3_14b_branch/run.json"))
for a, r in j.get("arms", {}).items():
    if isinstance(r, dict) and r.get("accept_length_mean") is not None:
        print(f"  {a:<18} MAT {r['accept_length_mean']:.3f}  "
              f"step {r.get('step_ms', float('nan')):.2f}ms  n={r.get('n_samples')}")
EOF
echo "BRANCH_ARM_DONE"
