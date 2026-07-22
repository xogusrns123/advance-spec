#!/bin/bash
# EAGLE distributional-confidence capture (the one untested selection signal).
#
# Re-runs ONLY the record+oracle arms for the 14B eval slice with
# --log-eagle-dist, so the oracle decision log carries, per depth:
#   eagle_entropy, eagle_top2_margin, p_eagle_eagle, p_eagle_suffix
# ALONGSIDE the counterfactual labels it already has (gt_token, oracle_hit for
# BOTH proposers). One arm, one GPU pass — no calib fit, no pin, no replay.
#
# Then run (host, no GPU):
#   python3 simulation/scripts/select1_ladder/eagle_dist_probe.py --dir $OUTDIR
# which asks: does adding eagle-dist raise the joint-Bayes decisive-selacc
# ceiling above 0.790 (14B) / 0.907 (27B), and do the new features have
# AUC>0.5 in the confident-inversion region where all logged features are dead?
#
# RUN INSIDE sglang-bench (has arctic_inference + sglang), as root:
#   docker exec sglang-bench bash /workspace/simulation/scripts/experiments/run_eagle_dist_capture.sh
# Self-gates on GPU0 being free (the user has another experiment queued).
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
OUTDIR="${OUTDIR:-simulation/results/chain_hybrid_perdepth/qwen3_14b_edist}"
TRAIN_N="${TRAIN_N:-30}"   # eval slice offset (matches qwen3_14b_ar)
EVAL_N="${EVAL_N:-20}"
TAIL="${TAIL:-64}"
mkdir -p "$OUTDIR"

echo "=== waiting for GPU0 (<15GB) — user has another experiment queued ==="
while true; do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  echo "  GPU0 used=${U} MiB $(date +%H:%M:%S)"; [ "${U:-99999}" -lt 15000 ] && break; sleep 120
done
sleep 20

echo "=== eagle-dist capture: record+select1_oracle, ${EVAL_N} eval tasks (offset ${TRAIN_N}) ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" \
  --workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1 \
  --train-n-tasks "$TRAIN_N" --n-tasks "$EVAL_N" --port "$PORT" --tail-max-tokens "$TAIL" \
  --arms record,select1_oracle --log-eagle-dist \
  --output "$OUTDIR/run.json" \
  --extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch
echo "CAPTURE_RC=$?"
echo "oracle log -> $OUTDIR/decisions_select1_oracle.jsonl"
echo "next (host): python3 simulation/scripts/select1_ladder/eagle_dist_probe.py --dir $OUTDIR"
echo "EAGLE_DIST_CAPTURE_DONE"
