#!/bin/bash
# Direction 2: multi-feature per-proposer calibration, served pinned to the same
# 14B eval trajectory as the online sweep + refs (qwen3_14b_online), so MAT is
# directly comparable. Reuses the frozen maps fit by
# fit_chain_hybrid_calib_multifeat.py (qwen3_14b_multifeat/).
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }
done
: "${CUDA_HOME:?no nvcc}"; export PATH="$CUDA_HOME/bin:$PATH"
COMMON="--workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1"
EXTRA="--extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch"
B=simulation/results/chain_hybrid_perdepth
PRESET="${PRESET:-qwen3_14b}"
DIR="${DIR:-$B/qwen3_14b_online}"; MF="${MF:-$B/qwen3_14b_multifeat}"
PORT="${PORT:-30021}"; TAIL="${TAIL:-64}"
NDT="${NDT:-0}"; NDTARG=""; [ "$NDT" != "0" ] && NDTARG="--spec-num-draft-tokens $NDT"
TRAIN_N="${TRAIN_N:-30}"; EVAL_N="${EVAL_N:-20}"
LABELS="${LABELS:-accept_rate target_p}"
for LABEL in $LABELS; do
  echo "##### MULTIFEAT label=$LABEL #####"
  python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
    --train-n-tasks "$TRAIN_N" --n-tasks "$EVAL_N" --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
    --replay-existing --skip-calib-fit --pin-trajectory \
    --multifeat-map "$MF/multifeat_${LABEL}.json" \
    --arms select1_multifeat --output "$DIR/run_multifeat_${LABEL}.json" $EXTRA
  echo "RC_mf_${LABEL}=$?"
  for k in timing decisions; do
    f="$DIR/${k}_select1_multifeat.jsonl"
    [ -f "$f" ] && mv -f "$f" "$DIR/${k}_multifeat_${LABEL}.jsonl"
  done
done
echo "MULTIFEAT_DONE"
