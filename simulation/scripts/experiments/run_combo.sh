#!/bin/bash
# COMBO: windowed MULTI-FEATURE online calibrator + conditional, served pinned to
# the same 14B eval trajectory as everything else (qwen3_14b_online) so MAT is
# directly comparable to raw / online / multifeat / oracle.
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
DIR=$B/qwen3_14b_online
PORT=30021; TAIL=64; MINS="${MINS:-20}"
WINDOWS="${WINDOWS:-256 1024}"; LABELS="${LABELS:-accept_rate target_p}"
for LABEL in $LABELS; do
  for W in $WINDOWS; do
    echo "##### COMBO label=$LABEL window=$W (multifeat+conditional, continuous) #####"
    python3 simulation/scripts/measure_chain_hybrid.py --preset qwen3_14b $COMMON \
      --train-n-tasks 30 --n-tasks 20 --port "$PORT" --tail-max-tokens "$TAIL" \
      --replay-existing --skip-calib-fit --pin-trajectory \
      --online-window "$W" --online-min-samples "$MINS" --online-scope continuous \
      --online-label "$LABEL" --online-conditional \
      --arms select1_online_multifeat --output "$DIR/run_combo_${LABEL}_w${W}.json" $EXTRA
    echo "RC_${LABEL}_w${W}=$?"
    for k in timing decisions online_pairs; do
      f="$DIR/${k}_select1_online_multifeat.jsonl"
      [ -f "$f" ] && mv -f "$f" "$DIR/${k}_combo_${LABEL}_w${W}.jsonl"
    done
  done
done
echo "COMBO_DONE"
