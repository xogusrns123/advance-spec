#!/bin/bash
# Online windowed calibration sweep (direction 1: break the per-depth ceiling
# with serving-time sliding-window calibration, no training). Pinned to the
# EXISTING 14B eval trajectory (from qwen3_14b_ar) so online arms are directly
# comparable to raw / oracle / offline-calib on the identical token path.
#
# Sweep: label {target_p, accept_rate} x window {256,1024} x method {4},
#        conditional ON (ingest depth only if accept_len>=depth), scope continuous.
# Each (label,window) = one measure invocation (4 online arms); calib/decisions/
# pairs are renamed *_<label>_w<window> so all 16 arms coexist for plotting.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }
done
: "${CUDA_HOME:?no CUDA toolkit with nvcc}"
export PATH="$CUDA_HOME/bin:$PATH"
COMMON="--workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1"
EXTRA="--extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch"
B=simulation/results/chain_hybrid_perdepth
PRESET="${PRESET:-qwen3_14b}"
SRC="${SRC:-$B/qwen3_14b_ar}"
DIR="${DIR:-$B/qwen3_14b_online}"
PORT="${PORT:-30021}"; TAIL="${TAIL:-64}"; MINS="${MINS:-20}"
NDT="${NDT:-0}"; NDTARG=""; [ "$NDT" != "0" ] && NDTARG="--spec-num-draft-tokens $NDT"
TRAIN_N="${TRAIN_N:-30}"; EVAL_N="${EVAL_N:-20}"
WINDOWS="${WINDOWS:-256 1024}"; LABELS="${LABELS:-target_p accept_rate}"
mkdir -p "$DIR"

# stage the eval trajectory + conversation + reference-arm timings (same path)
cp "$SRC"/gt_tokens.jsonl "$SRC"/agent_results_record.json "$DIR"/ || { echo "ABORT: missing src trajectory"; exit 1; }
for a in baseline select1 select1_oracle suffix; do
  [ -f "$SRC/timing_$a.jsonl" ]    && cp "$SRC/timing_$a.jsonl" "$DIR"/
  [ -f "$SRC/decisions_$a.jsonl" ] && cp "$SRC/decisions_$a.jsonl" "$DIR"/
done
cp "$SRC/run_all-trained.json" "$DIR/run_refs.json"

ARMS=select1_online_histogram,select1_online_isotonic,select1_online_logistic,select1_online_beta
for LABEL in $LABELS; do
  for W in $WINDOWS; do
    echo "##### ONLINE label=$LABEL window=$W (conditional, continuous) #####"
    python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
      --train-n-tasks "$TRAIN_N" --n-tasks "$EVAL_N" --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
      --replay-existing --skip-calib-fit --pin-trajectory \
      --online-window "$W" --online-min-samples "$MINS" --online-scope continuous \
      --online-label "$LABEL" --online-conditional \
      --arms "$ARMS" --output "$DIR/run_online_${LABEL}_w${W}.json" $EXTRA
    echo "RC_${LABEL}_w${W}=$?"
    for m in histogram isotonic logistic beta; do
      for k in timing decisions online_pairs; do
        f="$DIR/${k}_select1_online_${m}.jsonl"
        [ -f "$f" ] && mv -f "$f" "$DIR/${k}_online_${m}_${LABEL}_w${W}.jsonl"
      done
    done
  done
done
echo "ONLINE_SWEEP_DONE"
