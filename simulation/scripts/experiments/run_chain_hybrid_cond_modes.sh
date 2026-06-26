#!/bin/bash
# Measure the NEW cond-trained calibration MODES (global / depth_feature) as extra
# real-serving arms, REUSING the existing eval trajectory + pin — identical fairness
# to STEP G2 of run_chain_hybrid_perdepth.sh (--replay-existing reads
# $DIR/agent_results_record.json; --pin-trajectory reads $DIR/gt_tokens.jsonl).
# The per-position calib maps must already be fit into $DIR/calib_<MODE>/ by
#   fit_chain_hybrid_calib_perpos.py ... --accept-conditioned --mode <m> --out-dir $DIR/calib_<MODE>
# Outputs are renamed timing/decisions_select1_calib_<method>_<MODE>.jsonl so they
# sit alongside the existing *_cond-trained (per_depth) arm for direct comparison.
#
# Usage: run_chain_hybrid_cond_modes.sh <PRESET> <PORT> <DIR> <TAIL> <NDT> <MODE...>
#   14B target_p: run_chain_hybrid_cond_modes.sh qwen3_14b 30021 qwen3_14b_tp 64 0 cond-global cond-depthfeat
set -uo pipefail
PRESET="${1:?preset}"; PORT="${2:?port}"; DIR="${3:?dir}"; TAIL="${4:?tail}"
NDT="${5:?ndt(0=default)}"; shift 5
MODES="$*"
[ -n "$MODES" ] || { echo "ABORT: give at least one MODE (e.g. cond-global)"; exit 2; }
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
: "${CUDA_HOME:?no CUDA toolkit with nvcc found under /usr/local}"
export PATH="$CUDA_HOME/bin:$PATH"
echo "using CUDA_HOME=$CUDA_HOME"
COMMON="--workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1"
EXTRA="--extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch"
BASE="simulation/results/chain_hybrid_perdepth"
METHODS="histogram isotonic logistic beta"
NDTARG=""; [ "$NDT" != "0" ] && NDTARG="--spec-num-draft-tokens $NDT"
TRAIN_N="${TRAIN_N:-30}"; EVAL_N="${EVAL_N:-20}"
D="$BASE/$DIR"
[ -s "$D/agent_results_record.json" ] || { echo "ABORT: $D/agent_results_record.json missing (need a prior record for --replay-existing)"; exit 1; }
[ -s "$D/gt_tokens.jsonl" ] || { echo "ABORT: $D/gt_tokens.jsonl missing (need it for --pin-trajectory)"; exit 1; }

for MODE in $MODES; do
  echo "=== MODE=$MODE: replay-existing + pin, 4 calib arms ($(date +%H:%M:%S)) ==="
  [ -s "$D/calib_$MODE/calib_pp_isotonic.json" ] || { echo "ABORT: $D/calib_$MODE maps missing"; exit 1; }
  cp "$D"/calib_$MODE/calib_pp_*.json "$D"/
  python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
    --train-n-tasks "$TRAIN_N" --n-tasks "$EVAL_N" --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
    --replay-existing --skip-calib-fit --pin-trajectory \
    --arms select1_calib_histogram,select1_calib_isotonic,select1_calib_logistic,select1_calib_beta \
    --output "$D/run_$MODE.json" $EXTRA
  echo "MODE_${MODE}_RC=$?"
  for m in $METHODS; do
    for k in timing decisions; do
      [ -f "$D/${k}_select1_calib_${m}.jsonl" ] && \
        mv -f "$D/${k}_select1_calib_${m}.jsonl" "$D/${k}_select1_calib_${m}_${MODE}.jsonl"
    done
  done
done
echo "COND_MODES_DONE $DIR : $MODES"
