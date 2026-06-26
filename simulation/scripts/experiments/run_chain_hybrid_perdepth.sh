#!/bin/bash
# Chain-hybrid per-depth calibration study: ALL-TRAINED vs COND-TRAINED
# (accept-conditioned) calib,
# under a chosen calibration OBJECTIVE, measured in REAL SERVING with every arm
# PINNED to one standalone-eagle3 trajectory (so depth curves are comparable
# despite greedy FP-tie flips, while each arm rebuilds its own draft chain and
# re-extracts its own eagle features — which offline log analysis cannot do).
#
# One (model, objective) cell:
#   STEP D    record+oracle on 30 TRAIN tasks (offset 0)        -> train-oracle log
#   STEP E.5  (target_p only) capture q_target per drafted token (teacher-forced)
#   STEP E    fit per-position calib maps TWICE: ALL-TRAINED (pooled) +
#             COND-TRAINED (--accept-conditioned)
#             -> $DIR/calib_all-trained , $DIR/calib_cond-trained
#   STEP G1   PASS1: replay-all (re-records the EVAL standalone trajectory) +
#             --pin-trajectory; arms = baseline,select1,4xcalib(ALL-TRAINED),oracle
#             -> run_all-trained.json ; rename calib outputs *_all-trained
#   STEP G2   PASS2: replay-existing (reuse PASS1 record) + --pin-trajectory;
#             arms = 4xcalib(COND-TRAINED) -> run_cond-trained.json ;
#             rename calib outputs *_cond-trained
#   then      suffix-only offline sim + plot_o4_objectives (MAT/survival/conditional)
#
# OBJECTIVE in {target_p, accept_rate}. target_p regresses onto q_target; accept_rate
# uses the binary token==gt label and skips STEP E.5 (no capture).
#
# Usage:  run_chain_hybrid_perdepth.sh <PRESET> <PORT> <DIR> <TAIL> <NDT> <OBJECTIVE>
#   14B target_p :  run_chain_hybrid_perdepth.sh qwen3_14b   30021 qwen3_14b_tp 64 0  target_p
#   14B accept_r :  REUSE_TRAIN=simulation/results/chain_hybrid_perdepth/qwen3_14b_tp_train \
#                   run_chain_hybrid_perdepth.sh qwen3_14b   30021 qwen3_14b_ar 64 0  accept_rate
#   27B target_p :  run_chain_hybrid_perdepth.sh qwen35_27b_mtp 30022 qwen35_27b_tp 32 49 target_p
#
# Optional env: REUSE_TRAIN=<train dir> reuses an existing STEP D train dir (skip
# STEP D; for the accept_rate cell that piggybacks on the target_p cell's trajectory).
set -uo pipefail
PRESET="${1:?preset}"; PORT="${2:?port}"; DIR="${3:?dir}"; TAIL="${4:?tail}"
NDT="${5:?ndt(0=default)}"; OBJ="${6:?objective target_p|accept_rate}"
case "$OBJ" in target_p|accept_rate) ;; *) echo "ABORT: OBJECTIVE must be target_p|accept_rate"; exit 2 ;; esac
cd /workspace
export CUDA_VISIBLE_DEVICES=0
# First CUDA toolkit that actually has nvcc (sglang JIT-compiles sm_120 at boot).
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
: "${CUDA_HOME:?no CUDA toolkit with nvcc found under /usr/local}"
export PATH="$CUDA_HOME/bin:$PATH"
echo "using CUDA_HOME=$CUDA_HOME ($($CUDA_HOME/bin/nvcc --version | tail -1))"
COMMON="--workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1"
EXTRA="--extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch"
BASE="simulation/results/chain_hybrid_perdepth"
METHODS="histogram isotonic logistic beta"
NDTARG=""; [ "$NDT" != "0" ] && NDTARG="--spec-num-draft-tokens $NDT"
# Task counts / fit knobs (env-overridable for a fast smoke; defaults = full run).
TRAIN_N="${TRAIN_N:-30}"   # train tasks (also the EVAL offset)
EVAL_N="${EVAL_N:-20}"     # eval tasks
FIT_MIN="${FIT_MIN:-500}"  # fitter --min-samples (lower for tiny smoke)
CAP_LIMIT="${CAP_LIMIT:-}" # capture_target_probs --limit-rids (empty = all)
mkdir -p "$BASE/$DIR"

MODEL=$(python3 -c "import sys; sys.path.insert(0,'simulation/scripts'); from measure_chain_hybrid import MODEL_PRESETS; print(MODEL_PRESETS['$PRESET']['model'])")
[ -n "$MODEL" ] || { echo "ABORT: could not resolve model for preset $PRESET"; exit 1; }
echo "target model = $MODEL ; objective = $OBJ"

wait_gpu0() {
  echo "=== waiting for GPU0 (<15GB) ==="
  while true; do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    echo "  GPU0 used=${U} MiB $(date +%H:%M:%S)"; [ "${U:-99999}" -lt 15000 ] && break; sleep 120
  done
  sleep 20
}

TRAINDIR="${REUSE_TRAIN:-$BASE/${DIR}_train}"
if [ -z "${REUSE_TRAIN:-}" ]; then
  wait_gpu0
  echo "=== STEP D: train-oracle (record+oracle, $TRAIN_N train, offset 0) tail=$TAIL ndt=$NDT ==="
  python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
    --n-tasks "$TRAIN_N" --train-n-tasks 0 --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
    --arms record,select1_oracle --output "$TRAINDIR/run.json" $EXTRA
  echo "STEP_D_RC=$?"
else
  echo "=== STEP D skipped: reusing train dir $TRAINDIR ==="
fi
TR="$TRAINDIR/decisions_select1_oracle.jsonl"
GT_TRAIN="$TRAINDIR/gt_tokens.jsonl"
[ -s "$TR" ] || { echo "ABORT: train-oracle log missing ($TR)"; exit 1; }
[ -s "$GT_TRAIN" ] || { echo "ABORT: train gt_tokens.jsonl missing ($GT_TRAIN)"; exit 1; }

FITLABEL=""
if [ "$OBJ" = "target_p" ]; then
  echo "=== STEP E.5: capture q_target via offline target teacher-forcing ==="
  wait_gpu0   # train server killed; reclaim GPU0 for HF
  TP="$TRAINDIR/target_probs.jsonl"
  python3 simulation/scripts/capture_target_probs.py \
    --decision-log "$TR" --gt "$GT_TRAIN" --model "$MODEL" --out "$TP" \
    ${CAP_LIMIT:+--limit-rids "$CAP_LIMIT"}
  echo "STEP_E5_RC=$?"
  [ -s "$TP" ] || { echo "ABORT: target_probs.jsonl missing"; exit 1; }
  FITLABEL="--target-prob-labels $TP"
else
  FITLABEL="--oracle-labels"
fi

echo "=== STEP E: fit per-position calib maps ALL-TRAINED + COND-TRAINED ($OBJ) ==="
python3 simulation/scripts/fit_chain_hybrid_calib_perpos.py \
  --decision-log "$TR" $FITLABEL --min-samples "$FIT_MIN" --out-dir "$BASE/$DIR/calib_all-trained"
echo "FIT_ALL_TRAINED_RC=$?"
python3 simulation/scripts/fit_chain_hybrid_calib_perpos.py \
  --decision-log "$TR" $FITLABEL --accept-conditioned --min-samples "$FIT_MIN" \
  --out-dir "$BASE/$DIR/calib_cond-trained"
echo "FIT_COND_TRAINED_RC=$?"
[ -s "$BASE/$DIR/calib_all-trained/calib_pp_isotonic.json" ] || { echo "ABORT: all-trained calib maps missing"; exit 1; }
[ -s "$BASE/$DIR/calib_cond-trained/calib_pp_isotonic.json" ] || { echo "ABORT: cond-trained calib maps missing"; exit 1; }

echo "=== STEP G1: PASS1 ALL-TRAINED maps — replay-all (re-records EVAL trajectory) + pin ==="
wait_gpu0
cp "$BASE/$DIR"/calib_all-trained/calib_pp_*.json "$BASE/$DIR"/
python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
  --train-n-tasks "$TRAIN_N" --n-tasks "$EVAL_N" --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
  --replay-all --skip-calib-fit --pin-trajectory \
  --arms record,baseline,select1,select1_calib_histogram,select1_calib_isotonic,select1_calib_logistic,select1_calib_beta,select1_oracle \
  --output "$BASE/$DIR/run_all-trained.json" $EXTRA
echo "STEP_G1_RC=$?"
for m in $METHODS; do
  for k in timing decisions; do
    [ -f "$BASE/$DIR/${k}_select1_calib_${m}.jsonl" ] && \
      mv -f "$BASE/$DIR/${k}_select1_calib_${m}.jsonl" "$BASE/$DIR/${k}_select1_calib_${m}_all-trained.jsonl"
  done
done

echo "=== STEP G2: PASS2 COND-TRAINED maps — replay-existing (reuse PASS1 record) + pin ==="
wait_gpu0
cp "$BASE/$DIR"/calib_cond-trained/calib_pp_*.json "$BASE/$DIR"/
python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
  --train-n-tasks "$TRAIN_N" --n-tasks "$EVAL_N" --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
  --replay-existing --skip-calib-fit --pin-trajectory \
  --arms select1_calib_histogram,select1_calib_isotonic,select1_calib_logistic,select1_calib_beta \
  --output "$BASE/$DIR/run_cond-trained.json" $EXTRA
echo "STEP_G2_RC=$?"
for m in $METHODS; do
  for k in timing decisions; do
    [ -f "$BASE/$DIR/${k}_select1_calib_${m}.jsonl" ] && \
      mv -f "$BASE/$DIR/${k}_select1_calib_${m}.jsonl" "$BASE/$DIR/${k}_select1_calib_${m}_cond-trained.jsonl"
  done
done

echo "=== suffix-only sim + plots ($OBJ) ==="
# sim_suffix_only annotates out_dir/run.json; this study writes
# run_all-trained/cond-trained.json, so give it a run.json to update
# (timing_suffix.jsonl is what the plot consumes).
[ -f "$BASE/$DIR/run_all-trained.json" ] && cp -f "$BASE/$DIR/run_all-trained.json" "$BASE/$DIR/run.json"
python3 simulation/scripts/sim_suffix_only.py --gt "$BASE/$DIR/gt_tokens.jsonl" --out-dir "$BASE/$DIR" || echo "suffix-sim failed"
python3 simulation/scripts/plot_o4_objectives.py --dir "$BASE/$DIR" --objective "$OBJ" --variants all-trained,cond-trained || echo "plot failed"
echo "CHAIN_HYBRID_PERDEPTH_${DIR}_${OBJ}_DONE"
