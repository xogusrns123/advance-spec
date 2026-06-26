#!/bin/bash
# Chain-hybrid O4 pipeline with the CONTINUOUS target-probability calibration
# objective (regress each proposer's score onto q_target(token|GT prefix) instead
# of the binary token==gt accept event). One shot:
#   STEP D    record+oracle on 30 TRAIN tasks (offset 0)  -> train-oracle log
#             (the oracle arm now also logs per-rid input_ids "req" rows)
#   STEP E.5  capture q_target for each drafted token via an offline target
#             teacher-forced rerun        -> target_probs.jsonl
#   STEP E    fit per-position calib maps on the CONTINUOUS q_target label
#             (meta.label=target_p)       -> calib_pp_*.json in the eval dir
#   STEP G    8-arm eval on 20 TEST tasks (offset 30), replay-all, --skip-calib-fit
#   then      suffix-only offline sim + plot_o4 (MAT, survival x2, EAGLE3-chosen)
#
# Differs from run_o4_default.sh only in STEP E.5 (new) and STEP E (label). The
# serving arms (select1_calib_*) are unchanged — they load whatever calib_pp_*.json
# sits in the eval dir, so a target_p map IS the experiment. Compare within-run
# (calib-raw) and (oracle-raw) MAT deltas against the token_gt run from
# run_o4_default.sh (raw MAT drifts run-to-run via live web_search).
#
# Tail-parameterized exactly like run_o4_default.sh.
#
# Usage:  run_o4_targetprob.sh <PRESET> <PORT> <DIR> <TAIL> <NDT>
#   14B tail=64:  run_o4_targetprob.sh qwen3_14b 30021 qwen3_14b_tp 64 0
#   27B tail=32:  run_o4_targetprob.sh qwen35_27b_mtp 30022 qwen35_27b_tp 32 49
set -uo pipefail
PRESET="${1:?preset}"; PORT="${2:?port}"; DIR="${3:?dir}"; TAIL="${4:?tail}"; NDT="${5:?ndt(0=default)}"
cd /workspace
export CUDA_VISIBLE_DEVICES=0
# Pick the first CUDA toolkit that actually has nvcc — sglang JIT-compiles the
# fused-rope kernel for sm_120 (Blackwell) at server boot and needs a real nvcc
# that supports compute_120 (>=12.8). The container has drifted 12.8 -> 13.0
# (cu130 sglang stack), so hardcoding 12.8 fails with "nvcc: not found".
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
: "${CUDA_HOME:?no CUDA toolkit with nvcc found under /usr/local}"
export PATH="$CUDA_HOME/bin:$PATH"
echo "using CUDA_HOME=$CUDA_HOME ($($CUDA_HOME/bin/nvcc --version | tail -1))"
COMMON="--workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1"
EXTRA="--extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch"
BASE="simulation/results/o4_perdepth"
NDTARG=""
[ "$NDT" != "0" ] && NDTARG="--spec-num-draft-tokens $NDT"

# Resolve the TARGET HF model name from the same preset table measure uses.
MODEL=$(python3 -c "import sys; sys.path.insert(0,'simulation/scripts'); from measure_chain_hybrid import MODEL_PRESETS; print(MODEL_PRESETS['$PRESET']['model'])")
[ -n "$MODEL" ] || { echo "ABORT: could not resolve model for preset $PRESET"; exit 1; }
echo "target model = $MODEL"

wait_gpu0() {
  echo "=== waiting for GPU0 (<15GB) ==="
  while true; do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    echo "  GPU0 used=${U} MiB $(date +%H:%M:%S)"; [ "${U:-99999}" -lt 15000 ] && break; sleep 120
  done
  sleep 20
}

wait_gpu0
echo "=== STEP D: train-oracle (record+oracle, 30 train, offset 0) tail=$TAIL ndt=$NDT ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
  --n-tasks 30 --train-n-tasks 0 --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
  --arms record,select1_oracle --output "$BASE/${DIR}_train/run.json" $EXTRA
echo "STEP_D_RC=$?"
TR="$BASE/${DIR}_train/decisions_select1_oracle.jsonl"
GT="$BASE/${DIR}_train/gt_tokens.jsonl"
[ -s "$TR" ] || { echo "ABORT: train-oracle log missing"; exit 1; }
[ -s "$GT" ] || { echo "ABORT: gt_tokens.jsonl missing"; exit 1; }

echo "=== STEP E.5: capture q_target via offline target teacher-forcing ==="
wait_gpu0   # the eval server from STEP D has been killed; reclaim GPU0 for HF
python3 simulation/scripts/capture_target_probs.py \
  --decision-log "$TR" --gt "$GT" --model "$MODEL" \
  --out "$BASE/${DIR}_train/target_probs.jsonl"
echo "STEP_E5_RC=$?"
TP="$BASE/${DIR}_train/target_probs.jsonl"
[ -s "$TP" ] || { echo "ABORT: target_probs.jsonl missing"; exit 1; }

echo "=== STEP E: fit CONTINUOUS target-prob per-position calib maps ==="
python3 simulation/scripts/fit_chain_hybrid_calib_perpos.py \
  --decision-log "$TR" --target-prob-labels "$TP" --out-dir "$BASE/$DIR"
echo "STEP_E_RC=$?"
[ -s "$BASE/$DIR/calib_pp_isotonic.json" ] || { echo "ABORT: calib maps missing"; exit 1; }

echo "=== STEP G: 8-arm eval (offset 30), replay-all, --skip-calib-fit, tail=$TAIL ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
  --train-n-tasks 30 --n-tasks 20 --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
  --replay-all --skip-calib-fit \
  --arms record,baseline,select1,select1_calib_histogram,select1_calib_isotonic,select1_calib_logistic,select1_calib_beta,select1_oracle \
  --output "$BASE/$DIR/run.json" $EXTRA
echo "STEP_G_RC=$?"

echo "=== suffix-only sim + plots ==="
python3 simulation/scripts/sim_suffix_only.py --gt "$BASE/$DIR/gt_tokens.jsonl" --out-dir "$BASE/$DIR" || echo "suffix-sim failed"
python3 simulation/scripts/plot_o4.py --dir "$BASE/$DIR" || echo "plot failed"
echo "O4_TARGETPROB_${DIR}_DONE"
