#!/bin/bash
# Full DEFAULT chain-hybrid pipeline (oracle-fit calibration), one shot:
#   STEP D  record+oracle on 30 TRAIN tasks (offset 0)  -> train-oracle log
#   STEP E  fit oracle-fit per-position calib maps (token==gt) into eval dir
#   STEP G  8-arm eval on 20 TEST tasks (offset 30), replay-all, --skip-calib-fit
#   then    suffix-only offline sim + plot_o4 (MAT, survival x2, EAGLE3-chosen)
#
# Tail-parameterized. For Mamba/27B with tail>0, pass NDT = steps+1+TAIL so the
# static mamba spec cache is oversized (server_args topk==1 reset bypassed via
# SGLANG_CHAIN_HYBRID_TAIL>0). Waits for GPU0 first. Run inside sglang-bench root.
#
# Usage:  run_o4_default.sh <PRESET> <PORT> <DIR> <TAIL> <NDT>
#   14B tail=64:  run_o4_default.sh qwen3_14b 30021 qwen3_14b_def 64 0
#   27B tail=32:  run_o4_default.sh qwen35_27b_mtp 30022 qwen35_27b_default 32 49
#   27B tail=off: run_o4_default.sh qwen35_27b_mtp 30022 qwen35_27b_default 0 0
set -uo pipefail
PRESET="${1:?preset}"; PORT="${2:?port}"; DIR="${3:?dir}"; TAIL="${4:?tail}"; NDT="${5:?ndt(0=default)}"
cd /workspace
export CUDA_VISIBLE_DEVICES=0
# Pick the first CUDA toolkit with a real nvcc (sglang JIT-compiles sm_120
# kernels at boot). Container drifted 12.8 -> 13.0; hardcoding 12.8 fails.
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [ -x "$_c/bin/nvcc" ]; then export CUDA_HOME="$_c"; break; fi
done
: "${CUDA_HOME:?no CUDA toolkit with nvcc found under /usr/local}"
export PATH="$CUDA_HOME/bin:$PATH"
echo "using CUDA_HOME=$CUDA_HOME"
COMMON="--workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1"
EXTRA="--extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch"
BASE="simulation/results/o4_perdepth"
NDTARG=""
[ "$NDT" != "0" ] && NDTARG="--spec-num-draft-tokens $NDT"

echo "=== waiting for GPU0 (<15GB) ==="
while true; do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  echo "  GPU0 used=${U} MiB $(date +%H:%M:%S)"; [ "${U:-99999}" -lt 15000 ] && break; sleep 120
done
sleep 20

echo "=== STEP D: train-oracle (record+oracle, 30 train, offset 0) tail=$TAIL ndt=$NDT ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" $COMMON \
  --n-tasks 30 --train-n-tasks 0 --port "$PORT" --tail-max-tokens "$TAIL" $NDTARG \
  --arms record,select1_oracle --output "$BASE/${DIR}_train/run.json" $EXTRA
echo "STEP_D_RC=$?"
TR="$BASE/${DIR}_train/decisions_select1_oracle.jsonl"
[ -s "$TR" ] || { echo "ABORT: train-oracle log missing"; exit 1; }

echo "=== STEP E: fit ORACLE-FIT (token==gt) per-position calib maps ==="
python3 simulation/scripts/fit_chain_hybrid_calib_perpos.py \
  --decision-log "$TR" --out-dir "$BASE/$DIR" --oracle-labels
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
echo "O4_DEFAULT_${DIR}_DONE"
