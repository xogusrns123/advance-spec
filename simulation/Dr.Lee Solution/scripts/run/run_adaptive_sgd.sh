#!/usr/bin/env bash
# ============================================================================
# EXPLICIT-LOSS SGD arms (user formulation) + gamma extension (2026-07-19)
#
#   sgdgain — per-side gain-residual loss, BOTH scalars continuous:
#             head L=(G_hat(u)-head_acc)^2, tail L=(w*s-tail_acc)^2 at chosen k
#             (verify-time labels only -> deployable). Per-round SGD.
#   sgdmat  — total-accept loss, single continuous w: V(w) = sum_k
#             softmax(beta*val_k(w))*acc_k, per-round gradient ascent on the
#             counterfactual per-k accept vector (full-info; _cens = censored).
#   ratio gamma extension: calib-half gamma 0.5/0.6 (curve still rising at 0.4)
#   test-half runs of the current calib-half argmax gammas (ratio 0.4, sr 0.45).
#
#   docker exec -d -e JOBS=10 sglang-bench bash \
#     "/workspace/simulation/Dr.Lee Solution/scripts/run/run_adaptive_sgd.sh"
# ============================================================================
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RLOG=readable_outputs/figures/replay_logs
ST=results/adaptive_scalar
JOBS=${JOBS:-10}
mkdir -p "$ST" "$RLOG"
rm -f "$ST/SGD_DONE" "$ST/SGD_FAILED"

log() { echo "[$(TZ=Asia/Seoul date +%H:%M)KST] $*" >> "$ST/driver.log"; }

declare -A REC=(
  [specbench]=results/perpos_specbench_alleval/specbench_4way.jsonl
  [bfcl]=results/perpos_bfcl_alleval/bfcl_4way.jsonl
  [swebench]=results/perpos_swebench_alleval/swebench_4way.jsonl
  [spider]=results/perpos_spider_alleval/spider_4way.jsonl
  [tau2]=results/perpos_tau2_alleval/tau2_4way.jsonl)
DSS="specbench bfcl swebench spider tau2"

# test-half arms
declare -A ARMS=(
  [sgdgain_e3]="--adaptive sgdgain --adaptive-eta 1e-3 --adaptive-eta-head 1e-4 --adaptive-init 0.125 --adaptive-init-u 1.4"
  [sgdgain_e4]="--adaptive sgdgain --adaptive-eta 3e-4 --adaptive-eta-head 3e-5 --adaptive-init 0.125 --adaptive-init-u 1.4"
  [sgdmat_e4]="--adaptive sgdmat --adaptive-head raw --adaptive-eta 1e-4 --adaptive-beta 10 --adaptive-init 0.075"
  [sgdmat_e3]="--adaptive sgdmat --adaptive-head raw --adaptive-eta 3e-4 --adaptive-beta 10 --adaptive-init 0.075"
  [sgdmat_cens_e4]="--adaptive sgdmat --adaptive-head raw --adaptive-eta 1e-4 --adaptive-beta 10 --adaptive-init 0.075 --adaptive-censor"
  [ratio_g04]="--adaptive ratio --adaptive-head raw --adaptive-gamma 0.4 --adaptive-window 8000"
  [sr_g045]="--adaptive succratio --adaptive-head raw --adaptive-gamma 0.45 --adaptive-window 8000"
)
# calib-half gamma extension
declare -A CARMS=(
  [ratio_g05]="--adaptive ratio --adaptive-head raw --adaptive-gamma 0.5 --adaptive-window 8000"
  [ratio_g06]="--adaptive ratio --adaptive-head raw --adaptive-gamma 0.6 --adaptive-window 8000"
)

JOBLIST="$ST/sgd_jobs.txt"; : > "$JOBLIST"
for ds in $DSS; do
  for arm in "${!ARMS[@]}"; do echo "$ds $arm test"; done
  for arm in "${!CARMS[@]}"; do echo "$ds $arm calib"; done
done | shuf --random-source=<(yes 42) > "$JOBLIST"
log "sgd batch start: $(wc -l < "$JOBLIST") jobs, JOBS=$JOBS"

: > "$ST/sgd_failed.txt"
run_one() {
  local ds=$1 arm=$2 half=$3
  local flags out
  if [ "$half" = "calib" ]; then
    flags="${CARMS[$arm]} --eval-half calib"
    out="$RLOG/mat_${ds}_adw_${arm}_calibhalf.replay.txt"
  else
    flags="${ARMS[$arm]}"
    out="$RLOG/mat_${ds}_adw_${arm}_split.replay.txt"
  fi
  # shellcheck disable=SC2086
  python3 scripts/replay_extension.py --record "${REC[$ds]}" --max-rounds 4096 \
    --props calib --three-way --group-mode convlabel $flags \
    --adaptive-trace "$ST/trace_${ds}_${arm}.json" \
    > "$out" 2> "$ST/re_${ds}_${arm}.err"
  if grep -q "calib: K=" "$out"; then rm -f "$ST/re_${ds}_${arm}.err"; else
    echo "$ds $arm $half" >> "$ST/sgd_failed.txt"
  fi
}

running=0
while read -r ds arm half; do
  run_one "$ds" "$arm" "$half" &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running-1)); fi
done < "$JOBLIST"
wait

if [ -s "$ST/sgd_failed.txt" ]; then
  log "SGD batch FAILED jobs: $(wc -l < "$ST/sgd_failed.txt")"
  touch "$ST/SGD_FAILED"
else
  log "SGD BATCH DONE"
  touch "$ST/SGD_DONE"
fi
