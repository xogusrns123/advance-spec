#!/usr/bin/env bash
# ============================================================================
# gamma tuning for the ratio / succratio adaptive arms — CALIB half only
# (decision-level fit, same protocol that tuned the fixed champion w).
# The winning gamma* then reads off one test-half arm in the main sweep
# (or one extra run if gamma* is outside {0.2, 0.3} / {0.375}).
#
#   docker exec -d -e JOBS=4 sglang-bench bash \
#     "/workspace/simulation/Dr.Lee Solution/scripts/run/run_adaptive_gamma_tune.sh"
# ============================================================================
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RLOG=readable_outputs/figures/replay_logs
ST=results/adaptive_scalar
JOBS=${JOBS:-4}
mkdir -p "$ST" "$RLOG"
rm -f "$ST/GAMMA_DONE" "$ST/GAMMA_FAILED"

log() { echo "[$(TZ=Asia/Seoul date +%H:%M)KST] $*" >> "$ST/driver.log"; }

declare -A REC=(
  [specbench]=results/perpos_specbench_alleval/specbench_4way.jsonl
  [bfcl]=results/perpos_bfcl_alleval/bfcl_4way.jsonl
  [swebench]=results/perpos_swebench_alleval/swebench_4way.jsonl
  [spider]=results/perpos_spider_alleval/spider_4way.jsonl
  [tau2]=results/perpos_tau2_alleval/tau2_4way.jsonl)
DSS="specbench bfcl swebench spider tau2"

declare -A ARMS=(
  [ratio_g015]="--adaptive ratio --adaptive-head raw --adaptive-gamma 0.15 --adaptive-window 8000"
  [ratio_g02]="--adaptive ratio --adaptive-head raw --adaptive-gamma 0.2 --adaptive-window 8000"
  [ratio_g025]="--adaptive ratio --adaptive-head raw --adaptive-gamma 0.25 --adaptive-window 8000"
  [ratio_g03]="--adaptive ratio --adaptive-head raw --adaptive-gamma 0.3 --adaptive-window 8000"
  [ratio_g04]="--adaptive ratio --adaptive-head raw --adaptive-gamma 0.4 --adaptive-window 8000"
  [sr_g03]="--adaptive succratio --adaptive-head raw --adaptive-gamma 0.3 --adaptive-window 8000"
  [sr_g0375]="--adaptive succratio --adaptive-head raw --adaptive-gamma 0.375 --adaptive-window 8000"
  [sr_g045]="--adaptive succratio --adaptive-head raw --adaptive-gamma 0.45 --adaptive-window 8000"
)

JOBLIST="$ST/gamma_jobs.txt"; : > "$JOBLIST"
for ds in $DSS; do for arm in "${!ARMS[@]}"; do echo "$ds $arm"; done; done \
  | shuf --random-source=<(yes 42) > "$JOBLIST"
log "gamma-tune start: $(wc -l < "$JOBLIST") calib-half jobs, JOBS=$JOBS"

: > "$ST/gamma_failed.txt"
run_one() {
  local ds=$1 arm=$2
  local out="$RLOG/mat_${ds}_adw_${arm}_calibhalf.replay.txt"
  # shellcheck disable=SC2086
  python3 scripts/replay_extension.py --record "${REC[$ds]}" --max-rounds 4096 \
    --props calib --three-way --eval-half calib --group-mode convlabel \
    ${ARMS[$arm]} > "$out" 2> "$ST/re_${ds}_${arm}_ch.err"
  if grep -q "calib: K=" "$out"; then rm -f "$ST/re_${ds}_${arm}_ch.err"; else
    echo "$ds $arm" >> "$ST/gamma_failed.txt"
  fi
}

running=0
while read -r ds arm; do
  run_one "$ds" "$arm" &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running-1)); fi
done < "$JOBLIST"
wait

if [ -s "$ST/gamma_failed.txt" ]; then
  log "GAMMA-TUNE FAILED jobs: $(wc -l < "$ST/gamma_failed.txt")"
  touch "$ST/GAMMA_FAILED"
else
  log "GAMMA-TUNE DONE"
  touch "$ST/GAMMA_DONE"
fi
