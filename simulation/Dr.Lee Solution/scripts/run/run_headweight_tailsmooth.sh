#!/usr/bin/env bash
# ============================================================================
# HEAD-WEIGHT × TAIL-SMOOTHING ARM MATRIX (2026-07-16)
#
# User ask: "head = online beta calibration, tail = laplace smoothing" — plus
# diverse variants: single-scalar heads, fixed head-weight + tail smoothing,
# offline-calibrated head + smoothing, and smoothed-and-scaled tails. Every arm
# is a deployable per-depth composition (--props calib) evaluated on the TEST
# half of the disjoint 3-way convlabel split; MAT = mean accepted / round.
#
# Building blocks
#   head:  online-beta / online-logistic  (windowed, deployable)
#          scale u  = min(1, u*conf)       (ONE scalar, no fit, no window)
#          beta/logistic (offline, fit on calibrate half)
#          raw / affine(0.69c+0.29)        (existing baselines, not re-run here)
#   tail:  succession (k+1)/(n+2) Laplace  |  kt (k+.5)/(n+1) Jeffreys  (0 param)
#          fixed w = w*rawscore            (ONE scalar)
#          succscale = w*Laplace           (head weight + scaled smoothing)
#          raw = identity                  (single-weight-head arms)
#
# NEW online-tail=raw path (replay_extension.py): keeps the online tail map at
# identity so the Laplace/KT rescore inside ArcticSuffix IS the tail estimate
# (--online-head beta --online-tail raw --tail-cal succession).
#
#   docker exec -d sglang-bench bash \
#     "/workspace/simulation/Dr.Lee Solution/scripts/run/run_headweight_tailsmooth.sh"
#
# Outputs
#   replay logs  readable_outputs/figures/replay_logs/mat_{ds}_hwts_{arm}_split.replay.txt
#   status       results/headweight_tailsmooth/{driver.log,DONE,FAILED,failed_jobs.txt}
# ============================================================================
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RLOG=readable_outputs/figures/replay_logs
ST=results/headweight_tailsmooth
GM=convlabel
JOBS=${JOBS:-12}
mkdir -p "$ST" "$RLOG"
rm -f "$ST/DONE" "$ST/FAILED"

log() { echo "[$(TZ=Asia/Seoul date +%H:%M)KST] $*" >> "$ST/driver.log"; }
fail() { log "FAILED: $*"; echo "$*" > "$ST/FAILED"; exit 1; }

declare -A REC=(
  [specbench]=results/perpos_specbench_alleval/specbench_4way.jsonl
  [bfcl]=results/perpos_bfcl_alleval/bfcl_4way.jsonl
  [swebench]=results/perpos_swebench_alleval/swebench_4way.jsonl
  [spider]=results/perpos_spider_alleval/spider_4way.jsonl
  [tau2]=results/perpos_tau2_alleval/tau2_4way.jsonl)
DSS="specbench bfcl swebench spider tau2"

declare -A ARMS=(
  # --- A: online head + Laplace/KT tail (THE ASK + neighbors) ---------------
  [onlbeta_succ_raw]="--online-calib --online-window 8000 --online-head beta --online-tail raw --tail-cal succession"
  [onlbeta_kt_raw]="--online-calib --online-window 8000 --online-head beta --online-tail raw --tail-cal kt"
  [onllog_succ_raw]="--online-calib --online-window 8000 --online-head logistic --online-tail raw --tail-cal succession"
  [onlbeta_succ_iso]="--online-calib --online-window 8000 --online-head beta --online-tail isotonic --tail-cal succession"
  [onlbeta_kt_iso]="--online-calib --online-window 8000 --online-head beta --online-tail isotonic --tail-cal kt"
  # --- B: fixed head-weight u (scale) + Laplace tail (ONE scalar, no fit) ----
  [hs10_succ]="--head-cal scale --head-scale 1.0 --tail-cal succession"
  [hs12_succ]="--head-cal scale --head-scale 1.2 --tail-cal succession"
  [hs14_succ]="--head-cal scale --head-scale 1.4 --tail-cal succession"
  [hs16_succ]="--head-cal scale --head-scale 1.6 --tail-cal succession"
  [hs18_succ]="--head-cal scale --head-scale 1.8 --tail-cal succession"
  # --- B': fixed head-weight + KT tail --------------------------------------
  [hs12_kt]="--head-cal scale --head-scale 1.2 --tail-cal kt"
  [hs14_kt]="--head-cal scale --head-scale 1.4 --tail-cal kt"
  [hs16_kt]="--head-cal scale --head-scale 1.6 --tail-cal kt"
  # --- C: SINGLE weight only ------------------------------------------------
  [hs12_rawtail]="--head-cal scale --head-scale 1.2 --tail-cal raw"
  [hs14_rawtail]="--head-cal scale --head-scale 1.4 --tail-cal raw"
  [hs16_rawtail]="--head-cal scale --head-scale 1.6 --tail-cal raw"
  [rawhead_fix010]="--head-cal raw --tail-cal fixed --tail-scale 0.10"
  [rawhead_fix0125]="--head-cal raw --tail-cal fixed --tail-scale 0.125"
  [rawhead_fix015]="--head-cal raw --tail-cal fixed --tail-scale 0.15"
  [rawhead_fix020]="--head-cal raw --tail-cal fixed --tail-scale 0.20"
  # --- D: offline calibrated head + Laplace/KT ------------------------------
  [betahead_succ]="--head-cal beta --tail-cal succession"
  [loghead_succ]="--head-cal logistic --tail-cal succession"
  [betahead_kt]="--head-cal beta --tail-cal kt"
  # --- E: head weight u + scaled Laplace (succscale = w*Laplace) ------------
  [hs14_succscale05]="--head-cal scale --head-scale 1.4 --tail-cal succscale --tail-scale 0.5"
  [hs14_succscale075]="--head-cal scale --head-scale 1.4 --tail-cal succscale --tail-scale 0.75"
  [hs14_succscale10]="--head-cal scale --head-scale 1.4 --tail-cal succscale --tail-scale 1.0"
  [hs16_succscale075]="--head-cal scale --head-scale 1.6 --tail-cal succscale --tail-scale 0.75"
  # --- anchor: two-scalar compose reproduction (u=1.4, w=0.125*raw) ---------
  [twoscalar_h14w0125]="--head-cal scale --head-scale 1.4 --tail-cal fixed --tail-scale 0.125"
)

for ds in $DSS; do [ -f "${REC[$ds]}" ] || fail "record ${REC[$ds]} missing"; done

JOBLIST="$ST/jobs.txt"; : > "$JOBLIST"
for ds in $DSS; do for arm in "${!ARMS[@]}"; do echo "$ds $arm"; done; done \
  | shuf --random-source=<(yes 42) > "$JOBLIST"   # mix fast/slow workloads across lanes
NJOBS=$(wc -l < "$JOBLIST")
log "start: $NJOBS jobs (${#ARMS[@]} arms x 5 workloads), JOBS=$JOBS parallel"

: > "$ST/failed_jobs.txt"
run_one() {
  local ds=$1 arm=$2
  local out="$RLOG/mat_${ds}_hwts_${arm}_split.replay.txt"
  local err="$ST/re_${ds}_${arm}.err"
  # shellcheck disable=SC2086
  python3 scripts/replay_extension.py --record "${REC[$ds]}" --max-rounds 4096 \
    --props calib --three-way --group-mode $GM ${ARMS[$arm]} \
    > "$out" 2> "$err"
  if grep -q "calib: K=" "$out"; then rm -f "$err"; else
    echo "$ds $arm" >> "$ST/failed_jobs.txt"
  fi
}

running=0
while read -r ds arm; do
  run_one "$ds" "$arm" &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running-1)); fi
done < "$JOBLIST"
wait

NFAIL=$(wc -l < "$ST/failed_jobs.txt")
if [ "$NFAIL" -ne 0 ]; then
  log "WARNING: $NFAIL jobs produced no K (see failed_jobs.txt)"
  fail "$NFAIL replays failed"
fi
log "ALL $NJOBS HEAD-WEIGHT x TAIL-SMOOTHING REPLAYS DONE"
touch "$ST/DONE"
