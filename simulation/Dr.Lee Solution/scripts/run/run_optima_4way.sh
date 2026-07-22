#!/usr/bin/env bash
# ============================================================================
# 4-FAMILY OPTIMUM SWEEP (2026-07-16): raw+raw / weight+raw / raw+weight /
# weight+weight, each to be compared at its BEST scalar. This driver fills the
# grid points not yet run: head-only high-u tail (weight+raw peak) and the
# two-scalar (u,w) grid around the low-w tail optimum. tail-only w=0.05/0.075
# and the earlier w=0.10..0.20 come from the hwts logs already on disk.
#   docker exec -d sglang-bench bash \
#     "/workspace/simulation/Dr.Lee Solution/scripts/run/run_optima_4way.sh"
# Outputs: readable_outputs/figures/replay_logs/mat_{ds}_hwts_{arm}_split.replay.txt
#          results/optima_4way/{driver.log,DONE,FAILED}
# ============================================================================
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RLOG=readable_outputs/figures/replay_logs
ST=results/optima_4way
GM=convlabel
JOBS=${JOBS:-10}
mkdir -p "$ST" "$RLOG"; rm -f "$ST/DONE" "$ST/FAILED"
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
  # weight+raw (head-only): push u past 1.6 to find the peak
  [hs18_rawtail]="--head-cal scale --head-scale 1.8 --tail-cal raw"
  [hs20_rawtail]="--head-cal scale --head-scale 2.0 --tail-cal raw"
  [hs25_rawtail]="--head-cal scale --head-scale 2.5 --tail-cal raw"
  # weight+weight (two-scalar): (u,w) grid around the low-w tail optimum
  [h14w005]="--head-cal scale --head-scale 1.4 --tail-cal fixed --tail-scale 0.05"
  [h14w0075]="--head-cal scale --head-scale 1.4 --tail-cal fixed --tail-scale 0.075"
  [h14w010]="--head-cal scale --head-scale 1.4 --tail-cal fixed --tail-scale 0.10"
  [h16w005]="--head-cal scale --head-scale 1.6 --tail-cal fixed --tail-scale 0.05"
  [h16w0075]="--head-cal scale --head-scale 1.6 --tail-cal fixed --tail-scale 0.075"
  [h16w010]="--head-cal scale --head-scale 1.6 --tail-cal fixed --tail-scale 0.10"
)

for ds in $DSS; do [ -f "${REC[$ds]}" ] || fail "record ${REC[$ds]} missing"; done
JOBLIST="$ST/jobs.txt"; : > "$JOBLIST"
for ds in $DSS; do for arm in "${!ARMS[@]}"; do echo "$ds $arm"; done; done \
  | shuf --random-source=<(yes 42) > "$JOBLIST"
NJOBS=$(wc -l < "$JOBLIST")
log "start: $NJOBS jobs (${#ARMS[@]} arms x 5 wl), JOBS=$JOBS"

: > "$ST/failed_jobs.txt"
run_one() {
  local ds=$1 arm=$2
  local out="$RLOG/mat_${ds}_hwts_${arm}_split.replay.txt"
  # shellcheck disable=SC2086
  python3 scripts/replay_extension.py --record "${REC[$ds]}" --max-rounds 4096 \
    --props calib --three-way --group-mode $GM ${ARMS[$arm]} \
    > "$out" 2> "$ST/re_${ds}_${arm}.err"
  grep -q "calib: K=" "$out" && rm -f "$ST/re_${ds}_${arm}.err" || echo "$ds $arm" >> "$ST/failed_jobs.txt"
}
running=0
while read -r ds arm; do
  run_one "$ds" "$arm" &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running-1)); fi
done < "$JOBLIST"
wait
NFAIL=$(wc -l < "$ST/failed_jobs.txt")
[ "$NFAIL" -ne 0 ] && fail "$NFAIL replays failed"
log "ALL $NJOBS OPTIMA-4WAY REPLAYS DONE"; touch "$ST/DONE"
