#!/usr/bin/env bash
# ============================================================================
# METHOD-FAMILIES JOB RUNNER (2026-07-18)  — weight / online-calib / smoothing
# arms for the 4/6-way MAT-per-workload extension. Every arm is a deployable
# compose (--props calib) on the TEST half of the disjoint 3-way convlabel
# split; MAT = mean accepted / round. Same protocol as run_headweight_tailsmooth
# and run_deployable_split so ALL bars (base split logs + these) are comparable.
#
# Usage (inside sglang-bench, CPU-only):
#   JOBS=10 JOBSFILE=/path/to/jobs.txt bash scripts/run/run_mfam_jobs.sh
#
# jobs.txt: one job per line "ds arm <extra replay_extension flags>"
#   e.g.  specbench hs30_rawtail --head-cal scale --head-scale 3.0 --tail-cal raw
# Output: readable_outputs/figures/replay_logs/mat_{ds}_mfam_{arm}_split.replay.txt
# Skips a job whose output already exists with a "calib: K=" line (idempotent).
# Big records (specbench) are capped to MAXBIG concurrent to bound memory.
# ============================================================================
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RLOG=readable_outputs/figures/replay_logs
JOBS=${JOBS:-10}
MAXBIG=${MAXBIG:-5}          # max concurrent specbench (1.2G record) procs
JOBSFILE=${JOBSFILE:?set JOBSFILE}
ST=${ST:-results/method_families}
mkdir -p "$ST" "$RLOG"

declare -A REC=(
  [specbench]=results/perpos_specbench_alleval/specbench_4way.jsonl
  [bfcl]=results/perpos_bfcl_alleval/bfcl_4way.jsonl
  [swebench]=results/perpos_swebench_alleval/swebench_4way.jsonl
  [spider]=results/perpos_spider_alleval/spider_4way.jsonl
  [tau2]=results/perpos_tau2_alleval/tau2_4way.jsonl
  [swelite]=results/perpos_swebench_lite/swebench_4way.jsonl)

log() { echo "[$(TZ=Asia/Seoul date +%H:%M:%S)KST] $*" >> "$ST/driver.log"; }

: > "$ST/failed_jobs.txt"
big_running=0

run_one() {
  local ds=$1 arm=$2; shift 2
  local out="$RLOG/mat_${ds}_mfam_${arm}_split.replay.txt"
  local err="$ST/re_${ds}_${arm}.err"
  if grep -q "calib: K=" "$out" 2>/dev/null; then
    log "skip (exists) $ds $arm"; return 0
  fi
  # shellcheck disable=SC2086
  python3 scripts/replay_extension.py --record "${REC[$ds]}" --max-rounds 4096 \
    --props calib --three-way --group-mode convlabel "$@" \
    > "$out" 2> "$err"
  if grep -q "calib: K=" "$out"; then rm -f "$err"; log "done $ds $arm"; else
    echo "$ds $arm" >> "$ST/failed_jobs.txt"; log "FAIL $ds $arm"
  fi
}

NJOBS=$(grep -cve '^\s*$' "$JOBSFILE")
log "START $NJOBS jobs, JOBS=$JOBS MAXBIG=$MAXBIG"
running=0
while read -r ds arm rest; do
  [ -z "$ds" ] && continue
  # throttle specbench (big record) to MAXBIG concurrent ([s] avoids self-match)
  if [ "$ds" = "specbench" ]; then
    while [ "$(pgrep -fc '[s]pecbench_4way.jsonl')" -ge "$MAXBIG" ]; do sleep 3; done
  fi
  # shellcheck disable=SC2086
  run_one "$ds" "$arm" $rest &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running-1)); fi
done < "$JOBSFILE"
wait

NFAIL=$(grep -cve '^\s*$' "$ST/failed_jobs.txt" 2>/dev/null || echo 0)
log "ALL DONE. failed=$NFAIL"
[ "$NFAIL" -eq 0 ] && touch "$ST/DONE" || touch "$ST/DONE_WITH_FAILURES"
