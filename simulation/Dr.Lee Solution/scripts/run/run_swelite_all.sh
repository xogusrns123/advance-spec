#!/usr/bin/env bash
# SWE-bench Lite method-families data: base arms + fallback + method sweep, same
# TEST-half three-way convlabel protocol as the other workloads. ds-tag=swelite,
# record = perpos_swebench_lite/swebench_4way.jsonl. CPU-only.
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
REC=results/perpos_swebench_lite/swebench_4way.jsonl
RLOG=readable_outputs/figures/replay_logs
SEG=/workspace/simulation/results/pipeline_deployable/segments
ST=results/method_families_swelite
mkdir -p "$ST" "$SEG"
log(){ echo "[$(TZ=Asia/Seoul date +%H:%M:%S)KST] $*" >> "$ST/driver.log"; }
GM=convlabel

log "START base+fallback"
# base arms (parallel)
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 --props dflash suffix \
  --three-way --group-mode $GM > "$RLOG/mat_swelite_4way_singles_split.replay.txt" 2> "$ST/singles.err" &
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 --props calib --raw-cals \
  --three-way --group-mode $GM > "$RLOG/mat_swelite_4way_calib_raw_split.replay.txt" 2> "$ST/calibraw.err" &
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 --props oracle \
  --three-way --group-mode $GM > "$RLOG/mat_swelite_4way_oracle_split.replay.txt" 2> "$ST/oracle.err" &
python3 scripts/replay_fallback_sweep.py --records "swelite=$REC" --taus 0.5 1 2 4 8 16 32 \
  --max-rounds 4096 --three-way --group-mode $GM --out "$SEG/fallback_sweep_fresh_swelite.json" \
  > "$ST/fallback.log" 2>&1 &
wait
log "base+fallback done"

# method sweep
log "START method sweep (102 arms)"
JOBS=8 MAXBIG=8 JOBSFILE=scripts/run/mfam_jobs/jobs_swelite.txt ST="$ST" \
  bash scripts/run/run_mfam_jobs.sh
log "ALL SWELITE DONE"
touch "$ST/ALL_DONE"
