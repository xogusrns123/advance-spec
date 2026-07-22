#!/usr/bin/env bash
# Per-segment MAT replays (CPU only) on the 2026-07-03 snapshot records, all
# four workloads x five arms, 3 processes per workload:
#   A: dflash suffix fallback (tau = pooled-best from fallback_sweep_pre.json)
#   B: calib   (in-sample beta hazard + isotonic tail — the heavy probe arm)
#   C: oracle  (W+1 tail speculations per round)
#
#   docker exec -d sglang-bench bash -c \
#     'nohup bash "/workspace/simulation/Dr.Lee Solution/scripts/run_segments_5way.sh" \
#      > /workspace/simulation/results/pipeline_4way/segments/driver_seg.out 2>&1'
#
# Markers: segments/DONE_SEG / FAILED_SEG.
set -uo pipefail
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace

SEG=/workspace/simulation/results/pipeline_4way/segments
mkdir -p "$SEG"
rm -f "$SEG/DONE_SEG" "$SEG/FAILED_SEG"

declare -A REC=(
  [spider]=results/perpos_spider/spider.jsonl
  [specbench]=results/perpos_specbench_full/specbench.jsonl
  [swebench]=results/perpos_swebench/swebench_quick.jsonl
  [bfcl]=results/perpos_bfcl_full/bfcl_v4_full.jsonl
)
declare -A TAU=([spider]=16 [specbench]=32 [swebench]=16 [bfcl]=16)

log() { echo "[$(date -u +%F' '%T)UTC] $*" | tee -a "$SEG/seg.log"; }

PIDS=()
for ds in spider specbench swebench bfcl; do
  rec=${REC[$ds]}
  [ -f "$rec" ] || { log "SKIP $ds: $rec missing"; continue; }
  log "launch $ds (tau=${TAU[$ds]})"
  python3 scripts/replay_segments_5way.py --record "$rec" --kind "$ds" \
    --arms dflash suffix fallback --tau "${TAU[$ds]}" \
    --out-dir "$SEG" --tag _pre > "$SEG/run_${ds}_A.log" 2>&1 &
  PIDS+=($!)
  python3 scripts/replay_segments_5way.py --record "$rec" --kind "$ds" \
    --arms calib \
    --out-dir "$SEG" --tag _pre > "$SEG/run_${ds}_B.log" 2>&1 &
  PIDS+=($!)
  python3 scripts/replay_segments_5way.py --record "$rec" --kind "$ds" \
    --arms oracle \
    --out-dir "$SEG" --tag _pre > "$SEG/run_${ds}_C.log" 2>&1 &
  PIDS+=($!)
done

log "waiting for ${#PIDS[@]} procs"
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
if [ $RC -ne 0 ]; then
  log "SEGMENT REPLAYS FAILED (see $SEG/run_*.log)"
  touch "$SEG/FAILED_SEG"; exit 1
fi
log "SEGMENT REPLAYS COMPLETE"
touch "$SEG/DONE_SEG"
