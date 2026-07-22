#!/usr/bin/env bash
# FRESH-record segment replays (2026-07-05 recaptures), fully autonomous:
#   stage 1: per-workload fallback tau sweeps (parallel) on the fresh records
#   stage 2: segment replays (3 proc-groups per workload) at the fresh best tau
#   stage 3: wait for the bfcl fresh capture (separate GPU driver), then its
#            sweep + segment replays
#   stage 4: regenerate all four MAT_segments_{ds}.png from the fresh logs
#
#   docker exec -d sglang-bench bash -c \
#     'nohup bash "/workspace/simulation/Dr.Lee Solution/scripts/run_segments_5way_fresh.sh" \
#      > /workspace/simulation/results/pipeline_4way/segments/driver_seg_fresh.out 2>&1'
#
# Markers: segments/DONE_SEG_FRESH / FAILED_SEG_FRESH. Fresh logs are untagged
# (seg_{ds}_{arm}.jsonl) so the *_pre logs from the Jul-3 snapshot stay intact.
set -uo pipefail
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace

SEG=/workspace/simulation/results/pipeline_4way/segments
mkdir -p "$SEG"
rm -f "$SEG/DONE_SEG_FRESH" "$SEG/FAILED_SEG_FRESH"

declare -A REC=(
  [spider]=results/perpos_spider/spider_4way.jsonl
  [specbench]=results/perpos_specbench_full/specbench_4way.jsonl
  [swebench]=results/perpos_swebench/swebench_4way.jsonl
  [bfcl]=results/perpos_bfcl_full/bfcl_4way.jsonl
)

log() { echo "[$(date -u +%F' '%T)UTC] $*" | tee -a "$SEG/seg_fresh.log"; }
fail() { log "FAILED: $*"; touch "$SEG/FAILED_SEG_FRESH"; exit 1; }

best_tau() {  # $1 = per-ds sweep json (single-key)
  python3 -c "
import json, sys
d = next(iter(json.load(open('$1')).values()))
print(max(d.items(), key=lambda kv: kv[1]['K'])[0])"
}

run_ds() {    # $1 = ds; sweep -> tau -> 3 replay groups (waits for them)
  local ds=$1 rec=${REC[$1]} tau pids=()
  log "[$ds] sweep start"
  python3 scripts/replay_fallback_sweep.py --records "$ds=$rec" \
    --taus 0.5 1 2 4 8 16 32 --out "$SEG/fallback_sweep_fresh_$ds.json" \
    > "$SEG/sweep_fresh_$ds.log" 2>&1 || { log "[$ds] sweep FAILED"; return 1; }
  tau=$(best_tau "$SEG/fallback_sweep_fresh_$ds.json") || return 1
  log "[$ds] best tau = $tau; segment replays start"
  python3 scripts/replay_segments_5way.py --record "$rec" --kind "$ds" \
    --arms dflash suffix fallback --tau "$tau" \
    --out-dir "$SEG" > "$SEG/run_fresh_${ds}_A.log" 2>&1 &
  pids+=($!)
  python3 scripts/replay_segments_5way.py --record "$rec" --kind "$ds" \
    --arms calib --out-dir "$SEG" > "$SEG/run_fresh_${ds}_B.log" 2>&1 &
  pids+=($!)
  python3 scripts/replay_segments_5way.py --record "$rec" --kind "$ds" \
    --arms oracle --out-dir "$SEG" > "$SEG/run_fresh_${ds}_C.log" 2>&1 &
  pids+=($!)
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] && log "[$ds] segment replays done" || log "[$ds] replays FAILED"
  return $rc
}

# ---- stages 1+2: the three already-captured workloads, in parallel ----------
PIDS=(); FAILED=0
for ds in spider specbench swebench; do
  [ -f "${REC[$ds]}" ] || { log "SKIP $ds: record missing"; continue; }
  run_ds "$ds" &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p" || FAILED=1; done
[ $FAILED -ne 0 ] && fail "one of spider/specbench/swebench legs (see $SEG/*fresh*.log)"

# ---- stage 3: bfcl — wait for its fresh capture, then the same leg ----------
BREC=${REC[bfcl]}
BTR=${BREC%.jsonl}.traces.json
log "waiting for bfcl fresh capture ($BREC + traces)"
t0=$SECONDS
while ! { [ -f "$BREC" ] && [ -f "$BTR" ] && [ "$BTR" -nt "$BREC" ]; }; do
  [ $((SECONDS - t0)) -gt $((8 * 3600)) ] && fail "8h timeout waiting for bfcl capture"
  sleep 120
done
log "bfcl capture detected after $(((SECONDS-t0)/60))min"
run_ds bfcl || fail "bfcl leg (see $SEG/*fresh*bfcl*.log)"

# ---- stage 4: taus json + figures -------------------------------------------
python3 - "$SEG" <<'EOF'
import json, sys
seg = sys.argv[1]
taus = {}
for ds in ("spider", "specbench", "swebench", "bfcl"):
    try:
        d = next(iter(json.load(open(f"{seg}/fallback_sweep_fresh_{ds}.json")).values()))
        taus[ds] = float(max(d.items(), key=lambda kv: kv[1]["K"])[0])
    except FileNotFoundError:
        pass
json.dump(taus, open(f"{seg}/taus_fresh.json", "w"))
print("taus:", taus)
EOF
python3 scripts/plot_mat_segments.py \
  --note "offline replay · records: 2026-07-05 captures" \
  --taus-json "$SEG/taus_fresh.json" >> "$SEG/seg_fresh.log" 2>&1 \
  || fail "plot_mat_segments.py"
log "SEGMENT FRESH PIPELINE COMPLETE"
touch "$SEG/DONE_SEG_FRESH"
