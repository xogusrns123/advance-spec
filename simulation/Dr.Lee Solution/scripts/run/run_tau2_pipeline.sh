#!/usr/bin/env bash
# tau2-bench leg: wait for the GPU0 capture to finish, then CPU replays (4-way +
# fallback sweep + 5-arm segments) and regenerate ALL figures with tau2 added as
# the 5th workload. Only the AGENT (assistant) generations are captured/replayed
# (user turns are LLM-simulated and out of scope; excluded at conv_map time).
#
#   docker exec -d sglang-bench bash -c \
#     'nohup bash "/workspace/simulation/Dr.Lee Solution/scripts/run_tau2_pipeline.sh" \
#      > /workspace/simulation/results/pipeline_4way/driver_tau2.out 2>&1'
#
# Markers: pipeline_4way/DONE_TAU2 / FAILED_TAU2.
set -uo pipefail
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace

R=/workspace/simulation/results
ST=$R/pipeline_4way
SEG=$ST/segments
FIG=readable_outputs/figures
RLOG=$FIG/replay_logs
PP=results/perpos_tau2_full
REC=$PP/tau2_4way.jsonl
TR=$PP/tau2_4way.traces.json
mkdir -p "$SEG" "$RLOG"
rm -f "$ST/DONE_TAU2" "$ST/FAILED_TAU2"

log() { echo "[$(TZ=Asia/Seoul date +%F' '%T)KST] $*" | tee -a "$ST/tau2.log"; }
fail() { log "FAILED: $*"; echo "$*" > "$ST/FAILED_TAU2"; exit 1; }

# ---- stage W: wait for the capture process to exit -------------------------
log "waiting for tau2 capture (capture_traj.py .. tau2_4way)"
t0=$SECONDS
while pgrep -f "capture_traj.py.*tau2_4way" >/dev/null 2>&1; do
  [ $((SECONDS - t0)) -gt $((6 * 3600)) ] && fail "stage W: 6h timeout"
  sleep 60
done
sleep 5
[ -f "$TR" ] || fail "traces.json missing after capture"
grep -q "STOPPED EARLY" "$PP/capture_0705.log" 2>/dev/null \
  && log "NOTE: capture stopped early (partial)" || true
log "capture done ($(wc -l < "$REC") records / $(python3 -c "import json;print(len(json.load(open('$TR'))['eval_traces']))") eval calls)"

# ---- stage 1: 4-way replays ------------------------------------------------
PIDS=()
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 \
  --props dflash suffix \
  > "$RLOG/mat_tau2_4way_singles.replay.txt" 2> "$ST/replay_tau2_singles.err" & PIDS+=($!)
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 \
  --props calib --calib-insample --hazard-fit beta \
  > "$RLOG/mat_tau2_4way_calib.replay.txt" 2> "$ST/replay_tau2_calib.err" & PIDS+=($!)
python3 scripts/replay_extension.py --record "$REC" --max-rounds 4096 \
  --props oracle \
  > "$RLOG/mat_tau2_4way_oracle.replay.txt" 2> "$ST/replay_tau2_oracle.err" & PIDS+=($!)
RC=0; for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
[ $RC -ne 0 ] && fail "stage 1: 4-way replay (see $ST/replay_tau2_*.err)"
log "stage 1: 4-way replays done"

# ---- stage 2: fallback sweep -> best tau -----------------------------------
python3 scripts/replay_fallback_sweep.py --records "tau2=$REC" \
  --taus 0.5 1 2 4 8 16 32 --out "$SEG/fallback_sweep_fresh_tau2.json" \
  > "$SEG/sweep_fresh_tau2.log" 2>&1 || fail "stage 2: fallback sweep"
TAU=$(python3 -c "import json;d=next(iter(json.load(open('$SEG/fallback_sweep_fresh_tau2.json')).values()));print(max(d.items(),key=lambda kv:kv[1]['K'])[0])")
log "stage 2: tau2 best tau=$TAU"

# ---- stage 3: 5-arm segment replays ----------------------------------------
PIDS=()
python3 scripts/replay_segments_5way.py --record "$REC" --kind tau2 \
  --arms dflash suffix fallback --tau "$TAU" --out-dir "$SEG" \
  > "$SEG/run_fresh_tau2_A.log" 2>&1 & PIDS+=($!)
python3 scripts/replay_segments_5way.py --record "$REC" --kind tau2 \
  --arms calib --out-dir "$SEG" > "$SEG/run_fresh_tau2_B.log" 2>&1 & PIDS+=($!)
python3 scripts/replay_segments_5way.py --record "$REC" --kind tau2 \
  --arms oracle --out-dir "$SEG" > "$SEG/run_fresh_tau2_C.log" 2>&1 & PIDS+=($!)
RC=0; for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
[ $RC -ne 0 ] && fail "stage 3: segment replay (see $SEG/run_fresh_tau2_*.log)"
log "stage 3: segment replays done"

# ---- stage 4: merge taus + regenerate ALL figures --------------------------
python3 - "$SEG" <<'EOF'
import json, sys
seg = sys.argv[1]; taus = {}
for ds in ("spider", "specbench", "swebench", "bfcl", "tau2"):
    try:
        d = next(iter(json.load(open(f"{seg}/fallback_sweep_fresh_{ds}.json")).values()))
        taus[ds] = float(max(d.items(), key=lambda kv: kv[1]["K"])[0])
    except FileNotFoundError:
        pass
json.dump(taus, open(f"{seg}/taus_fresh.json", "w")); print("taus:", taus)
EOF
python3 scripts/plot_mat_4way.py >> "$ST/tau2.log" 2>&1 || fail "stage 4: plot_mat_4way"
python3 scripts/plot_mat_subtask.py >> "$ST/tau2.log" 2>&1 || fail "stage 4: plot_mat_subtask"
python3 scripts/plot_mat_segments.py --taus-json "$SEG/taus_fresh.json" \
  >> "$ST/tau2.log" 2>&1 || fail "stage 4: plot_mat_segments"
log "TAU2 PIPELINE COMPLETE (tau2 joined all figures)"
touch "$ST/DONE_TAU2"
