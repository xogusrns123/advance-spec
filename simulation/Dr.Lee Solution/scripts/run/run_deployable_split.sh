#!/usr/bin/env bash
# ============================================================================
# DEPLOYABLE-SPLIT replays (2026-07-13): the in-sample calibration protocol is
# not deployable, so every fitted quantity — compose calibrators AND the
# SD-paper hybrid's tau* — is fit on the CALIBRATE half (even within-label conv
# ranks) and every arm's MAT is measured on the TEST half (odd ranks), disjoint
# (replay_extension --three-way --group-mode convlabel). CPU-only; safe to run
# while the GPU0 collection is up.
#
#   docker exec -d sglang-bench bash "/workspace/simulation/Dr.Lee Solution/scripts/run/run_deployable_split.sh"
#
# Outputs
#   replay logs   readable_outputs/figures/replay_logs/mat_{ds}_4way_*_split.replay.txt
#   sweeps+segs   /workspace/simulation/results/pipeline_deployable/segments/
#   status        results/pipeline_deployable/{driver.log,DONE,FAILED}
#
# NOTE: PYTHONPATH prepends /workspace/tmp/clean_pkgs — the SWE-bench Lite
# collection's in-container `pip install -e .` replaced sympy/matplotlib with
# task-repo editables (breaking torch import); the clean copies shadow them
# without touching the running collection's site-packages.
# ============================================================================
set -u
cd "/workspace/simulation/Dr.Lee Solution"
export PYTHONPATH=/workspace/tmp/clean_pkgs:/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""

RLOG=readable_outputs/figures/replay_logs
ST=results/pipeline_deployable
SEGD=/workspace/simulation/results/pipeline_deployable/segments
GM=convlabel
mkdir -p "$ST" "$SEGD"
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

# ---- wave A: base arms + fallback tau sweep (calib/test halves) --------------
log "wave A: singles/calib/oracle/fallback-sweep, three-way($GM)"
PIDS=()
for ds in $DSS; do
  rec=${REC[$ds]}
  [ -f "$rec" ] || fail "record $rec missing"
  python3 scripts/replay_extension.py --record "$rec" --max-rounds 4096 \
    --props dflash suffix --three-way --group-mode $GM \
    > "$RLOG/mat_${ds}_4way_singles_split.replay.txt" 2> "$ST/re_${ds}_singles.err" &
  PIDS+=($!)
  python3 scripts/replay_extension.py --record "$rec" --max-rounds 4096 \
    --props calib --three-way --group-mode $GM --hazard-fit beta \
    > "$RLOG/mat_${ds}_4way_calib_split.replay.txt" 2> "$ST/re_${ds}_calib.err" &
  PIDS+=($!)
  python3 scripts/replay_extension.py --record "$rec" --max-rounds 4096 \
    --props oracle --three-way --group-mode $GM \
    > "$RLOG/mat_${ds}_4way_oracle_split.replay.txt" 2> "$ST/re_${ds}_oracle.err" &
  PIDS+=($!)
  python3 scripts/replay_fallback_sweep.py --records "$ds=$rec" \
    --taus 0.5 1 2 4 8 16 32 --max-rounds 4096 --three-way --group-mode $GM \
    --out "$SEGD/fallback_sweep_fresh_${ds}.json" \
    > "$ST/sweep_${ds}.log" 2>&1 &
  PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
[ $RC -ne 0 ] && fail "wave A: a replay failed (see $ST/re_*.err / sweep_*.log)"
log "wave A done"

# deployable tau* per workload: argmax K on the CALIB half of the split sweep
python3 - "$SEGD" "$ST/taus_split.json" <<'EOF' || fail "tau* extraction"
import json, sys
from pathlib import Path
segd, out = Path(sys.argv[1]), sys.argv[2]
taus = {}
for fp in sorted(segd.glob("fallback_sweep_fresh_*.json")):
    ds = fp.stem.replace("fallback_sweep_fresh_", "")
    cal = next(iter(json.load(open(fp)).values()))["calib"]
    taus[ds] = float(max(cal, key=lambda t: cal[t]["K"]))
json.dump(taus, open(out, "w"), indent=1)
print("tau* (calib-picked):", taus)
EOF
log "tau* extracted: $(cat "$ST/taus_split.json" | tr -d '\n ')"

# ---- wave C (background): per-segment rounds, 5 arms, test half --------------
seg_all() {
  local P=()
  for ds in $DSS; do
    local tau
    tau=$(python3 -c "import json;print(json.load(open('$ST/taus_split.json'))['$ds'])")
    python3 scripts/replay_segments_5way.py --record "${REC[$ds]}" --kind "$ds" \
      --tau "$tau" --max-rounds 4096 --three-way --group-mode $GM \
      --tag _split --out-dir "$SEGD" > "$ST/seg_${ds}.log" 2>&1 &
    P+=($!)
  done
  local rc=0
  for p in "${P[@]}"; do wait "$p" || rc=1; done
  return $rc
}
seg_all & SEG_PID=$!
log "wave C launched in background (segments, 5 arms x 5 workloads)"

# ---- wave B: compose calibration-variant ladder (8 tags) ---------------------
declare -A FLAG=(
  [raw]="--raw-cals"
  [affine]="--affine-cals"
  [betahead]="--hazard-fit beta --beta-head-only"
  [isotail]="--hazard-fit beta --iso-tail-only"
  [logiso]="--head-cal logistic --tail-cal isotonic"
  [H-linear]="--head-cal linear --tail-cal raw"
  [H-logistic]="--head-cal logistic --tail-cal raw"
  [T-linear]="--head-cal raw --tail-cal linear")
for tag in raw affine betahead isotail logiso H-linear H-logistic T-linear; do
  PIDS=()
  for ds in $DSS; do
    python3 scripts/replay_extension.py --record "${REC[$ds]}" --max-rounds 4096 \
      --props calib --three-way --group-mode $GM ${FLAG[$tag]} \
      > "$RLOG/mat_${ds}_4way_calib_${tag}_split.replay.txt" \
      2> "$ST/re_${ds}_${tag}.err" & PIDS+=($!)
  done
  RC=0
  for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
  [ $RC -ne 0 ] && fail "wave B[$tag]: a replay failed"
  log "wave B[$tag] done"
done
log "wave B done"

wait "$SEG_PID" || fail "wave C: a segment replay failed (see $ST/seg_*.log)"
log "wave C done"

log "ALL DEPLOYABLE-SPLIT REPLAYS DONE"
touch "$ST/DONE"
