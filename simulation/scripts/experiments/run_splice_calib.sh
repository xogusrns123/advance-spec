#!/bin/bash
# Per-group BEST-CALIBRATED splice experiment (objective held = target_p for scale
# comparability; method chosen per group by lowest test ECE):
#   eagle group  <- target_p histogram (ECE 0.002; non-parametric, no high-p over-extrapolation)
#   suffix group <- target_p logistic  (ECE 0.008; tied-lowest, and the uniform MAT winner)
# Spliced into one calib_pp map and served on the SAME T_tp trajectory via
# --replay-existing (deterministic), so MAT is directly comparable to the uniform
# target_p arms already measured on T_tp (raw 1.384, oracle 1.823, tp-hist 1.379,
# tp-logistic 1.454). Arms: select1 (raw'), select1_calib_logistic (=spliced),
# select1_oracle (oracle'). Waits for GPU0. Run inside sglang-bench as root.
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for c in /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda; do
  [ -x "$c/bin/nvcc" ] && export CUDA_HOME="$c" && break
done
export PATH="$CUDA_HOME/bin:$PATH"
BASE=simulation/results/o4_perdepth
SRC="$BASE/qwen3_14b_tp"           # T_tp trajectory + target_p maps
DST="$BASE/xtraj_splice_on_tp"
COMMON="--workload bfcl_v4 --include-category web_search --steps 16 --tp-size 1"
EXTRA="--extra-args --attention-backend triton --speculative-draft-attention-backend triton --sampling-backend pytorch"
PORT=30021; TAIL=64

echo "=== waiting for GPU0 (<15GB) ==="
while true; do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ');
  echo "  GPU0=${U}MiB $(date +%H:%M:%S)"; [ "${U:-99999}" -lt 15000 ] && break; sleep 30; done
sleep 15

mkdir -p "$DST"
cp -f "$SRC/agent_results_record.json" "$DST/"
cp -f "$SRC/gt_tokens.jsonl" "$DST/"
# splice: eagle <- tp histogram, suffix <- tp logistic  -> calib_pp_logistic.json slot
python3 - "$SRC" "$DST" << 'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
H = json.load(open(f"{src}/calib_pp_histogram.json"))
L = json.load(open(f"{src}/calib_pp_logistic.json"))
out = {"meta": {**L.get("meta", {}), "label": "target_p",
                "splice": "eagle<-histogram, suffix<-logistic (per-group best ECE)"},
       "groups": {"eagle": H["groups"]["eagle"], "suffix": L["groups"]["suffix"]}}
json.dump(out, open(f"{dst}/calib_pp_logistic.json", "w"))
print(f"spliced -> {dst}/calib_pp_logistic.json  "
      f"(eagle depths={len(out['groups']['eagle'])} from histogram, "
      f"suffix depths={len(out['groups']['suffix'])} from logistic)")
PYEOF

echo "=== serve spliced arm on T_tp (replay-existing) ==="
python3 simulation/scripts/measure_chain_hybrid.py --preset qwen3_14b $COMMON \
  --train-n-tasks 30 --n-tasks 20 --port "$PORT" --tail-max-tokens "$TAIL" \
  --replay-existing --skip-calib-fit \
  --arms select1,select1_calib_logistic,select1_oracle \
  --output "$DST/run.json" $EXTRA
echo "SPLICE_RC=$?"
echo "SPLICE_DONE"
