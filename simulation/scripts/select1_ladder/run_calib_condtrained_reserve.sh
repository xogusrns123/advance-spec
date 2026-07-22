#!/bin/bash
# Re-serve select1_calib_logistic with the CORRECT objective (token==gt + accept-conditioned =
# "cond-trained"), replacing the bad default 'survival'-label calib that over-picked suffix and
# made calib look like it "hurts". Same method (logistic) as the originally-reported numbers, so
# apples-to-apples. Live realized, --skip-calib-fit, output to run_calib_ct.json (preserve originals).
set -uo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0
for _c in /usr/local/cuda-12.8 /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }; done
export PATH="$CUDA_HOME/bin:$PATH"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SGLANG_CHAIN_HYBRID_DFLASH_AUX SGLANG_CHAIN_HYBRID_SEL3 || true
N="${N:-80}"
RB="simulation/results/chain_hybrid_perdepth"

serve() {  # $1=dir $2=preset $3=extra...
  local DIR="$RB/$1"; local PRESET="$2"; shift 2
  echo "============ RE-SERVE cond-trained calib_logistic: $1 ($PRESET) ============"
  cp "$DIR/calib_cond-trained/calib_pp_logistic.json" "$DIR/calib_pp_logistic.json"
  python3 simulation/scripts/measure_chain_hybrid.py --preset "$PRESET" --workload gsm8k_humaneval \
    --steps 16 --tp-size 1 --n-tasks "$N" --tail-max-tokens 0 --skip-calib-fit \
    --arms select1_calib_logistic --port 31082 --mem-fraction-static 0.65 \
    --output "$DIR/run_calib_ct.json" "$@"
  echo "$1 RC=$?"
}

serve gsm8k_humaneval_2way     qwen35_27b_mtp
serve gsm8k_humaneval_14b_2way qwen3_14b --extra-args --attention-backend triton --sampling-backend pytorch
echo "CALIB_CT_DONE"
