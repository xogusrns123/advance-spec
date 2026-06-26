# DFlash / CUDA-13 stack.
#
# DFlash speculative decoding needs sglang>=0.5.11, which hard-pins torch==2.11
# (cu130). Assembling that in-place on top of the old cu128 venv is ABI hell
# (kernels / sgl-kernel / deep_gemm / torch c10 symbol mismatch), so we base on
# the OFFICIAL sglang cu130 image where torch + sglang + sgl-kernel + deep_gemm
# are pre-built and ABI-matched for Blackwell sm_120.
#
# 0.5.12 (not 0.5.13) avoids the DFlash accept_length anomaly (sglang#27924).
FROM lmsysorg/sglang:v0.5.12-cu130

# Project deps for ALL experiments (latency / DFlash + chain-hybrid + calibration
# + bfcl_v4 agent). numpy/openai/datasets/tqdm/yaml are already in the base image.
# NOTE: bfcl-eval pulls qwen-agent which pins numpy<2, so the base image's
# numpy 2.x is downgraded to 1.26.x — harmless for these workloads (timing is
# torch/cuda; offline analysis uses numpy/sklearn either way).
RUN python3 -m pip install --no-cache-dir \
    requests matplotlib psutil pyyaml tqdm pytest ruff \
    scikit-learn ddgs langchain langchain-openai bfcl-eval ray

# Vendored ArcticInference (patched: reversible undo + raw counts) for
# SuffixDecoding / chain-hybrid. Builds the C++ `_C` extension (cmake) against
# this image's torch + python. Patch is tracked under vendor/patches/.
RUN git clone https://github.com/snowflakedb/ArcticInference.git /opt/ArcticInference \
 && cd /opt/ArcticInference \
 && git checkout fba641f8ffbaa25f6715140f4dc85692d6cf7465
COPY vendor/patches/ArcticInference.patch /tmp/ArcticInference.patch
RUN cd /opt/ArcticInference \
 && git apply /tmp/ArcticInference.patch \
 && python3 -m pip install . --no-deps \
 && cd / && rm -rf /opt/ArcticInference /tmp/ArcticInference.patch

WORKDIR /workspace

# Project source is bind-mounted at /workspace (see docker-compose.yml). sglang
# source patches are applied at RUNTIME (they live in /sgl-workspace, reset on
# rebuild), by the experiment scripts:
#   - oracle latency / chain-hybrid hooks: `python -m simulation.oracle.install_hook`
#     (called automatically by measure_*.py before each server boot)
#   - STANDALONE draft arch fix (Qwen3.5): `python simulation/oracle/patch_sglang_standalone.py`
# Keep the container alive for `docker exec`.
ENTRYPOINT []
CMD ["sleep", "infinity"]
