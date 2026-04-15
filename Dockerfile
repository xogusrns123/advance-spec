FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# System dependencies + Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        build-essential git curl libnuma-dev \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Layer 1: PyTorch (CUDA 12.2)
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu122

# Layer 2: SGLang + sgl-kernel (SM89/RTX 4090 호환)
RUN . /opt/venv/bin/activate && \
    uv pip install "sglang[all]" && \
    uv pip install --force-reinstall "sgl-kernel>=0.1.0"

# Layer 3: transformers 5.x (glm4_moe_lite 아키텍처 지원)
RUN . /opt/venv/bin/activate && \
    uv pip install "transformers>=5.0.0"

# Layer 4: 프로젝트 의존성 + 유틸리티
COPY pyproject.toml uv.lock ./
RUN . /opt/venv/bin/activate && \
    uv pip install numpy requests pyyaml matplotlib datasets pytest ruff \
        arctic-inference ray openai bfcl-eval tqdm \
        langchain langchain-openai psutil ddgs

# Layer 5: SGLang 패치
# 5a. Glm4MoeLiteModel enable_a2a_moe 버그
RUN sed -i 's/if self.enable_a2a_moe and i > self.first_k_dense_replace:/if getattr(self, "enable_a2a_moe", False) and i > self.first_k_dense_replace:/' \
    /opt/venv/lib/python3.11/site-packages/sglang/srt/models/deepseek_v2.py
# 5b. Oracle vanilla hook (SGLANG_ORACLE_VANILLA=1일 때만 활성화)
RUN EAGLE_PY=/opt/venv/lib/python3.11/site-packages/sglang/srt/speculative/eagle_worker.py && \
    SENTINEL="self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)" && \
    if ! grep -q "oracle_patch" "$EAGLE_PY"; then \
        sed -i "s|$SENTINEL|$SENTINEL\n\n        # Oracle vanilla patch: log draft tokens per step\n        import os as _os\n        if _os.environ.get('SGLANG_ORACLE_VANILLA', '0') == '1':\n            from hybrid_spec_decoding.sglang_integration.oracle_patch import patch_eagle_worker_full\n            patch_eagle_worker_full(self)|" "$EAGLE_PY"; \
    fi

# Layer 6: 소스코드
COPY . .
RUN . /opt/venv/bin/activate && \
    uv pip install --no-deps -e ".[dev]"

ENV PATH="/opt/venv/bin:$PATH" \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

CMD ["bash"]
