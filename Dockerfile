ARG BASE_IMAGE=davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.3@sha256:99ae7b28bb7069b9f7a96f75ea815be56266d2cccf7808d4c497340bb8658bd5
FROM ${BASE_IMAGE}

ARG VERSION=1.4
LABEL org.opencontainers.image.title="vllm-glm52" \
      org.opencontainers.image.description="GLM-5.2 TP4+DCP4 serving with exact-shape W4A16 Grid188 decode and MTP/DCP synchronization fixes" \
      org.opencontainers.image.source="https://github.com/davidsyoung/vllm-glm52" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.base.name="davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.3" \
      org.opencontainers.image.base.digest="sha256:99ae7b28bb7069b9f7a96f75ea815be56266d2cccf7808d4c497340bb8658bd5"

ENV HYBRID_TC_DECODE=1 \
    HYBRID_NF3_TC_DECODE=1 \
    HYBRID_HETERO_DECODE=1 \
    B12X_EMPTY_CACHE_AFTER_WARMUP=1 \
    VLLM_USE_AOT_COMPILE=1 \
    VLLM_DISABLE_COMPILE_CACHE=1

COPY overlays/hybrid_loader.py /opt/venv/lib/python3.12/site-packages/hybrid_loader.py
COPY overlays/b12x/gemm/block_fp8_linear.py /opt/venv/lib/python3.12/site-packages/b12x/gemm/block_fp8_linear.py
COPY overlays/b12x/moe/fused/w4a16/kernel.py /opt/venv/lib/python3.12/site-packages/b12x/moe/fused/w4a16/kernel.py
COPY overlays/b12x/moe/fused/w4a16/route_pack.py /opt/venv/lib/python3.12/site-packages/b12x/moe/fused/w4a16/route_pack.py
COPY overlays/vllm/model_executor/layers/attention/mla_attention.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/mla_attention.py
COPY overlays/vllm/v1/attention/backends/utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/utils.py
COPY overlays/vllm/v1/sample/ops/topk_topp_sampler.py /opt/venv/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_sampler.py
COPY overlays/vllm/v1/worker/gpu_worker.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py
COPY overlays/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py

RUN python3 -m py_compile \
    /opt/venv/lib/python3.12/site-packages/hybrid_loader.py \
    /opt/venv/lib/python3.12/site-packages/b12x/gemm/block_fp8_linear.py \
    /opt/venv/lib/python3.12/site-packages/b12x/moe/fused/w4a16/kernel.py \
    /opt/venv/lib/python3.12/site-packages/b12x/moe/fused/w4a16/route_pack.py \
    /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/mla_attention.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/utils.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_sampler.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
