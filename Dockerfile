ARG BASE_IMAGE=davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.2@sha256:994fb9dfa20ea37544fb5454d076a25edb3947a6553d55c13c2ddea60adbd18d
FROM ${BASE_IMAGE}

ARG VERSION=1.3
LABEL org.opencontainers.image.title="vllm-glm52" \
      org.opencontainers.image.description="GLM-5.2 TP4+DCP4 serving image with fast647 MLA workspace reuse and guarded B12X DCP A2A" \
      org.opencontainers.image.source="https://github.com/davidsyoung/vllm-glm52" \
      org.opencontainers.image.version="${VERSION}"

ENV B12X_INDEXER_TWO_LEVEL_FOLD=0 \
    B12X_MLA_DCP_GATHER_IN_WORKSPACE=1 \
    VLLM_USE_B12X_DCP_A2A=1 \
    VLLM_DCP_A2A_MAX_TOKENS=16 \
    VLLM_DCP_A2A_LARGE_BACKEND=ag_rs

COPY overlays/vllm/model_executor/layers/attention/mla_attention.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/mla_attention.py
COPY overlays/vllm/v1/attention/backends/mla/b12x_mla_sparse.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
COPY overlays/vllm/v1/attention/ops/common.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/common.py
COPY overlays/vllm/distributed/parallel_state.py /opt/venv/lib/python3.12/site-packages/vllm/distributed/parallel_state.py
COPY overlays/vllm/distributed/device_communicators/cuda_communicator.py /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/cuda_communicator.py
COPY overlays/b12x/attention/indexer/paged.py /opt/venv/lib/python3.12/site-packages/b12x/attention/indexer/paged.py

RUN python3 -m py_compile \
    /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/mla_attention.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/common.py \
    /opt/venv/lib/python3.12/site-packages/vllm/distributed/parallel_state.py \
    /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/cuda_communicator.py \
    /opt/venv/lib/python3.12/site-packages/b12x/attention/indexer/paged.py
