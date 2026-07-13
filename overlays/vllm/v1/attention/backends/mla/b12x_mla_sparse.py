# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x sparse-MLA backend for SM120 / SM121 (consumer Blackwell).

Counterpart to ``SparseMLASm120Backend`` (FlashInfer V32 v2). Same envelope --
``fp8_ds_mla`` KV cache (656 B/token), head_size = 576, paged block_size = 64,
V32-family models with an ``index_topk`` config (DeepSeek V3.2, GLM-5.1, Kimi
K2.5) -- but the decode/extend kernels come from b12x's unified SM120 backend
via the ``b12x.integration.mla`` front door (``sparse_mla_decode_forward`` /
``sparse_mla_extend_forward``). On SM120+ CUDA those front-door functions route
to ``b12x/attention/mla/unified_sm120`` automatically (GLM_NSA q_head_dim==576
contract). Selecting this backend also selects b12x's sparse indexer/top-k path.

Scratch philosophy (eager PLAN -> BIND -> KERNEL; no workspace/arena, ever):
b12x workspaces/arenas are sglang-only and forbidden here. We build a caller-
owned-scratch ``plan_sparse_mla_scratch`` PLAN once per mode (decode / extend),
and each forward maps a vLLM ``current_workspace_manager()`` scratch tensor into
a plain ``B12XSparseMLAScratch`` views CONTAINER via ``plan.bind(...)`` -- a pure
narrow()+view() mapping that allocates nothing and constructs no workspace. The
binding holds views (never a ``B12XAttentionWorkspace``); the unified SM120
sparse-MLA decode/extend kernels duck-type the container's
``tmp_output`` / ``tmp_lse`` / ``output_buffer`` / ``final_lse`` /
``num_chunks_ptr`` / ``set_split_chunk_config`` fields, so the binding is a
drop-in with no kernel-signature change. q-concat and the scratch are borrowed in
ONE ``get_simultaneous`` call so they never alias.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import get_mla_dims
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_dcp_global_index_to_local_index,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# Split-K tile width. Mirrors SparseMLASm120's _DECODE_SPLIT_TILE: the number of
# split-K chunks is ceil(topk / tile). This bounds the chunk dim of the borrowed
# mid_out/mid_lse scratch and the workspace ``max_chunks_per_row`` cap; b12x's
# wave-balanced planner picks num_splits <= this cap.
_DECODE_SPLIT_TILE = 64
_HEAD_ALIGNMENT = 8
_BF16_BYTES = 2
_EXTEND_PREWARM_DONE: set[tuple[int | None, int, int, int, int, int, bool]] = set()


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d", name, value, default)
        return default
    return parsed


@triton.jit
def _mask_page_table_after_nsa_len_kernel(
    page_table_ptr,
    nsa_len_ptr,
    page_stride0,
    page_stride1,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs < width
    nsa_len = tl.load(nsa_len_ptr + row)
    tl.store(
        page_table_ptr + row * page_stride0 + offs * page_stride1,
        -1,
        mask=valid & (offs >= nsa_len),
    )


def _mask_page_table_after_nsa_len(
    page_table: torch.Tensor,
    nsa_cache_seqlens: torch.Tensor,
) -> None:
    width = page_table.shape[1]
    if width == 0 or page_table.shape[0] == 0:
        return
    block_n = 128
    _mask_page_table_after_nsa_len_kernel[
        (page_table.shape[0], triton.cdiv(width, block_n))
    ](
        page_table,
        nsa_cache_seqlens,
        page_table.stride(0),
        page_table.stride(1),
        width,
        BLOCK_N=block_n,
    )


class B12xMLASparseBackend(AttentionBackend):
    """b12x unified sparse-MLA backend (SM120 / SM121).

    Same envelope as ``SparseMLASm120Backend`` (head 576, fp8_ds_mla, block 64,
    index_topk) but driven by b12x's unified decode/extend kernels.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
        "nf3_ds_mla",
        "nf3bf16_ds_mla",
        "fp8",  # aliases for fp8_ds_mla on this backend
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Must equal DeepseekV32IndexerBackend.get_supported_kernel_block_sizes
        # on CUDA (= [64]); the unified b12x decode/extend kernels dispatch
        # page_block_size == 64 natively (matches the fp8_ds_mla layout).
        return [64]

    @staticmethod
    def get_name() -> str:
        return "B12X_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["B12xMLASparseImpl"]:
        return B12xMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["B12xMLASparseMetadataBuilder"]:
        return B12xMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # GLM_NSA contract: q_head_dim = kv_lora_rank (512) + qk_rope_head_dim
        # (64) = 576. The unified decode raises on any other q_head_dim.
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Consumer Blackwell SM120 / SM121. The unified b12x kernels gate on
        # get_sm_version(device) >= 120 internally.
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # Require an indexer-equipped (index_topk) model, same as SPARSE_MLA_SM120.
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            if not hasattr(hf_text_config, "index_topk"):
                return "B12X_MLA_SPARSE requires a model with index_topk config"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # = 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # V32 fp8_ds_mla packed: 656 B/token (512 NoPE + 16 inline FP32
            # scales + 128 BF16 RoPE). Mirrors the FlashMLA / SPARSE_MLA_SM120
            # layout; b12x's GLM_NSA decode reads the same record.
            return (num_blocks, block_size, 656)
        if cache_dtype_str == "nvfp4_ds_mla":
            # NVFP4 MLA latent: 256 B NoPE data + 32 B E4M3 scales +
            # 64 B E4M3 RoPE + 4 B fp32 RoPE scale + 12 B pad.
            return (num_blocks, block_size, 368)
        if cache_dtype_str == "nf3_ds_mla":
            # NF3 MLA latent: 192 B NF3 3-bit NoPE + 32 B E4M3 scales +
            # 64 B E4M3 RoPE + 4 B fp32 rope scale + 12 B pad.
            return (num_blocks, block_size, 304)
        if cache_dtype_str == "nf3bf16_ds_mla":
            # NF3 diagnostic twin: 192 B NF3 NoPE + 32 B E4M3 scales +
            # 16 B pad + 128 B verbatim BF16 RoPE.
            return (num_blocks, block_size, 368)
        return (num_blocks, block_size, head_size)


@dataclass
class B12xMLASparseMetadata(AttentionMetadata):
    """Attention metadata for the B12X_MLA_SPARSE backend."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    num_decode_tokens: int
    num_prefill_tokens: int

    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    # DCP keeps global logical top-k ids until forward_mqa maps the entries
    # owned by this rank to local physical slots. These buffers are unnecessary
    # for the direct native-slot path when DCP is disabled.
    req_id_per_token: torch.Tensor | None
    page_table_1: torch.Tensor | None
    nsa_cache_seqlens: torch.Tensor | None
    # Per-request computed KV length (decode cache_seqlens_int32).
    seq_lens: torch.Tensor
    cache_seq_lens_per_req: torch.Tensor
    # Per-token causal KV length consumed directly by the sparse MLA kernel.
    # For pure decode this equals ``seq_lens`` (one token per request).
    cache_seq_lens_per_token: torch.Tensor

    block_size: int = 64
    topk_tokens: int = 2048


class B12xMLASparseMetadataBuilder(AttentionMetadataBuilder[B12xMLASparseMetadata]):
    """Builder for B12X_MLA_SPARSE attention metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_exact_metadata_reuse: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device

        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_seqs = vllm_config.scheduler_config.max_num_seqs
        # Max-batched-token scratch buffers so cudagraph capture sees stable
        # allocations (sliced per build()).
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        self.cache_seq_lens_per_req_buffer = torch.empty(
            (max_seqs,), dtype=torch.int32, device=device
        )
        if self.dcp_world_size > 1:
            self.req_id_per_token_buffer = torch.empty(
                (max_tokens,), dtype=torch.int32, device=device
            )
            self.page_table_1_buffer = torch.empty(
                (max_tokens, self.topk_tokens), dtype=torch.int32, device=device
            )
            self.nsa_cache_seqlens_buffer = torch.empty(
                (max_tokens,), dtype=torch.int32, device=device
            )
            self.req_ids_arange = torch.arange(
                max_tokens, dtype=torch.int32, device=device
            )
        else:
            self.req_id_per_token_buffer = None
            self.page_table_1_buffer = None
            self.nsa_cache_seqlens_buffer = None
            self.req_ids_arange = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12xMLASparseMetadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens
        if cm.batch_topology is not None:
            _, _, num_decode_tokens, num_prefill_tokens = (
                cm.batch_topology.split_decodes_and_prefills(
                    cm,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=True,
                )
            )
        else:
            _, _, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    cm,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=True,
                )
            )
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        use_dcp = self.dcp_world_size > 1
        seq_lens_for_req = (
            cm.dcp_local_seq_lens
            if use_dcp and cm.dcp_local_seq_lens is not None
            else cm.seq_lens
        )
        req_id_per_token_tensor = None

        # Per-token causal KV length. In pure decode the common metadata already
        # has exactly the graph-stable tensor both b12x consumers need, so bind it
        # directly instead of staging two identical D2D copies.
        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:
            if use_dcp:
                assert self.req_ids_arange is not None
                req_id_per_token_tensor = self.req_ids_arange[:num_tokens]
                self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                    seq_lens_for_req[:num_tokens], non_blocking=True
                )
                self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                    seq_lens_for_req[: cm.num_reqs], non_blocking=True
                )
                cache_seq_lens_per_token = self.cache_seq_lens_per_token_buffer[
                    :num_tokens
                ]
                cache_seq_lens_per_req = self.cache_seq_lens_per_req_buffer[
                    : cm.num_reqs
                ]
            else:
                cache_seq_lens_per_token = seq_lens_for_req[:num_tokens]
                cache_seq_lens_per_req = seq_lens_for_req[: cm.num_reqs]
        else:
            if cm.batch_topology is not None:
                starts = cm.batch_topology.query_start_loc_np[: cm.num_reqs + 1]
                query_lens = cm.batch_topology.query_lens_np
                req_id_per_token_np = cm.batch_topology.req_id_per_token_np
            else:
                starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
                query_lens = np.diff(starts)
                req_id_per_token_np = np.repeat(
                    np.arange(cm.num_reqs, dtype=np.int32), query_lens
                )
            num_query_tokens = int(starts[-1])
            if num_query_tokens > num_tokens:
                raise RuntimeError(
                    "B12X sparse MLA metadata received query_start_loc with "
                    f"{num_query_tokens} tokens, exceeding padded capacity "
                    f"{num_tokens}"
                )

            req_ids = None
            if use_dcp:
                req_ids = np.zeros((num_tokens,), dtype=np.int32)
                if num_query_tokens:
                    req_ids[:num_query_tokens] = req_id_per_token_np

            # Avoid the blocking seq_lens device->host sync. cm.seq_lens_cpu is a
            # lazy `.to("cpu")`; under --async-scheduling the runner keeps the GPU
            # tensor authoritative (_seq_lens_cpu=None), so reading it here forces a
            # full D2H copy every (MTP) decode step and serializes the pipeline that
            # async scheduling exists to overlap. The indexer that selects the
            # top-k for this same step already reads seq_lens_cpu_upper_bound; mirror
            # it. The indexer writes -1 for invalid tail entries and MLA clamps the
            # dynamic length to topk, so an optimistic (>=) bound remains safe.
            seq_lens_cpu_src = (
                cm.seq_lens_cpu_upper_bound
                if cm.seq_lens_cpu_upper_bound is not None
                else cm.seq_lens_cpu
            )
            seq_lens_cpu = seq_lens_cpu_src.numpy().astype(np.int32, copy=False)
            per_token_lens = np.zeros((num_tokens,), dtype=np.int32)
            for req_id, q_len in enumerate(query_lens):
                if q_len <= 0:
                    continue
                start = int(starts[req_id])
                end = int(starts[req_id + 1])
                context_len = int(seq_lens_cpu[req_id]) - int(q_len)
                if use_dcp:
                    global_per_token_lens = torch.arange(
                        context_len + 1,
                        context_len + int(q_len) + 1,
                        dtype=torch.int32,
                    )
                    per_token_lens[start:end] = get_dcp_local_seq_lens(
                        global_per_token_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    ).numpy()
                else:
                    per_token_lens[start:end] = np.arange(
                        context_len + 1,
                        context_len + int(q_len) + 1,
                        dtype=np.int32,
                    )

            per_token_lens_t = torch.from_numpy(per_token_lens)
            if per_token_lens_t.device.type == "cpu":
                per_token_lens_t = per_token_lens_t.pin_memory()
            if req_ids is not None:
                assert self.req_id_per_token_buffer is not None
                req_ids_t = torch.from_numpy(req_ids)
                if req_ids_t.device.type == "cpu":
                    req_ids_t = req_ids_t.pin_memory()
                self.req_id_per_token_buffer[:num_tokens].copy_(
                    req_ids_t, non_blocking=True
                )
                req_id_per_token_tensor = self.req_id_per_token_buffer[:num_tokens]
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                per_token_lens_t, non_blocking=True
            )
            self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                seq_lens_for_req[: cm.num_reqs], non_blocking=True
            )
            cache_seq_lens_per_token = self.cache_seq_lens_per_token_buffer[:num_tokens]
            cache_seq_lens_per_req = self.cache_seq_lens_per_req_buffer[: cm.num_reqs]

        return B12xMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=num_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token_tensor,
            page_table_1=(
                self.page_table_1_buffer[:num_tokens]
                if self.page_table_1_buffer is not None
                else None
            ),
            nsa_cache_seqlens=(
                self.nsa_cache_seqlens_buffer[:num_tokens]
                if self.nsa_cache_seqlens_buffer is not None
                else None
            ),
            seq_lens=cache_seq_lens_per_req,
            cache_seq_lens_per_req=cache_seq_lens_per_req,
            cache_seq_lens_per_token=cache_seq_lens_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class B12xMLASparseImpl(SparseMLAAttentionImpl[B12xMLASparseMetadata]):
    """b12x unified sparse-MLA implementation (decode + extend/prefill)."""

    can_return_lse_for_decode: bool = True
    supports_dcp_project_before_merge: bool = True
    supports_dcp_gather_query_in_workspace: bool = True
    supports_dcp_project_before_merge_in_workspace: bool = True
    supports_dcp_reduce_scatter_output_in_workspace: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "B12X_MLA_SPARSE does not support alibi_slopes / sliding_window "
                "/ logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X_MLA_SPARSE only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA dims (absorbed: Q post-projection is [T, H, kv_lora_rank + rope]).
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.v_head_dim: int = mla_args.get("v_head_dim", 512)
        # GLM_NSA contract: q_head_dim = kv_lora_rank (512) + qk_rope (64) = 576.
        self.q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.force_contiguous_mla_bmm_input = True
        self.force_contiguous_mla_bmm_weight = True
        self.force_contiguous_mla_bmm_output = True

        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )
        assert self.topk_indices_buffer is not None, (
            "B12X_MLA_SPARSE requires sparse-MLA top-k indices "
            "(model with index_topk in its config)."
        )
        self.topk_tokens = int(self.topk_indices_buffer.shape[-1])

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.dcp_workspace_non_dbo = not bool(parallel_config.enable_dbo)
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.tp_world_size = int(parallel_config.tensor_parallel_size)
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        self.need_to_return_lse_for_decode = self.dcp_world_size > 1

        expects_physical_slots = self.dcp_world_size == 1
        if (
            indexer is not None
            and bool(indexer.output_physical_slots) != expects_physical_slots
        ):
            expected = "physical" if expects_physical_slots else "logical"
            raise RuntimeError(
                f"B12X_MLA_SPARSE requires {expected} sparse-indexer output "
                f"when dcp_world_size={self.dcp_world_size}"
            )

        scheduler_config = vllm_config.scheduler_config
        self.device = torch.device(f"cuda:{torch.accelerator.current_device_index()}")
        max_batched = int(scheduler_config.max_num_batched_tokens)
        max_num_seqs = int(scheduler_config.max_num_seqs)
        self.block_size = 64
        # MLAAttention all-gathers the local query-head shard before entering a
        # DCP backend. The kernel must therefore plan for, and return, the full
        # gathered head set; the outer layer reduces/scatters it back afterward.
        self._input_num_heads = self.num_heads * self.dcp_world_size
        # kingdom(nvfp4_ds_mla): b12x ScaleFormat selects the packed-latent
        # record in the unified SM120 decode/extend kernels: NVFP4_E4M3 == 2
        # (368 B/token), NF3_E4M3 == 3 (304 B/token, e4m3 rope + fp32 rope
        # scale), NF3_BF16ROPE == 4 (368 B/token diagnostic). None keeps the
        # dtype-inferred format (ARBITRARY_FP32 for the 656 B fp8_ds_mla
        # record).
        self._b12x_scale_format = {
            "nvfp4_ds_mla": 2,
            "nf3_ds_mla": 3,
            "nf3bf16_ds_mla": 4,
        }.get(self.kv_cache_dtype)
        # kingdom(nvfp4_ds_mla): forwarded into every plan/decode/extend b12x
        # call ONLY for the packed-latent records, so fp8_ds_mla serving keeps
        # the stock b12x call signature (works on a b12x tree without the
        # packed-latent read-path port; the port is required only to serve
        # nvfp4_ds_mla / nf3_ds_mla / nf3bf16_ds_mla).
        self._b12x_nvfp4_kwargs: dict[str, Any] = (
            {}
            if self._b12x_scale_format is None
            else {"scale_format": self._b12x_scale_format}
        )

        # Split-K cap: ceil(topk / tile). Bounds the borrowed mid_out/mid_lse
        # chunk dim and the workspace max_chunks_per_row.
        self._num_splits_cap = max(1, _cdiv(self.topk_tokens, _DECODE_SPLIT_TILE))
        self._kernel_num_heads = (
            _cdiv(self._input_num_heads, _HEAD_ALIGNMENT) * _HEAD_ALIGNMENT
        )
        self._pad_heads = self._kernel_num_heads != self._input_num_heads

        self.spec_decode_max_q = _env_int("VLLM_B12X_MLA_SPEC_DECODE_MAX_Q", 8)
        # The decode kernel handles independent one-token query rows. MTP
        # verification has multiple query rows per request, and later rows must
        # attend to earlier draft rows in the same verifier batch. Route those
        # batches through the extend path unless explicitly overridden.
        self.spec_extend_as_decode = (
            os.getenv("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "0") != "0"
        )

        # Decode query rows per request (1, plus speculative draft tokens).
        q_per_req = 1
        spec = getattr(vllm_config, "speculative_config", None)
        if spec is not None and getattr(spec, "num_speculative_tokens", None):
            q_per_req = 1 + int(spec.num_speculative_tokens)
        if self.spec_extend_as_decode:
            q_per_req = max(q_per_req, self.spec_decode_max_q)
        self._decode_max_rows = min(max_num_seqs * q_per_req, max_batched)
        if self._decode_max_rows < max_num_seqs:
            self._decode_max_rows = max_num_seqs

        self._max_batched = int(max_batched)

        # Lazily import b12x only on this opt-in path.
        from b12x.integration.mla import (
            sparse_mla_decode_forward,
            sparse_mla_extend_forward,
        )
        from b12x.integration.sparse_mla_scratch import (
            B12XSparseMLAScratchCaps,
            plan_sparse_mla_scratch,
        )

        self._sparse_mla_decode_forward = sparse_mla_decode_forward
        self._sparse_mla_extend_forward = sparse_mla_extend_forward

        # Eager PLAN -> BIND -> KERNEL (no b12x workspace/arena, ever). We build a
        # caller-owned-scratch PLAN once per mode; each forward maps a vLLM
        # workspace-manager scratch tensor into a plain B12XSparseMLAScratch views
        # CONTAINER via plan.bind(). The unified SM120 sparse-MLA decode/extend
        # kernels duck-type the container's tmp_output/tmp_lse/output_buffer/
        # final_lse fields. The planner fixes the split count for each captured
        # graph and the merge specializes on that count, so no device-side control
        # scalar initialization is needed. final_lse is pre-materialized as a view
        # so the legacy lazy torch.empty(final_lse) never fires during capture.
        def _make_plan(
            mode: str, max_q_rows: int, num_q_heads: int, max_batch: int
        ) -> Any:
            # kingdom(nvfp4_ds_mla): the FP4 record needs the caps to carry the
            # cache dtype + b12x ScaleFormat so the scratch planner sizes for
            # the 432 B record; omit both for fp8_ds_mla so the caps stay
            # constructible on a stock (pre-nvfp4-port) b12x tree.
            caps_kwargs: dict[str, Any] = (
                {}
                if self._b12x_scale_format is None
                else {
                    "kv_cache_dtype": self.kv_cache_dtype,
                    "scale_format": self._b12x_scale_format,
                }
            )
            return plan_sparse_mla_scratch(
                B12XSparseMLAScratchCaps(
                    device=self.device,
                    num_q_heads=int(num_q_heads),
                    max_q_rows=int(max_q_rows),
                    max_width=self.topk_tokens,
                    dtype=torch.bfloat16,
                    kv_dtype=torch.uint8,
                    head_dim=self.q_head_dim,
                    v_head_dim=self.kv_lora_rank,
                    mode=mode,
                    max_batch=int(max_batch),
                    max_chunks_per_row=self._num_splits_cap,
                    page_size=self.block_size,
                    **caps_kwargs,
                )
            )

        self._decode_plan = _make_plan(
            "decode",
            self._decode_max_rows,
            self._kernel_num_heads,
            self._decode_max_rows,
        )
        self._extend_plan = _make_plan(
            "extend", max_batched, self._kernel_num_heads, max_num_seqs
        )
        # One caller-owned uint8 scratch tensor covers either path (the larger
        # layout); the per-mode materializer carves its views from the prefix.
        self._scratch_nbytes = max(
            int(self._decode_plan.layout.nbytes),
            int(self._extend_plan.layout.nbytes),
        )

        # Pre-touch q-concat + the attention scratch TOGETHER so the workspace
        # manager grows during warmup (before lock_workspace() runs
        # post-cudagraph-capture) and so the two always come from ONE
        # get_simultaneous call -> distinct, non-overlapping offsets. The manager
        # packs every call from offset 0, so borrowing q and the scratch the kernel
        # writes in separate calls would alias them.
        workspace_specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            (
                (max_batched, self._kernel_num_heads, self.q_head_dim),
                torch.bfloat16,
            )
        ]
        if self._pad_heads:
            workspace_specs.append(
                (
                    (max_batched, self._input_num_heads, self.kv_lora_rank),
                    torch.bfloat16,
                )
            )
        workspace_specs.append(((self._scratch_nbytes,), torch.uint8))
        self._workspace_specs = tuple(workspace_specs)
        self._borrow_workspaces()
        self._prewarm_extend_kernels_once(max_batched)

        # Q arrives BF16; the unified kernel quantizes inside.
        self.supports_quant_query_input = False

    def _borrow_workspaces(self) -> list[torch.Tensor]:
        return current_workspace_manager().get_simultaneous(*self._workspace_specs)

    def dcp_all_gather_query_in_workspace(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Gather rank-local query heads through the borrowed MLA workspaces."""
        tuple_input = isinstance(q, tuple)
        input_spans: list[tuple[str, int, int]] = []
        if tuple_input:
            if len(q) != 2:
                raise ValueError(
                    "B12X workspace DCP gather tuple input must contain exactly "
                    "(noPE, RoPE)."
                )
            ql_nope, q_pe = q
            if not isinstance(ql_nope, torch.Tensor) or not isinstance(
                q_pe, torch.Tensor
            ):
                raise TypeError(
                    "B12X workspace DCP gather tuple components must be tensors."
                )
            if ql_nope.ndim != 3 or q_pe.ndim != 3:
                raise ValueError(
                    "B12X workspace DCP gather tuple components must both be "
                    "rank-3 tensors, got "
                    f"noPE={tuple(ql_nope.shape)}, RoPE={tuple(q_pe.shape)}."
                )
            num_tokens, local_heads, nope_dim = ql_nope.shape
            pe_tokens, pe_heads, rope_dim = q_pe.shape
            if (
                nope_dim != 512
                or rope_dim != 64
                or pe_tokens != num_tokens
                or pe_heads != local_heads
            ):
                raise ValueError(
                    "B12X workspace DCP gather tuple geometry must be exactly "
                    "(noPE[T, H, 512], RoPE[T, H, 64]), got "
                    f"noPE={tuple(ql_nope.shape)}, RoPE={tuple(q_pe.shape)}."
                )
            if ql_nope.dtype != torch.bfloat16 or q_pe.dtype != torch.bfloat16:
                raise TypeError(
                    "B12X workspace DCP gather tuple components must both be "
                    f"BF16, got noPE={ql_nope.dtype}, RoPE={q_pe.dtype}."
                )
            if (
                ql_nope.device != q_pe.device
                or ql_nope.device != self.device
                or ql_nope.device.type != "cuda"
            ):
                raise ValueError(
                    "B12X workspace DCP gather tuple components must share the "
                    f"planned CUDA device {self.device}, got "
                    f"noPE={ql_nope.device}, RoPE={q_pe.device}."
                )
            for component_name, component in (
                ("noPE", ql_nope),
                ("RoPE", q_pe),
            ):
                component_strides = tuple(component.stride())
                if component_strides[-1] != 1 or any(
                    stride <= 0 for stride in component_strides
                ):
                    raise ValueError(
                        "B12X workspace DCP gather tuple components require a "
                        "unit-stride last dimension and positive strides, got "
                        f"{component_name} stride={component_strides}."
                    )
                span_begin = component.data_ptr()
                vector_alignment = 32 if component_name == "noPE" else 4
                outer_stride_bytes = tuple(
                    stride * component.element_size()
                    for stride in component_strides[:2]
                )
                if span_begin % vector_alignment != 0 or any(
                    stride_bytes % vector_alignment != 0
                    for stride_bytes in outer_stride_bytes
                ):
                    raise ValueError(
                        "B12X workspace DCP gather tuple component does not "
                        "satisfy concat_mla_q vector alignment: "
                        f"{component_name} requires {vector_alignment}-byte "
                        f"base/outer strides, got data_ptr={span_begin}, "
                        f"outer_stride_bytes={outer_stride_bytes}."
                    )
                span_elements = 1 + sum(
                    (dimension - 1) * stride
                    for dimension, stride in zip(
                        component.shape, component_strides
                    )
                )
                span_nbytes = span_elements * component.element_size()
                span_end = span_begin + span_nbytes
                if span_nbytes <= 0 or span_end <= span_begin:
                    raise ValueError(
                        "B12X workspace DCP gather tuple component has a "
                        f"non-positive live storage span: {component_name} "
                        f"span={span_nbytes} bytes."
                    )
                input_spans.append((component_name, span_begin, span_end))
            query_device = ql_nope.device
            head_dim = nope_dim + rope_dim
        else:
            if not isinstance(q, torch.Tensor):
                raise TypeError(
                    "B12X workspace DCP gather requires a tensor or exact "
                    "(noPE, RoPE) tensor tuple."
                )
            if q.ndim != 3:
                raise ValueError(
                    "B12X workspace DCP gather requires a rank-3 local query, "
                    f"got shape={tuple(q.shape)}."
                )
            num_tokens, local_heads, head_dim = q.shape

        if (
            num_tokens < 1025
            or num_tokens > min(3072, self._max_batched)
            or not self.dcp_workspace_non_dbo
        ):
            raise ValueError(
                "B12X workspace DCP gather requires non-DBO tokens in the "
                f"validated range [1025, 3072], got tokens={num_tokens}, "
                f"max_tokens={self._max_batched}, "
                f"non_dbo={self.dcp_workspace_non_dbo}."
            )
        if not tuple_input:
            if q.dtype != torch.bfloat16:
                raise TypeError(
                    "B12X workspace DCP gather requires a BF16 local query, "
                    f"got {q.dtype}."
                )
            if q.device != self.device or q.device.type != "cuda":
                raise ValueError(
                    "B12X workspace DCP gather requires the planned CUDA device "
                    f"{self.device}, got {q.device}."
                )
            if not q.is_contiguous():
                raise ValueError(
                    "B12X workspace DCP gather requires a contiguous local query, "
                    f"got stride={tuple(q.stride())}."
                )
            query_device = q.device
            span_nbytes = q.numel() * q.element_size()
            span_begin = q.data_ptr()
            span_end = span_begin + span_nbytes
            if span_nbytes <= 0 or span_end <= span_begin:
                raise ValueError(
                    "B12X workspace DCP gather local query has a non-positive "
                    f"live storage span of {span_nbytes} bytes."
                )
            input_spans.append(("query", span_begin, span_end))
        if local_heads != self.num_heads or head_dim != self.q_head_dim:
            input_shapes = (
                f"noPE={tuple(ql_nope.shape)}, RoPE={tuple(q_pe.shape)}"
                if tuple_input
                else str(tuple(q.shape))
            )
            raise ValueError(
                "B12X workspace DCP gather local query geometry mismatch: "
                f"expected (*, {self.num_heads}, {self.q_head_dim}), "
                f"got {input_shapes}."
            )
        world_size = self.dcp_world_size
        global_heads = world_size * local_heads
        if (
            self._max_batched != 3072
            or not self.dcp_workspace_non_dbo
            or world_size != 4
            or self.tp_world_size != 4
            or local_heads != 16
            or global_heads != 64
            or self.num_heads != 16
            or self._input_num_heads != 64
            or self._kernel_num_heads != 64
            or self.q_head_dim != 576
            or self.kv_lora_rank != 512
        ):
            raise RuntimeError(
                "B12X workspace DCP gather is restricted to the validated "
                "production contract: max_tokens=3072, TP/DCP=4/4, "
                "local/global/kernel heads=16/64/64, q_dim=576, "
                "scratch value_dim=512; got "
                f"max_tokens={self._max_batched}, world={world_size}, "
                f"TP={self.tp_world_size}, "
                f"local/global/kernel heads={local_heads}/{global_heads}/"
                f"{self._kernel_num_heads}, q_dim={self.q_head_dim}, "
                f"scratch value_dim={self.kv_lora_rank}."
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "B12X workspace DCP gather is only valid on the eager large-batch "
                "AG/RS path, not during CUDA graph capture."
            )

        from vllm.distributed.parallel_state import get_dcp_group

        dcp_group = get_dcp_group()
        rank = dcp_group.rank_in_group
        if dcp_group.world_size != world_size or rank != self.dcp_rank:
            raise RuntimeError(
                "B12X workspace DCP gather group does not match the sparse-MLA "
                f"plan: group=({dcp_group.world_size}, {rank}), "
                f"plan=({world_size}, {self.dcp_rank})."
            )
        process_group = getattr(dcp_group, "device_group", None)
        if process_group is None:
            raise RuntimeError(
                "B12X workspace DCP gather requires the DCP device process group."
            )
        if (
            dist.get_world_size(group=process_group) != world_size
            or dist.get_rank(group=process_group) != rank
        ):
            raise RuntimeError(
                "B12X workspace DCP gather process-group rank/world mismatch."
            )

        workspace_tensors = self._borrow_workspaces()
        expected_workspace_count = 3 if self._pad_heads else 2
        if len(workspace_tensors) != expected_workspace_count:
            raise RuntimeError(
                "B12X workspace DCP gather borrowed an invalid workspace set: "
                f"expected {expected_workspace_count}, got {len(workspace_tensors)}."
            )
        q_workspace = workspace_tensors[0]
        scratch_storage = workspace_tensors[-1]
        expected_q_shape = (
            self._max_batched,
            self._kernel_num_heads,
            self.q_head_dim,
        )
        if (
            tuple(q_workspace.shape) != expected_q_shape
            or q_workspace.dtype != torch.bfloat16
            or q_workspace.device != query_device
            or not q_workspace.is_contiguous()
        ):
            raise RuntimeError(
                "B12X workspace DCP gather borrowed an invalid query workspace: "
                f"expected contiguous {expected_q_shape} on "
                f"{query_device}/torch.bfloat16, got {tuple(q_workspace.shape)} on "
                f"{q_workspace.device}/{q_workspace.dtype} with "
                f"stride={tuple(q_workspace.stride())}."
            )
        if (
            tuple(scratch_storage.shape) != (self._scratch_nbytes,)
            or scratch_storage.dtype != torch.uint8
            or scratch_storage.device != query_device
            or not scratch_storage.is_contiguous()
        ):
            raise RuntimeError(
                "B12X workspace DCP gather borrowed invalid raw scratch: "
                f"expected contiguous ({self._scratch_nbytes},) uint8 on "
                f"{query_device}, got {tuple(scratch_storage.shape)} "
                f"{scratch_storage.dtype} on {scratch_storage.device}."
            )
        q_begin = q_workspace.data_ptr()
        q_end = q_begin + q_workspace.numel() * q_workspace.element_size()
        scratch_begin = scratch_storage.data_ptr()
        scratch_end = scratch_begin + scratch_storage.numel()
        if tuple_input:
            q_workspace_stride_bytes = tuple(
                stride * q_workspace.element_size()
                for stride in q_workspace.stride()[:2]
            )
            if q_begin % 32 != 0 or any(
                stride_bytes % 32 != 0
                for stride_bytes in q_workspace_stride_bytes
            ):
                raise RuntimeError(
                    "B12X workspace DCP gather query workspace does not satisfy "
                    "concat_mla_q 32-byte output alignment: "
                    f"data_ptr={q_begin}, "
                    f"outer_stride_bytes={q_workspace_stride_bytes}."
                )
        if q_end <= q_begin or scratch_end <= scratch_begin:
            raise RuntimeError(
                "B12X workspace DCP gather borrowed a non-positive workspace "
                "storage span."
            )
        if q_begin < scratch_end and scratch_begin < q_end:
            raise RuntimeError(
                "B12X workspace DCP gather query and scratch workspaces overlap."
            )
        for input_name, input_begin, input_end in input_spans:
            if (
                (input_begin < q_end and q_begin < input_end)
                or (input_begin < scratch_end and scratch_begin < input_end)
            ):
                raise RuntimeError(
                    "B12X workspace DCP gather input must be storage-disjoint "
                    "from the borrowed query and scratch workspaces; overlapping "
                    f"component={input_name}."
                )

        # The extend plan necessarily has at least enough raw bytes for its BF16
        # [max_tokens, global_heads, v_head_dim] output buffer. Reuse an
        # output-sized prefix of the raw storage; this does not depend on the
        # output view's internal layout offset. For production [3072, 64, 512],
        # the prefix is 192 MiB and holds 2730 of the 3072
        # [world=4, local_heads=16, head_dim=576] rows per collective.
        gather_scratch_nbytes = (
            self._max_batched
            * self._kernel_num_heads
            * self.kv_lora_rank
            * _BF16_BYTES
        )
        if gather_scratch_nbytes > scratch_storage.numel():
            raise RuntimeError(
                "B12X workspace DCP gather raw scratch is smaller than the "
                f"required extend-output-sized prefix: {scratch_storage.numel()} "
                f"< {gather_scratch_nbytes} bytes."
            )
        bytes_per_chunk_row = (
            world_size * local_heads * head_dim * _BF16_BYTES
        )
        chunk_capacity = gather_scratch_nbytes // bytes_per_chunk_row
        if chunk_capacity != 2730:
            raise RuntimeError(
                "B12X workspace DCP gather production scratch capacity must be "
                f"2730 token rows, got {chunk_capacity}."
            )
        if chunk_capacity <= 0:
            raise RuntimeError(
                "B12X workspace DCP gather raw scratch cannot hold one "
                f"rank-major row ({bytes_per_chunk_row} bytes)."
            )

        q_workspace_flat = q_workspace.view(-1) if tuple_input else None
        chunk_start = 0
        while chunk_start < num_tokens:
            chunk_rows = min(chunk_capacity, num_tokens - chunk_start)
            gather_nbytes = chunk_rows * bytes_per_chunk_row
            if tuple_input:
                local_stage_offset = chunk_start * global_heads * head_dim
                local_stage_numel = chunk_rows * local_heads * head_dim
                local_stage_end = local_stage_offset + local_stage_numel
                prior_finalized_end = chunk_start * global_heads * head_dim
                chunk_destination_end = (
                    chunk_start + chunk_rows
                ) * global_heads * head_dim
                if (
                    local_stage_numel <= 0
                    or local_stage_offset != prior_finalized_end
                    or local_stage_end > chunk_destination_end
                    or local_stage_end > q_workspace.numel()
                ):
                    raise RuntimeError(
                        "B12X workspace DCP gather local tuple staging slice is "
                        "not wholly inside the current unfinalized destination "
                        "chunk or would overwrite prior finalized rows: "
                        f"stage=[{local_stage_offset}, {local_stage_end}), "
                        f"finalized_end={prior_finalized_end}, "
                        f"chunk_end={chunk_destination_end}, "
                        f"workspace_elements={q_workspace.numel()}."
                    )
                assert q_workspace_flat is not None
                local_chunk = (
                    q_workspace_flat.narrow(
                        0, local_stage_offset, local_stage_numel
                    ).view(chunk_rows, local_heads, head_dim)
                )
                if not local_chunk.is_contiguous():
                    raise RuntimeError(
                        "B12X workspace DCP gather produced a noncontiguous "
                        "local tuple staging slice."
                    )
                local_chunk_stride_bytes = tuple(
                    stride * local_chunk.element_size()
                    for stride in local_chunk.stride()[:2]
                )
                if local_chunk.data_ptr() % 32 != 0 or any(
                    stride_bytes % 32 != 0
                    for stride_bytes in local_chunk_stride_bytes
                ):
                    raise RuntimeError(
                        "B12X workspace DCP gather local tuple staging slice "
                        "does not satisfy concat_mla_q 32-byte output alignment: "
                        f"data_ptr={local_chunk.data_ptr()}, "
                        f"outer_stride_bytes={local_chunk_stride_bytes}."
                    )
                ops.concat_mla_q(
                    ql_nope.narrow(0, chunk_start, chunk_rows),
                    q_pe.narrow(0, chunk_start, chunk_rows),
                    local_chunk,
                )
            else:
                local_chunk = q.narrow(0, chunk_start, chunk_rows)
                if not local_chunk.is_contiguous():
                    raise RuntimeError(
                        "B12X workspace DCP gather produced a noncontiguous token "
                        "slice from a contiguous local query."
                    )
            gathered_chunk = (
                scratch_storage.narrow(0, 0, gather_nbytes)
                .view(torch.bfloat16)
                .view(world_size * chunk_rows, local_heads, head_dim)
            )
            dist.all_gather_into_tensor(
                gathered_chunk,
                local_chunk,
                group=process_group,
                async_op=False,
            )
            rank_major_chunk = gathered_chunk.view(
                world_size, chunk_rows, local_heads, head_dim
            )
            for source_rank in range(world_size):
                q_workspace.narrow(0, chunk_start, chunk_rows).narrow(
                    1, source_rank * local_heads, local_heads
                ).copy_(rank_major_chunk.select(0, source_rank))
            chunk_start += chunk_rows

        global_query = q_workspace.narrow(0, 0, num_tokens).narrow(
            1, 0, global_heads
        )
        expected_stride = (global_heads * head_dim, head_dim, 1)
        if (
            tuple(global_query.shape) != (num_tokens, global_heads, head_dim)
            or tuple(global_query.stride()) != expected_stride
            or not global_query.is_contiguous()
        ):
            raise RuntimeError(
                "B12X workspace DCP gather did not produce the exact contiguous "
                "global query layout."
            )
        for source_rank in range(world_size):
            rank_plane = global_query.narrow(
                1, source_rank * local_heads, local_heads
            )
            expected_offset = (
                global_query.storage_offset()
                + source_rank * local_heads * head_dim
            )
            if (
                rank_plane.storage_offset() != expected_offset
                or tuple(rank_plane.stride()) != expected_stride
            ):
                raise RuntimeError(
                    "B12X workspace DCP gather rank-major head ordering invariant "
                    f"failed for source rank {source_rank}."
                )
        return global_query

    def dcp_project_before_merge_in_workspace(
        self,
        attn_out: torch.Tensor,
        lse: torch.Tensor,
        w_uv: torch.Tensor,
    ) -> torch.Tensor:
        """Project a completed eager DCP prefill through the borrowed workspaces."""
        if attn_out.ndim != 3 or lse.ndim != 2 or w_uv.ndim != 3:
            raise ValueError(
                "B12X workspace DCP projection requires rank-3 attention/weight "
                "and rank-2 LSE tensors, got "
                f"attn_out={tuple(attn_out.shape)}, lse={tuple(lse.shape)}, "
                f"W_UV={tuple(w_uv.shape)}."
            )
        num_tokens = int(attn_out.shape[0])
        if (
            self._max_batched != 3072
            or num_tokens < 1025
            or num_tokens > 3072
            or not self.dcp_workspace_non_dbo
            or self.tp_world_size != 4
            or self.dcp_world_size != 4
            or self.num_heads != 16
            or self._input_num_heads != 64
            or self._kernel_num_heads != 64
            or self._pad_heads
            or self.q_head_dim != 576
            or self.kv_lora_rank != 512
            or self.v_head_dim != 256
        ):
            raise RuntimeError(
                "B12X workspace DCP projection is restricted to the validated "
                "production contract: max_tokens=3072, TP/DCP=4/4, "
                "local/global/kernel heads=16/64/64, q_dim=576, "
                "latent/value dims=512/256; got "
                f"max_tokens={self._max_batched}, tokens={num_tokens}, "
                f"TP/DCP={self.tp_world_size}/{self.dcp_world_size}, "
                f"local/global/kernel heads={self.num_heads}/"
                f"{self._input_num_heads}/{self._kernel_num_heads}, "
                f"pad_heads={self._pad_heads}, q_dim={self.q_head_dim}, "
                f"latent/value dims={self.kv_lora_rank}/{self.v_head_dim}."
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "B12X workspace DCP projection is eager-only and cannot run "
                "during CUDA graph capture."
            )

        expected_attn_shape = (num_tokens, 64, 512)
        expected_attn_stride = (64 * 512, 512, 1)
        if (
            tuple(attn_out.shape) != expected_attn_shape
            or tuple(attn_out.stride()) != expected_attn_stride
            or attn_out.dtype != torch.bfloat16
            or attn_out.device != self.device
            or attn_out.device.type != "cuda"
            or not attn_out.is_contiguous()
        ):
            raise ValueError(
                "B12X workspace DCP projection requires exact contiguous BF16 "
                f"attention output {expected_attn_shape} on {self.device}, got "
                f"shape={tuple(attn_out.shape)}, stride={tuple(attn_out.stride())}, "
                f"dtype={attn_out.dtype}, device={attn_out.device}."
            )
        expected_weight_shape = (64, 512, 256)
        if (
            tuple(w_uv.shape) != expected_weight_shape
            or tuple(w_uv.stride()) != (512 * 256, 256, 1)
            or w_uv.dtype != torch.bfloat16
            or w_uv.device != attn_out.device
            or not w_uv.is_contiguous()
        ):
            raise ValueError(
                "B12X workspace DCP projection requires exact contiguous BF16 "
                f"W_UV {expected_weight_shape} on {attn_out.device}, got "
                f"shape={tuple(w_uv.shape)}, stride={tuple(w_uv.stride())}, "
                f"dtype={w_uv.dtype}, device={w_uv.device}."
            )
        expected_lse_shape = (num_tokens, 64)
        if (
            tuple(lse.shape) != expected_lse_shape
            or tuple(lse.stride()) != (64, 1)
            or lse.dtype != torch.float32
            or lse.device != attn_out.device
            or not lse.is_contiguous()
        ):
            raise ValueError(
                "B12X workspace DCP projection requires exact contiguous FP32 "
                f"LSE {expected_lse_shape} on {attn_out.device}, got "
                f"shape={tuple(lse.shape)}, stride={tuple(lse.stride())}, "
                f"dtype={lse.dtype}, device={lse.device}."
            )

        workspace_tensors = self._borrow_workspaces()
        if len(workspace_tensors) != 2:
            raise RuntimeError(
                "B12X workspace DCP projection requires the exact unpadded "
                f"query/scratch pair, got {len(workspace_tensors)} tensors."
            )
        q_workspace = workspace_tensors[0]
        scratch_storage = workspace_tensors[1]
        expected_q_shape = (3072, 64, 576)
        if (
            tuple(q_workspace.shape) != expected_q_shape
            or q_workspace.dtype != torch.bfloat16
            or q_workspace.device != attn_out.device
            or not q_workspace.is_contiguous()
        ):
            raise RuntimeError(
                "B12X workspace DCP projection reborrowed an invalid query "
                f"workspace: expected contiguous {expected_q_shape} BF16 on "
                f"{attn_out.device}, got shape={tuple(q_workspace.shape)}, "
                f"stride={tuple(q_workspace.stride())}, dtype={q_workspace.dtype}, "
                f"device={q_workspace.device}."
            )
        if (
            tuple(scratch_storage.shape) != (self._scratch_nbytes,)
            or scratch_storage.dtype != torch.uint8
            or scratch_storage.device != attn_out.device
            or not scratch_storage.is_contiguous()
        ):
            raise RuntimeError(
                "B12X workspace DCP projection reborrowed invalid raw scratch: "
                f"expected contiguous ({self._scratch_nbytes},) uint8 on "
                f"{attn_out.device}, got shape={tuple(scratch_storage.shape)}, "
                f"dtype={scratch_storage.dtype}, device={scratch_storage.device}."
            )

        full_attn_nbytes = 3072 * 64 * 512 * _BF16_BYTES
        input_numel = 64 * num_tokens * 512
        projected_numel = num_tokens * 64 * 256
        projected_nbytes = projected_numel * _BF16_BYTES
        if q_workspace.numel() < input_numel:
            raise RuntimeError(
                "B12X workspace DCP projection query workspace is too small: "
                f"{q_workspace.numel()} < {input_numel} BF16 elements."
            )
        if scratch_storage.numel() < full_attn_nbytes:
            raise RuntimeError(
                "B12X workspace DCP projection raw scratch lacks the exact "
                f"192 MiB attention prefix: {scratch_storage.numel()} "
                f"< {full_attn_nbytes} bytes."
            )

        q_begin = q_workspace.data_ptr()
        q_end = q_begin + q_workspace.numel() * q_workspace.element_size()
        scratch_begin = scratch_storage.data_ptr()
        scratch_end = scratch_begin + scratch_storage.numel()
        if q_begin < scratch_end and scratch_begin < q_end:
            raise RuntimeError(
                "B12X workspace DCP projection query and scratch workspaces overlap."
            )

        attn_begin = attn_out.data_ptr()
        attn_end = attn_begin + attn_out.numel() * attn_out.element_size()
        if (
            attn_begin != scratch_begin
            or attn_end > scratch_end
            or attn_out.untyped_storage().data_ptr()
            != scratch_storage.untyped_storage().data_ptr()
        ):
            raise RuntimeError(
                "B12X workspace DCP projection attention output must be the "
                "exact contiguous raw-scratch output prefix."
            )

        projected_begin = scratch_begin
        projected_end = projected_begin + projected_nbytes
        lse_begin = lse.data_ptr()
        lse_end = lse_begin + lse.numel() * lse.element_size()
        if (
            lse_begin < q_end
            and q_begin < lse_end
            or lse_begin < projected_end
            and projected_begin < lse_end
            or lse_begin < attn_end
            and attn_begin < lse_end
        ):
            raise RuntimeError(
                "B12X workspace DCP projection LSE storage must be disjoint "
                "from the query input, full attention output, and projected "
                "output prefixes."
            )
        weight_begin = w_uv.data_ptr()
        weight_end = weight_begin + w_uv.numel() * w_uv.element_size()
        if (
            weight_begin < q_end
            and q_begin < weight_end
            or weight_begin < scratch_end
            and scratch_begin < weight_end
        ):
            raise RuntimeError(
                "B12X workspace DCP projection weights must be storage-disjoint "
                "from both borrowed workspaces."
            )

        q_projection_input = (
            q_workspace.view(-1)
            .narrow(0, 0, input_numel)
            .view(64, num_tokens, 512)
        )
        if (
            tuple(q_projection_input.shape) != (64, num_tokens, 512)
            or tuple(q_projection_input.stride()) != (num_tokens * 512, 512, 1)
            or q_projection_input.data_ptr() != q_begin
            or q_projection_input.untyped_storage().data_ptr()
            != q_workspace.untyped_storage().data_ptr()
            or not q_projection_input.is_contiguous()
        ):
            raise RuntimeError(
                "B12X workspace DCP projection failed to map the exact contiguous "
                "query-workspace prefix."
            )
        projected_head_major = (
            scratch_storage.narrow(0, 0, projected_nbytes)
            .view(torch.bfloat16)
            .view(64, num_tokens, 256)
        )
        if (
            tuple(projected_head_major.shape) != (64, num_tokens, 256)
            or tuple(projected_head_major.stride())
            != (num_tokens * 256, 256, 1)
            or projected_head_major.data_ptr() != scratch_begin
            or projected_head_major.untyped_storage().data_ptr()
            != scratch_storage.untyped_storage().data_ptr()
            or not projected_head_major.is_contiguous()
        ):
            raise RuntimeError(
                "B12X workspace DCP projection failed to map the exact contiguous "
                "head-major raw-scratch prefix."
            )
        projected = projected_head_major.transpose(0, 1)
        if (
            tuple(projected.shape) != (num_tokens, 64, 256)
            or tuple(projected.stride()) != (256, num_tokens * 256, 1)
            or projected.data_ptr() != scratch_begin
            or projected.untyped_storage().data_ptr()
            != scratch_storage.untyped_storage().data_ptr()
        ):
            raise RuntimeError(
                "B12X workspace DCP projection failed to expose the exact "
                "token-major view of the contiguous head-major scratch prefix."
            )

        q_projection_input.copy_(attn_out.transpose(0, 1))
        torch.bmm(
            q_projection_input,
            w_uv,
            out=projected_head_major,
        )
        return projected

    def dcp_reduce_scatter_output_in_workspace(
        self,
        corrected_attn_out: torch.Tensor,
    ) -> torch.Tensor:
        """Map the dead query prefix as eager DCP reduce-scatter output."""
        if corrected_attn_out.ndim != 3:
            raise ValueError(
                "B12X workspace DCP reduce-scatter output requires a rank-3 "
                f"corrected attention tensor, got {tuple(corrected_attn_out.shape)}."
            )
        num_tokens = int(corrected_attn_out.shape[0])
        if (
            self._max_batched != 3072
            or not self.dcp_workspace_non_dbo
            or num_tokens < 1025
            or num_tokens > 3072
            or self.tp_world_size != 4
            or self.dcp_world_size != 4
            or self.num_heads != 16
            or self._input_num_heads != 64
            or self._kernel_num_heads != 64
            or self._pad_heads
            or self.q_head_dim != 576
            or self.kv_lora_rank != 512
            or self.v_head_dim != 256
        ):
            raise RuntimeError(
                "B12X workspace DCP reduce-scatter output is restricted to the "
                "validated production contract: tokens=1025..3072, "
                "max_tokens=3072, TP/DCP=4/4, local/global/kernel "
                "heads=16/64/64, q_dim=576, latent/value dims=512/256; got "
                f"tokens={num_tokens}, max_tokens={self._max_batched}, "
                f"TP/DCP={self.tp_world_size}/{self.dcp_world_size}, "
                f"local/global/kernel heads={self.num_heads}/"
                f"{self._input_num_heads}/{self._kernel_num_heads}, "
                f"pad_heads={self._pad_heads}, q_dim={self.q_head_dim}, "
                f"latent/value dims={self.kv_lora_rank}/{self.v_head_dim}."
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "B12X workspace DCP reduce-scatter output is eager-only and "
                "cannot run during CUDA graph capture."
            )

        expected_input_shape = (num_tokens, 64, 256)
        expected_input_stride = (256, num_tokens * 256, 1)
        input_head_major = corrected_attn_out.movedim(0, 1)
        if (
            tuple(corrected_attn_out.shape) != expected_input_shape
            or tuple(corrected_attn_out.stride()) != expected_input_stride
            or corrected_attn_out.dtype != torch.bfloat16
            or corrected_attn_out.device != self.device
            or corrected_attn_out.device.type != "cuda"
            or tuple(input_head_major.shape) != (64, num_tokens, 256)
            or tuple(input_head_major.stride()) != (num_tokens * 256, 256, 1)
            or not input_head_major.is_contiguous()
        ):
            raise ValueError(
                "B12X workspace DCP reduce-scatter output requires the exact "
                "token-major view of a contiguous head-major corrected BF16 "
                f"input {expected_input_shape} on {self.device}, got "
                f"shape={tuple(corrected_attn_out.shape)}, "
                f"stride={tuple(corrected_attn_out.stride())}, "
                f"dtype={corrected_attn_out.dtype}, "
                f"device={corrected_attn_out.device}."
            )

        workspace_tensors = self._borrow_workspaces()
        if len(workspace_tensors) != 2:
            raise RuntimeError(
                "B12X workspace DCP reduce-scatter output requires the exact "
                f"unpadded query/scratch pair, got {len(workspace_tensors)} tensors."
            )
        q_workspace = workspace_tensors[0]
        scratch_storage = workspace_tensors[1]
        expected_q_shape = (3072, 64, 576)
        if (
            tuple(q_workspace.shape) != expected_q_shape
            or tuple(q_workspace.stride()) != (64 * 576, 576, 1)
            or q_workspace.dtype != torch.bfloat16
            or q_workspace.device != corrected_attn_out.device
            or not q_workspace.is_contiguous()
        ):
            raise RuntimeError(
                "B12X workspace DCP reduce-scatter output reborrowed an invalid "
                f"query workspace: expected contiguous {expected_q_shape} BF16 "
                f"on {corrected_attn_out.device}, got "
                f"shape={tuple(q_workspace.shape)}, "
                f"stride={tuple(q_workspace.stride())}, "
                f"dtype={q_workspace.dtype}, device={q_workspace.device}."
            )
        if (
            tuple(scratch_storage.shape) != (self._scratch_nbytes,)
            or scratch_storage.dtype != torch.uint8
            or scratch_storage.device != corrected_attn_out.device
            or not scratch_storage.is_contiguous()
        ):
            raise RuntimeError(
                "B12X workspace DCP reduce-scatter output reborrowed invalid raw "
                f"scratch: expected contiguous ({self._scratch_nbytes},) uint8 "
                f"on {corrected_attn_out.device}, got "
                f"shape={tuple(scratch_storage.shape)}, "
                f"dtype={scratch_storage.dtype}, device={scratch_storage.device}."
            )

        projected_nbytes = num_tokens * 64 * 256 * _BF16_BYTES
        output_numel = 16 * num_tokens * 256
        output_nbytes = output_numel * _BF16_BYTES
        if scratch_storage.numel() < projected_nbytes:
            raise RuntimeError(
                "B12X workspace DCP reduce-scatter input exceeds raw scratch: "
                f"{projected_nbytes} > {scratch_storage.numel()} bytes."
            )
        if q_workspace.numel() < output_numel:
            raise RuntimeError(
                "B12X workspace DCP reduce-scatter output exceeds the query "
                f"workspace: {output_numel} > {q_workspace.numel()} BF16 elements."
            )

        q_begin = q_workspace.data_ptr()
        q_end = q_begin + q_workspace.numel() * q_workspace.element_size()
        scratch_begin = scratch_storage.data_ptr()
        scratch_end = scratch_begin + scratch_storage.numel()
        if q_begin < scratch_end and scratch_begin < q_end:
            raise RuntimeError(
                "B12X workspace DCP reduce-scatter query and scratch workspaces "
                "overlap."
            )

        input_begin = corrected_attn_out.data_ptr()
        input_end = input_begin + projected_nbytes
        if (
            input_begin != scratch_begin
            or input_end > scratch_end
            or corrected_attn_out.untyped_storage().data_ptr()
            != scratch_storage.untyped_storage().data_ptr()
            or input_head_major.data_ptr() != scratch_begin
        ):
            raise RuntimeError(
                "B12X workspace DCP reduce-scatter input must be the exact "
                "head-major raw-scratch prefix exposed through its token-major view."
            )

        output_head_major = (
            q_workspace.view(-1)
            .narrow(0, 0, output_numel)
            .view(16, num_tokens, 256)
        )
        output = output_head_major.transpose(0, 1)
        output_begin = output.data_ptr()
        output_end = output_begin + output_nbytes
        if (
            tuple(output_head_major.shape) != (16, num_tokens, 256)
            or tuple(output_head_major.stride()) != (num_tokens * 256, 256, 1)
            or not output_head_major.is_contiguous()
            or tuple(output.shape) != (num_tokens, 16, 256)
            or tuple(output.stride()) != (256, num_tokens * 256, 1)
            or output_begin != q_begin
            or output_end > q_end
            or output.untyped_storage().data_ptr()
            != q_workspace.untyped_storage().data_ptr()
            or output.movedim(0, 1).data_ptr() != q_begin
            or not output.movedim(0, 1).is_contiguous()
        ):
            raise RuntimeError(
                "B12X workspace DCP reduce-scatter failed to map the exact "
                "token-major output view over the contiguous head-major query "
                "workspace prefix."
            )
        if (
            output_begin < scratch_end
            and scratch_begin < output_end
            or output_begin < input_end
            and input_begin < output_end
        ):
            raise RuntimeError(
                "B12X workspace DCP reduce-scatter output must be fully disjoint "
                "from the raw scratch and corrected input."
            )
        return output

    def _sync_warmup(self) -> None:
        if self.device.type == "cuda":
            torch.accelerator.synchronize(self.device)
        if self.dcp_world_size <= 1:
            return
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            get_dcp_group().barrier()
        except Exception:
            return
        finally:
            if self.device.type == "cuda":
                torch.accelerator.synchronize(self.device)

    def _prewarm_extend_kernels_once(self, max_batched: int) -> None:
        if self.device.type != "cuda":
            return
        key = (
            self.device.index,
            self.q_head_dim,
            self.kv_lora_rank,
            self._kernel_num_heads,
            int(self.topk_tokens),
            int(self.block_size),
            bool(self.need_to_return_lse_for_decode),
        )
        if key in _EXTEND_PREWARM_DONE:
            return
        _EXTEND_PREWARM_DONE.add(key)

        rows_to_warm = (1, 2, 4, max(1, int(max_batched)))
        seen_rows: set[int] = set()
        # GLM cache records are 656 B/token (fp8_ds_mla) or 368 B/token
        # (nvfp4_ds_mla); the real KV cache is laid out
        # (num_blocks, block_size, record_bytes) (see the allocator at the
        # block-shape branch above), so a page's stride(0) =
        # block_size*record_bytes. The prewarm dummy must match that layout --
        # (1, block_size, record_bytes) -- so _cache_block_stride_bytes sees
        # stride >= page_size*record_bytes. The prior (block_size, 1, ...)
        # shape put block_size in dim 0, giving stride(0) = record_bytes <
        # page_size*record_bytes, which tripped the SM120 stride assertion
        # whenever this prewarm ran (i.e. spec + cudagraphs, the first config
        # to reach here; verifier-only and eager-snap both skipped it).
        # One page is enough: prewarm top-k indices all point at slot zero.
        # kingdom(nvfp4_ds_mla): record width follows the cache dtype.
        record_bytes = {
            "nvfp4_ds_mla": 368,
            "nf3_ds_mla": 304,
            "nf3bf16_ds_mla": 368,
        }.get(self.kv_cache_dtype, 656)
        kv_cache = torch.zeros(
            (1, self.block_size, record_bytes),
            dtype=torch.uint8,
            device=self.device,
        )
        for rows in rows_to_warm:
            rows = int(rows)
            if rows in seen_rows:
                continue
            seen_rows.add(rows)
            q = torch.zeros(
                (rows, self._kernel_num_heads, self.q_head_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
            selected_indices = torch.zeros(
                (rows, self.topk_tokens), dtype=torch.int32, device=self.device
            )
            cache_seqlens = torch.full(
                (1,), self.block_size, dtype=torch.int32, device=self.device
            )
            nsa_cache_seqlens = torch.ones(
                (rows,), dtype=torch.int32, device=self.device
            )
            scratch_storage = torch.empty(
                (self._scratch_nbytes,), dtype=torch.uint8, device=self.device
            )
            binding = self._extend_plan.bind(
                scratch=scratch_storage,
                q=q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            if self.need_to_return_lse_for_decode:
                self._sparse_mla_extend_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    return_lse=True,
                    lse_scale="natural",
                    **self._b12x_nvfp4_kwargs,
                )
            else:
                self._sparse_mla_extend_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    **self._b12x_nvfp4_kwargs,
                )
            self._sync_warmup()

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # q arrives as (mqa_ql_nope[T, H, kv_lora_rank], mqa_q_pe[T, H, rope]);
        # b12x's GLM_NSA contract wants a single contiguous [T, H, 576] tensor.
        # Co-allocate the q-concat buffer and the per-call attention scratch in ONE
        # get_simultaneous call so they receive distinct, non-overlapping offsets:
        # the kernel reads q while writing the scratch (tmp_output/output), and the
        # manager packs every call from offset 0, so separate calls would alias q
        # with the scratch and corrupt the result.
        workspace_tensors = self._borrow_workspaces()
        q_workspace = workspace_tensors[0]
        dense_out_workspace = workspace_tensors[1] if self._pad_heads else None
        scratch_storage = workspace_tensors[-1]
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            num_actual_toks = ql_nope.shape[0]
            num_input_heads = ql_nope.shape[1]
            if num_input_heads != self._input_num_heads:
                raise ValueError(
                    "B12X_MLA_SPARSE query heads do not match the planned "
                    f"head count: {num_input_heads} != {self._input_num_heads}."
                )
            q_buffer = q_workspace[:num_actual_toks]
            q_all = q_buffer[:, :num_input_heads]
            ops.concat_mla_q(ql_nope, q_pe, q_all)
        else:
            num_actual_toks = q.shape[0]
            num_input_heads = q.shape[1]
            if num_input_heads != self._input_num_heads:
                raise ValueError(
                    "B12X_MLA_SPARSE query heads do not match the planned "
                    f"head count: {num_input_heads} != {self._input_num_heads}."
                )
            q_buffer = q_workspace[:num_actual_toks]
            q_all = q_buffer[:, :num_input_heads]
            exact_workspace_alias = (
                tuple(q.shape) == tuple(q_all.shape)
                and tuple(q.stride()) == tuple(q_all.stride())
                and q.dtype == q_all.dtype
                and q.device == q_all.device
                and q.untyped_storage().data_ptr()
                == q_all.untyped_storage().data_ptr()
                and q.storage_offset() == q_all.storage_offset()
            )
            if not exact_workspace_alias:
                q_all.copy_(q.contiguous())

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        if self.dcp_world_size > 1:
            # The indexer globally merges logical top-k ids across DCP ranks.
            # Compact just this rank's winners into local physical cache slots;
            # the outer MLA layer combines the rank-local outputs using LSE.
            assert attn_metadata.req_id_per_token is not None
            assert attn_metadata.page_table_1 is not None
            assert attn_metadata.nsa_cache_seqlens is not None
            selected_indices = attn_metadata.page_table_1[
                :num_actual_toks, : topk_indices.shape[1]
            ]
            nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[:num_actual_toks]
            triton_convert_dcp_global_index_to_local_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_world_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                out=selected_indices,
                valid_counts=nsa_cache_seqlens,
            )
            per_token_cache = attn_metadata.cache_seq_lens_per_token[:num_actual_toks]
            torch.minimum(
                nsa_cache_seqlens,
                per_token_cache,
                out=nsa_cache_seqlens,
            )
            _mask_page_table_after_nsa_len(selected_indices, nsa_cache_seqlens)
        else:
            # Without DCP, the b12x indexer writes flat physical cache slots
            # directly into the shared top-k buffer.
            selected_indices = topk_indices
            nsa_cache_seqlens = attn_metadata.cache_seq_lens_per_token[:num_actual_toks]

        # KV cache -> paged rank-3 uint8. B12X unified SM120 kernels consume
        # flat slot ids in selected_indices, but compute raw byte offsets as:
        #   block = slot // page_size, local = slot % page_size
        # so the cache tensor itself must expose a per-block stride of
        # block_size * record_bytes. The older split path used a token-flat
        # (num_slots, 1, bytes) view; that makes stride(0) one record and breaks
        # the unified block-stride contract.
        kv_u8 = kv_c_and_k_pe_cache.view(torch.uint8)
        if kv_u8.ndim == 3 and kv_u8.shape[1] == self.block_size:
            kv_cache = kv_u8
        elif kv_u8.ndim == 3 and kv_u8.shape[1] == 1:
            if kv_u8.shape[0] % self.block_size != 0:
                raise ValueError(
                    "B12X_MLA_SPARSE flat KV cache rows must be divisible by "
                    f"block_size={self.block_size}; got {kv_u8.shape[0]}"
                )
            kv_cache = kv_u8.reshape(-1, self.block_size, kv_u8.shape[-1])
        else:
            raise ValueError(
                "B12X_MLA_SPARSE expected fp8_ds_mla/nvfp4_ds_mla/nf3_ds_mla/"
                "nf3bf16_ds_mla KV cache as "
                f"(blocks,{self.block_size},bytes) or (slots,1,bytes), got "
                f"{tuple(kv_u8.shape)}"
            )
        if not kv_cache.is_contiguous():
            raise ValueError(
                "B12X_MLA_SPARSE requires a contiguous native paged KV cache; "
                f"got stride={tuple(kv_cache.stride())}"
            )

        use_decode_kernel = attn_metadata.max_query_len <= 1 or (
            self.spec_extend_as_decode
            and attn_metadata.max_query_len <= self.spec_decode_max_q
            and num_actual_toks <= attn_metadata.num_reqs * self.spec_decode_max_q
            and num_actual_toks <= self._decode_max_rows
        )
        if use_decode_kernel:
            cache_seqlens = (
                attn_metadata.cache_seq_lens_per_req
                if attn_metadata.max_query_len <= 1
                else attn_metadata.cache_seq_lens_per_token[:num_actual_toks]
            )
            decode_q = q_all
            if self._pad_heads:
                decode_q = q_buffer[:, : self._kernel_num_heads]
                decode_q[:, self._input_num_heads :, :].zero_()
            # Eager bind maps caller-owned scratch into views. forced_num_splits
            # pins the planner choice for this captured graph; the merge kernel is
            # specialized on that count and needs no device-side control fill.
            binding = self._decode_plan.bind(
                scratch=scratch_storage,
                q=decode_q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            if self.need_to_return_lse_for_decode:
                out, lse = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_decode_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        forced_num_splits=self._num_splits_cap,
                        return_lse=True,
                        lse_scale="natural",
                        **self._b12x_nvfp4_kwargs,
                    ),
                )
                if self._pad_heads:
                    assert dense_out_workspace is not None
                    dense_out = dense_out_workspace[:num_actual_toks]
                    dense_out.copy_(out[:, : self._input_num_heads, :])
                    out = dense_out
                    lse = lse[:, : self._input_num_heads]
                return out, lse
            out = cast(
                torch.Tensor,
                self._sparse_mla_decode_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    forced_num_splits=self._num_splits_cap,
                    **self._b12x_nvfp4_kwargs,
                ),
            )
            if self._pad_heads:
                assert dense_out_workspace is not None
                dense_out = dense_out_workspace[:num_actual_toks]
                dense_out.copy_(out[:, : self._input_num_heads, :])
                out = dense_out
            return out, None
        else:
            # Extend / prefill -> single-pass unified prefill (no split-K
            # scratch needed; only output_buffer is read). b12x supports 8-head
            # granularity, so only a non-aligned local tail is padded here.
            cache_seqlens = attn_metadata.cache_seq_lens_per_req
            prefill_q = q_all
            if self._pad_heads:
                prefill_q = q_buffer[:, : self._kernel_num_heads]
                prefill_q[:, self._input_num_heads :, :].zero_()

            # Eager bind into the extend views container (single-pass prefill;
            # no split-K, output_buffer is the only scratch the kernel writes).
            binding = self._extend_plan.bind(
                scratch=scratch_storage,
                q=prefill_q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            lse = None
            if self.need_to_return_lse_for_decode:
                out, lse = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_extend_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        return_lse=True,
                        lse_scale="natural",
                        **self._b12x_nvfp4_kwargs,
                    ),
                )
            else:
                out = cast(
                    torch.Tensor,
                    self._sparse_mla_extend_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        **self._b12x_nvfp4_kwargs,
                    ),
                )
            if self._pad_heads:
                assert dense_out_workspace is not None
                dense_out = dense_out_workspace[:num_actual_toks]
                dense_out.copy_(out[:, : self._input_num_heads, :])
                out = dense_out
                if lse is not None:
                    lse = lse[:, : self._input_num_heads]
        return out, lse
