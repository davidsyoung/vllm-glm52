"""
hybrid_loader.py — Path A: TP4-native NVFP4/NF3 hybrid loader for vLLM (modelopt path).
Deploy via .pth (`import hybrid_loader`). Fires in every worker.

GLM-5.2 hybrid checkpoint: routed experts are per-layer mixed — top-K (=64) NVFP4 (crisp) +
the rest NF3 (3-bit, group-32). Non-experts bf16 (excluded). Stock modelopt would allocate
uniform NVFP4 for all 256 experts (~420 GiB → OOM on 384). This loader allocates COMPACT
two-group storage (~326 GiB) and does a reference two-pass forward.

Interceptions (armed by an import hook on vllm...modelopt):
  1. ModelOptNvFp4Config.from_config -> stash `hybrid_bit_map` (stock strips unknown keys).
  2. FusedMoEMethodCls -> HybridNvFp4MoE.

NO mapping patch: stock RoutedExperts.make_expert_params_mapping is PREFIX-based, so
`...gate_proj.weight_packed` -> `...routed_experts.w13_weight_packed`, `...weight_scale` ->
`...w13_weight_scale`, etc. We register compact params under those exact names; the per-layer
weight_loader demuxes NVFP4 vs NF3 by the expert's group (this layer's remap) and TP-shards.

Facts baked in (verified against the eldritch image):
  * apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input): routing is
    upstream (sigmoid/noaux_tc/norm/2.5x already applied); shared experts run by the runner
    separately -> apply returns ROUTED-ONLY output.
  * moe_kernel stays None for hybrid layers -> is_monolithic False (via experts_cls) -> apply
    dispatched; maybe_make_prepare_finalize overridden to None so MK-init doesn't raise.
  * on-disk NVFP4 block scales are LINEAR (2D) -> dequantize_to_dtype(..., swizzle=False).
  * TP=4, moe_intermediate=2048 -> clean chunk, no padding. gate/up shard dim0, down shard dim1.
"""
import os, sys, importlib.abc, importlib.util

_NF3_VALS = [-1.0, -0.6047, -0.3563, -0.1275, 0.1275, 0.3563, 0.6047, 1.0]
_HYBRID_DEBUG = os.environ.get("HYBRID_DEBUG") not in (None, "", "0")  # eager-only: per-tier norms
# triton | b12x | b12x_nf3 | ref.  b12x_nf3 = BOTH tiers through the b12x W4A16
# CuteDSL kernel: kept-64 NVFP4 as weight_layout="packed", NF3-192 as the new
# weight_layout="nf3_2p1" (e4m3_k32 scales).  HYBRID_NF3 is ignored in that mode.
_HYBRID_KEPT = os.environ.get("HYBRID_KEPT", "triton")
_HYBRID_TIER = os.environ.get("HYBRID_TIER", "both")  # both|a|b (b12x_nf3 isolation)
_HYBRID_ACT_CAPTURE = os.environ.get("HYBRID_ACT_CAPTURE", "")  # dir: save MoE-layer inputs for GPTQ Hessians
_act_store = {"n": 0, "buf": {}, "flushed": {}}
_HYBRID_NF3 = os.environ.get("HYBRID_NF3", "fast")    # fast | ref  (NF3 3-bit tier)
_HYBRID_PROFILE = os.environ.get("HYBRID_PROFILE", "0") == "1"  # per-tier CUDA-event timing at prefill M>64
# Exact-shape SM120 TC decode is enabled by default. HYBRID_TC_DECODE is the
# master override for mixed-layer E64 NVFP4 admission; HYBRID_NF3_TC_DECODE
# additionally admits the mixed-layer E192 nf3_2p1 tier. Set either variable
# to 0 to diagnose or disable its path. Runtime selection remains exact m=4.
_HYBRID_TC_DECODE = os.environ.get("HYBRID_TC_DECODE", "1") == "1"
_HYBRID_NF3_TC_DECODE = os.environ.get("HYBRID_NF3_TC_DECODE", "1") == "1"
# Graph-capture-only one-grid heterogeneous decode. This remains an additive
# override: both serial direct-launch paths must also be enabled.
_HYBRID_HETERO_DECODE = os.environ.get("HYBRID_HETERO_DECODE", "1") == "1"
_prof_store = {"n": 0, "a_ms": 0.0, "b_ms": 0.0, "rows": 0, "pend": []}
_dbg_first_build = [True]   # keep kept-tier originals for ONLY the first b12x layer (VRAM)
_HBM = None  # cached hybrid_bit_map (per-process); read once from the checkpoint config.json

def _admit_tc_decode_compile(weight_layout, has_nf3_tier,
                             tc_decode_enabled, nf3_tc_decode_enabled):
    """Pure compile admission for the guarded mixed-layer direct launch."""
    return (tc_decode_enabled and has_nf3_tier
            and (weight_layout == "packed"
                 or (nf3_tc_decode_enabled
                     and weight_layout == "nf3_2p1")))


def _combined_tier_local_descriptors(remap):
    """Validate an exact E64/E192 partition and return its 256 descriptors."""
    descriptors = [-1] * 256
    seen_local = (set(), set())
    for global_id, tier_local in remap.items():
        try:
            global_id_i = int(global_id)
            tier, local_id = tier_local
            tier_i, local_id_i = int(tier), int(local_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid heterogeneous expert remap entry") from exc
        if global_id_i != global_id or not 0 <= global_id_i < 256:
            raise ValueError(f"invalid global expert ID {global_id!r}")
        if descriptors[global_id_i] != -1:
            raise ValueError(f"duplicate global expert ID {global_id_i}")
        local_limit = 64 if tier_i == 0 else 192 if tier_i == 1 else 0
        if (tier_i != tier or local_id_i != local_id
                or local_limit == 0 or not 0 <= local_id_i < local_limit):
            raise ValueError(
                f"invalid tier/local expert descriptor {(tier, local_id)!r}")
        if local_id_i in seen_local[tier_i]:
            raise ValueError(
                f"duplicate tier/local expert descriptor {(tier_i, local_id_i)!r}")
        seen_local[tier_i].add(local_id_i)
        descriptors[global_id_i] = (
            local_id_i if tier_i == 0 else 0x100 | local_id_i)
    if any(descriptor < 0 for descriptor in descriptors):
        raise ValueError("heterogeneous remap does not cover all 256 global experts")
    if seen_local[0] != set(range(64)) or seen_local[1] != set(range(192)):
        raise ValueError("heterogeneous remap is not a complete E64/E192 partition")
    return descriptors


def _borrow_one_grid_scratch(torch, buffers, device, scratch_elements):
    """Borrow exact one-grid prefixes from mutually exclusive serial buffers."""
    specs = (
        ("fc1", "intermediate_cache13", torch.bfloat16, 2, (32, 1024)),
        ("activated", "intermediate_cache2", torch.bfloat16, 2, (32, 512)),
        ("fc1_c_tmp", "fc1_c_tmp", torch.float32, 4,
         (int(scratch_elements),)),
        ("fc2_c_tmp", "fc2_c_tmp", torch.float32, 4,
         (int(scratch_elements),)),
    )
    borrowed = {}
    ranges = []
    source_storage_ids = set()
    retained_bytes = 0
    for target_name, source_name, dtype, expected_element_bytes, shape in specs:
        source = getattr(buffers, source_name, None)
        if source is None:
            raise RuntimeError(
                f"one-grid scratch source {source_name} is unavailable")
        elements = 1
        for extent in shape:
            elements *= int(extent)
        element_bytes = int(source.element_size())
        source_bytes = int(source.numel()) * element_bytes
        storage = source.untyped_storage()
        storage_id = int(storage.data_ptr())
        storage_bytes = int(storage.nbytes())
        storage_start = storage_id
        storage_stop = storage_start + storage_bytes
        source_start = int(source.data_ptr())
        source_stop = source_start + source_bytes
        source_offset_bytes = int(source.storage_offset()) * element_bytes
        if (source.dtype != dtype
                or source.device != device
                or not source.is_contiguous()
                or int(source.numel()) < elements
                or element_bytes != expected_element_bytes
                or source_start == 0
                or source_start % 16
                or storage_start == 0
                or source_offset_bytes != 0
                or storage_start + source_offset_bytes != source_start
                or source_start < storage_start
                or source_stop > storage_stop):
            raise RuntimeError(
                f"one-grid scratch source {source_name} failed "
                "dtype/device/contiguity/alignment/capacity/storage bounds")
        if storage_id in source_storage_ids:
            raise RuntimeError(
                "one-grid scratch sources do not have distinct storage identity")
        source_storage_ids.add(storage_id)

        view = source.view(-1)[:elements].view(shape)
        view_storage = view.untyped_storage()
        view_start = int(view.data_ptr())
        view_stop = view_start + int(view.numel()) * int(view.element_size())
        if (view.dtype != dtype
                or view.device != device
                or not view.is_contiguous()
                or tuple(view.shape) != shape
                or int(view.numel()) != elements
                or int(view.element_size()) != element_bytes
                or view_start % 16
                or view_start != source_start
                or int(view.storage_offset()) != int(source.storage_offset())
                or int(view_storage.data_ptr()) != storage_id
                or int(view_storage.nbytes()) != storage_bytes
                or view_start < source_start
                or view_stop > source_stop
                or view_start < storage_start
                or view_stop > storage_stop):
            raise RuntimeError(
                f"borrowed one-grid scratch {target_name} failed exact "
                "shape/storage admission")
        borrowed[target_name] = view
        ranges.append((target_name, source_name, view_start, view_stop))
        retained_bytes += source_bytes

    for left_index, left in enumerate(ranges):
        for right in ranges[left_index + 1:]:
            if left[2] < right[3] and right[2] < left[3]:
                raise RuntimeError(
                    f"borrowed one-grid scratch overlaps: {left[0]} and "
                    f"{right[0]}")
    borrowed_bytes = sum(stop - start for _, _, start, stop in ranges)
    return borrowed, borrowed_bytes, retained_bytes


# ---- b12x_nf3 backend constants / shared runtime ----
# Pinned CTA tiles (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n): the NF3
# flat-span weight layout is packed for a SPECIFIC tile_n, but the kernel's
# auto tile selection is m-dependent (fc1_tile_n flips 128<->256 across m).
# (64,256,64,256) validated for our shapes (fc1 N=1024 K=6144 / fc2 N=6144
# K=512) at BOTH moe_block_size 8 (decode) and 64 (prefill), both scale
# formats: smem fits (45-76KB <= 100.9KB) and the SM121 reg-count table has
# entries for (256,{1,4},16,4). It is also exactly what auto-selection picks
# for the max-m prefill, so prefill throughput is the natural one.
_B12X_NF3_TILES = (64, 256, 64, 256)
_B12X_NF3_DECODE_M = 8          # compile capacity; runtime selects only exact m=4
_B12X_NF3_MAX_TOKENS = int(os.environ.get("HYBRID_B12X_MAX_TOKENS", "8192"))
_b12x_nf3_rt = {                # module-level, shared across ALL layers
    "max_m": None,              # fixed at first apply: max(env, first-call m)
    "topk": None,
    "launches": {},             # (E, layout, scale_fmt, topk, max_m) -> (dec, pre)
    "buffers": None,            # W4A16PackedBuffers planned at max_m/route_E=256
    "out_a": None,              # [max_m, H] bf16 per-tier outputs (fully overwritten
    "out_b": None,              #  by every serial/fallback run_w4a16_moe call)
    # Exact one-grid scratch is shared by sequential mixed layers.  Each
    # admitted layer separately owns the stable output returned into its graph.
    "hetero_direct": None,
    "hetero_direct_failed": False,
    "hetero_direct_warned": False,
    # Mapped-global-ID compile failure is recoverable: local-ID one-grid stays armed.
    "hetero_mapped": None,
    "hetero_mapped_failed": False,
    "hetero_mapped_failure": None,
}


def _load_hbm(quant_config=None):
    """Reliable in-worker hybrid_bit_map: config-object attr, else read config.json off disk.
    (from_config stash on the config object does NOT survive pickling to TP workers.)"""
    global _HBM
    if _HBM is not None:
        return _HBM
    hbm = getattr(quant_config, "hybrid_bit_map", None) if quant_config is not None else None
    if hbm is None:
        try:
            import json, os
            from vllm.config import get_current_vllm_config
            mp = get_current_vllm_config().model_config.model
            cfgp = os.path.join(mp, "config.json")
            qc = json.load(open(cfgp)).get("quantization_config", {})
            hbm = qc.get("hybrid_bit_map")
            print(f"[hybrid_loader] hbm read from {cfgp}: {len(hbm) if hbm else 0} layers", flush=True)
        except Exception as e:
            print("[hybrid_loader] hbm config.json read failed:", e, flush=True)
    _HBM = hbm
    return hbm


def _unpack_nf3(packed, scale, out_cols, blk=32):
    """packed:[R, out_cols//8*3] uint8, scale:[R, out_cols//blk] fp8 -> [R, out_cols] bf16."""
    import torch
    nf = torch.tensor(_NF3_VALS, device=packed.device, dtype=torch.float32)
    R = packed.shape[0]
    p = packed.reshape(R, out_cols // 8, 3).to(torch.int32)
    w24 = p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
    codes = torch.stack([(w24 >> (3 * i)) & 0x7 for i in range(8)], -1).reshape(R, out_cols)
    return (nf[codes.long()] * scale.float().repeat_interleave(blk, 1)).to(torch.bfloat16)


def _fp8_ne_transform(weights):
    """Dequant fp8 non-expert weights -> bf16 on the fly. Speaks BOTH scale
    dialects: `.weight_scale_fp8` (rev-2, bf16 per-channel) and `.weight_scale`
    U8 (rev-3 MXFP8, e8m0 per-32 groups, [out, in//32]). Expert tensors never
    match (uint8-packed weights / F8 scales) and pass through untouched."""
    import torch
    scales, pend = {}, {}

    def _deq(w, s):
        kind, st = s
        if kind == "chan":
            return (w.to(torch.float32) * st.to(torch.float32).unsqueeze(1)).to(torch.bfloat16)
        sc = torch.pow(2.0, st.to(torch.float32) - 127.0)
        return (w.to(torch.float32) * sc.repeat_interleave(32, 1)).to(torch.bfloat16)

    def _stage(t):
        # fastsafetensors yields CUDA tensors; holding tier pairs in VRAM
        # mid-shard OOMs the load. CPU-stage ONLY the F8 tier (16GB of 327) —
        # experts keep streaming GPU-direct.
        return t.cpu() if getattr(t, "is_cuda", False) else t

    for name, t in weights:
        if name.endswith(".weight_scale_fp8"):
            wn = name[:-len(".weight_scale_fp8")] + ".weight"
            w = pend.pop(wn, None)
            if w is not None:
                yield wn, _deq(w, ("chan", _stage(t)))
            else:
                scales[wn] = ("chan", _stage(t))
        elif (name.endswith(".weight_scale")
              and getattr(t, "dtype", None) == torch.uint8 and t.dim() == 2):
            wn = name[:-len(".weight_scale")] + ".weight"
            w = pend.pop(wn, None)
            if w is not None:
                yield wn, _deq(w, ("mx", _stage(t)))
            else:
                scales[wn] = ("mx", _stage(t))
        elif name.endswith(".weight") and getattr(t, "dtype", None) == torch.float8_e4m3fn:
            s = scales.pop(name, None)
            if s is not None:
                yield name, _deq(_stage(t), s)
            else:
                pend[name] = _stage(t)
        else:
            yield name, t
    for wn, t in pend.items():   # unmatched fp8 weight (shouldn't happen) -> best-effort
        yield wn, t.to(torch.bfloat16)


def _deq_nv(w, s, g, gs, deq):
    """NVFP4 dequant (linear on-disk scales). g is [2] for fused w13 or scalar for w2."""
    import torch
    if g.numel() == 2:  # fused w13: gate rows use g[0], up rows use g[1]
        half = w.shape[0] // 2
        a = deq(w[:half], s[:half], g[0], torch.bfloat16, gs, swizzle=False)
        b = deq(w[half:], s[half:], g[1], torch.bfloat16, gs, swizzle=False)
        return torch.cat([a, b], 0)
    return deq(w, s, g.reshape(()), torch.bfloat16, gs, swizzle=False)


def _patch(mod):
    import re, torch
    import torch.nn.functional as F
    from vllm.model_executor.utils import set_weight_attrs
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import dequantize_to_dtype
    from vllm.distributed import (get_tensor_model_parallel_rank,
                                  get_tensor_model_parallel_world_size)
    Base = mod.ModelOptNvFp4FusedMoE
    Cfg = mod.ModelOptNvFp4Config

    # ---- 1. keep hybrid_bit_map alive through from_config ----
    _orig_fc = getattr(Cfg.from_config, "__func__", Cfg.from_config)
    def _from_config(cls, config):
        obj = _orig_fc(cls, config)
        hbm = None
        if isinstance(config, dict):
            hbm = config.get("hybrid_bit_map")
            if hbm is None and isinstance(config.get("quantization"), dict):
                hbm = config["quantization"].get("hybrid_bit_map")
        try:
            obj.hybrid_bit_map = hbm
            print(f"[hybrid_loader] hybrid_bit_map stashed: "
                  f"{len(hbm) if hbm else 0} layers", flush=True)
        except Exception as e:
            print("[hybrid_loader] stash failed:", e, flush=True)
        return obj
    Cfg.from_config = classmethod(_from_config)

    class HybridNvFp4MoE(Base):
        def maybe_make_prepare_finalize(self, *a, **k):
            return None  # we own the forward via apply(); no external prepare/finalize

        def _bits(self, layer):
            hbm = _load_hbm(self.quant_config)
            pfx = getattr(layer, "prefix", None)
            if pfx is None:
                pfx = getattr(layer, "layer_name", "") or ""
            m = re.search(r"layers\.(\d+)\b", str(pfx))
            b = hbm.get(str(int(m.group(1)))) if (hbm and m) else None
            if b is None:
                print(f"[hybrid][MISS] pfx={pfx!r} hbm_len={len(hbm) if hbm else 0} "
                      f"idx={m.group(1) if m else None}", flush=True)
            return b

        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype, **extra):
            bits = self._bits(layer)
            if bits is None:
                # Non-hybrid MoE (the MTP/nextn layer) — its experts are uniform NVFP4.
                # Route it through OUR path as all-kept (bits=4) so it uses our Triton NVFP4
                # kernel + our weight_loader. (Falling to super() uses the stock MoE loader,
                # whose weight-scale quant_method check at routed_experts.py:906 rejects it
                # under our method -> the MTP boot crash.)
                bits = [4] * num_experts
            H, I = hidden_size, intermediate_size_per_partition
            gs = self.quant_config.group_size
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            kept = [e for e, b in enumerate(bits) if b == 4]
            dem = [e for e, b in enumerate(bits) if b == 3]
            Kn, Km = len(kept), len(dem)
            layer.hyb = {"remap": {**{e: (0, i) for i, e in enumerate(kept)},
                                   **{e: (1, i) for i, e in enumerate(dem)}},
                         "H": H, "I": I, "gs": gs, "E": num_experts}
            _pfx = getattr(layer, "prefix", None) or getattr(layer, "layer_name", "?")
            layer.hyb["lname"] = str(_pfx)
            print(f"[hybrid] {_pfx}: {Kn}NVFP4 + {Km}NF3 (tp{tp_rank}/{tp_size} I={I})", flush=True)

            def wl(param, loaded, name_mapped=None, *, shard_id=None,
                   expert_id=None, return_success=False, **kw):
                nm = name_mapped or ""
                if "input_scale" in nm:            # W4A16 reference -> unused
                    return True
                grp, li = layer.hyb["remap"][int(expert_id)]
                fam = "w13" if "w13_" in nm else "w2"
                sh = shard_id
                if "weight_scale_2" in nm:         # NVFP4 per-tensor global (kept only)
                    tgt = getattr(layer, f"{fam}_weight_scale_2")
                    if fam == "w13":
                        tgt.data[li, 0 if sh == "w1" else 1] = loaded.reshape(()).to(tgt.dtype)
                    else:
                        tgt.data[li] = loaded.reshape(()).to(tgt.dtype)
                    return True
                # TP shard the block-quantized 2D tensor (gate/up -> dim0, down -> dim1)
                if tp_size > 1 and loaded.ndim >= 2:
                    if sh in ("w1", "w3"):
                        loaded = loaded.chunk(tp_size, 0)[tp_rank]
                    elif sh == "w2":
                        loaded = loaded.chunk(tp_size, 1)[tp_rank]
                if "weight_scale" in nm:           # block scale -> real storage, demux by group
                    tgt = getattr(layer, f"{fam}_nv_s" if grp == 0 else f"{fam}_nf_s")
                elif "weight_packed" in nm:        # NF3 packed weight
                    tgt = getattr(layer, f"{fam}_weight_packed")
                else:                              # plain NVFP4 weight
                    tgt = getattr(layer, f"{fam}_weight")
                d = tgt.data[li]
                if fam == "w13" and sh in ("w1", "w3"):   # gate->top half, up->bottom half
                    half = d.shape[0] // 2
                    d = d[:half] if sh == "w1" else d[half:]
                d.copy_(loaded.reshape(d.shape).to(d.dtype))
                return True

            def P(name, shape, dt=torch.uint8):
                p = torch.nn.Parameter(torch.zeros(shape, dtype=dt,
                                       device=torch.cuda.current_device()), requires_grad=False)
                set_weight_attrs(p, {"weight_loader": wl})
                layer.register_parameter(name, p)

            mk = lambda n: max(n, 1)
            # --- names the stock (prefix-based) mapping produces (routing needs no patch) ---
            P("w13_weight",         (mk(Kn), 2 * I, H // 2))         # NVFP4 weight (kept)
            P("w13_weight_packed",  (mk(Km), 2 * I, H // 8 * 3))     # NF3 packed weight (demoted)
            P("w13_weight_scale",   (1,))                            # dispatcher (routes to *_s)
            P("w13_weight_scale_2", (mk(Kn), 2), torch.float32)      # NVFP4 global (kept)
            P("w13_input_scale",    (1,), torch.float32)             # dispatcher (ignored)
            P("w2_weight",          (mk(Kn), H, I // 2))
            P("w2_weight_packed",   (mk(Km), H, I // 8 * 3))
            P("w2_weight_scale",    (1,))
            P("w2_weight_scale_2",  (mk(Kn),), torch.float32)
            P("w2_input_scale",     (1,), torch.float32)
            # --- real block-scale storage (filled by dispatcher; not mapping-routed) ---
            for nm, sh in [("w13_nv_s", (mk(Kn), 2 * I, H // gs)),
                           ("w13_nf_s", (mk(Km), 2 * I, H // 32)),
                           ("w2_nv_s",  (mk(Kn), H, I // gs)),
                           ("w2_nf_s",  (mk(Km), H, I // 32))]:
                layer.register_parameter(nm, torch.nn.Parameter(
                    torch.zeros(sh, dtype=torch.float8_e4m3fn,
                                device=torch.cuda.current_device()), requires_grad=False))

        def _build_kept_b12x(self, layer):
            """Build a REAL b12x NVFP4 fused-MoE kernel over just the Kn kept experts.
            Reuses the production kernel (graph-safe, sm120) via a cloned num_experts=Kn
            FusedMoEConfig. Validated standalone: correct + skips out-of-range ids + cudagraph-safe.
            apply() remaps topk so kept->[0,Kn), non-kept->Kn (the kernel drops the sentinel)."""
            import dataclasses
            import torch.nn as _nn
            from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
                select_nvfp4_moe_backend, convert_to_nvfp4_moe_kernel_format,
                make_nvfp4_moe_quant_config, make_nvfp4_moe_kernel)
            from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
            from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
            from vllm.model_executor.layers.fused_moe.activation import MoEActivation
            E = layer.hyb["E"]
            dev = layer.w13_weight.device
            remap = layer.hyb["remap"]
            Kn = sum(1 for (grp, li) in remap.values() if grp == 0)
            # Build the kept kernel as a NON-PARALLEL (tp=1) MoE over the PER-RANK intermediate:
            # the weights are already TP-sharded by the weight_loader (ispp per rank), so the
            # kernel must see intermediate_size == ispp and tp=1 (else it mis-indexes full vs
            # sharded -> per-element garbage). The layer's post-apply all-reduce handles TP.
            ispp = self.moe.intermediate_size_per_partition
            kept_moe = dataclasses.replace(
                self.moe, num_experts=Kn, num_local_experts=Kn, num_logical_experts=Kn,
                intermediate_size=ispp,
                moe_parallel_config=FusedMoEParallelConfig.make_no_parallel())
            backend, experts_cls = select_nvfp4_moe_backend(
                config=kept_moe, weight_key=kNvfp4Static, activation_key=None)
            KL = _nn.Module()
            KL.activation = getattr(layer, "activation", MoEActivation.SILU)
            KL.moe_config = kept_moe
            KL.local_num_experts = Kn
            a13 = torch.ones(Kn, 2, device=dev, dtype=torch.float32)
            a2 = torch.ones(Kn, device=dev, dtype=torch.float32)
            w13ks2 = layer.w13_weight_scale_2[:, 0].contiguous()
            conv = convert_to_nvfp4_moe_kernel_format(
                nvfp4_backend=backend, layer=KL,
                w13=layer.w13_weight, w13_scale=layer.w13_nv_s, w13_scale_2=w13ks2, a13_scale=a13,
                w2=layer.w2_weight, w2_scale=layer.w2_nv_s, w2_scale_2=layer.w2_weight_scale_2,
                a2_scale=a2, is_act_and_mul=True, use_a16=True)
            (cw13, cw13s, cw13s2, ca13, cw2, cw2s, cw2s2, ca2) = conv
            for k, v in [("w13_weight", cw13), ("w13_weight_scale", cw13s),
                         ("w13_weight_scale_2", cw13s2), ("w13_input_scale", ca13),
                         ("w2_weight", cw2), ("w2_weight_scale", cw2s),
                         ("w2_weight_scale_2", cw2s2), ("w2_input_scale", ca2)]:
                setattr(KL, k, v)
            qconf = make_nvfp4_moe_quant_config(backend, cw13s, cw2s, cw13s2, cw2s2,
                                                ca13, ca2, use_a16=True)
            kk = make_nvfp4_moe_kernel(qconf, kept_moe, experts_cls, routing_tables=None)
            kk.fused_experts.process_weights_after_loading(KL)
            # Mark this method as owning its modular kernel (supports_internal_mk -> True) so
            # vLLM's post-load maybe_init_modular_kernel() returns early instead of rebuilding
            # the parent kernel from the (freed) standard weight attrs.
            self.moe_kernel = kk
            # remap: global -> kept-local, else sentinel Kn (b12x skips ids >= num_experts)
            nvfp4_remap = torch.full((E,), Kn, dtype=torch.int32, device=dev)
            for g, (grp, li) in remap.items():
                if grp == 0:
                    nvfp4_remap[g] = li
            if _HYBRID_DEBUG and not _b12x_nf3_rt.get("_selftested"):
                _b12x_nf3_rt["_selftested"] = True
                try:
                    from vllm.distributed import get_tensor_model_parallel_rank as _gr
                    _rk = _gr()
                except Exception:
                    _rk = -1
                xt = torch.randn(4, layer.hyb["H"], device=dev, dtype=torch.bfloat16) * 0.03
                ids = (torch.arange(32, device=dev, dtype=torch.int32) % Kn).reshape(4, 8)
                tww = torch.full((4, 8), 0.125, device=dev, dtype=torch.float32)
                o = kk.apply(xt, KL.w13_weight, KL.w2_weight, tww, ids,
                             activation=KL.activation, global_num_experts=Kn,
                             expert_map=None, apply_router_weight_on_input=False,
                             shared_experts=None, shared_experts_input=None)
                torch.cuda.synchronize()
                print(f"[selftest] r{_rk} {getattr(layer, 'prefix', '?')}: "
                      f"out={o.float().norm():.4f} inf={bool(torch.isinf(o).any())} "
                      f"nan={bool(torch.isnan(o).any())} "
                      f"cw13={KL.w13_weight.view(torch.uint8).float().mean():.3f} "
                      f"cw13s2_max={KL.w13_weight_scale_2.float().abs().max():.3e} "
                      f"cw2s2_max={KL.w2_weight_scale_2.float().abs().max():.3e}", flush=True)
            layer.hyb["kept_kernel"] = kk
            layer.hyb["kept_KL"] = KL
            layer.hyb["Kn"] = Kn
            layer.hyb["nvfp4_remap"] = nvfp4_remap
            # CRITICAL: keep the converted tensors alive past this function. b12x's
            # process_weights compacts the source weight to (0,) but its prepared weights
            # VIEW the converted `cw13`/`cw2`; if those locals are freed on return the views
            # dangle -> per-layer garbage (works in a script where they stay in scope).
            layer.hyb["_keepalive"] = conv
            # free the BIG compact kept originals (KL holds the converted copies) -> flat VRAM.
            # keep the tiny *_weight_scale* params (get_fused_moe_quant_config may still read them).
            keep_dbg = _HYBRID_DEBUG and _dbg_first_build[0]   # keep originals for ONE layer only
            if keep_dbg:
                _dbg_first_build[0] = False
            for nm in ("w13_weight", "w2_weight", "w13_nv_s", "w2_nv_s"):
                if not keep_dbg and hasattr(layer, nm):
                    try:
                        delattr(layer, nm)
                    except Exception:
                        setattr(layer, nm, None)
            print(f"[hybrid] {getattr(layer, 'prefix', '?')}: kept b12x kernel built "
                  f"(Kn={Kn}, backend={backend}, kept_ispp={kept_moe.intermediate_size_per_partition}, "
                  f"kept_tp={kept_moe.moe_parallel_config.tp_size}, "
                  f"w13={tuple(KL.w13_weight.shape)})", flush=True)

        def _build_b12x_nf3(self, layer):
            """HYBRID_KEPT=b12x_nf3: drive BOTH tiers through the b12x W4A16
            CuteDSL MoE kernel.

            Object A (kept NVFP4)  -> the PRODUCTION vLLM chain via
              _build_kept_b12x (select_nvfp4_moe_backend -> convert_to_nvfp4_
              moe_kernel_format -> make_nvfp4_moe_kernel, no-parallel clone,
              keepalive). Manual composition through the prepare building
              blocks is numerically WRONG for varying real-range scales
              (harness-proven 2026-07-01: only the production convert chain
              passes vs the dequant reference; probe test_b12x.py TESTS A-D).
            Object B (NF3 192)     -> weight_layout="nf3_2p1"/e4m3_k32 packed
              from the 837 checkpoint planes (chunked over experts) through
              our NF3 kernel format (GPU unit test + in-model norms PASS).
            Launches/buffers are built lazily at first apply() (topk + real max
            m known there; the first forward is vLLM's EAGER profile run, so
            nothing compiles inside CUDA-graph capture)."""
            import torch
            from b12x.moe.fused.w4a16.prepare import (
                PreparedNF3MoeWeights,
                W4A16PackedWeights,
                _make_workspace,
                _nf3_pack_code_experts,
                _nf3_pack_scale_experts,
                _permute_nvfp4_scales,
                _repack_weight,
            )
            import nf3_replan
            hyb = layer.hyb
            E, H, I = hyb["E"], hyb["H"], hyb["I"]
            remap = hyb["remap"]
            dev = layer.w13_weight.device
            Kn = sum(1 for (grp, _li) in remap.values() if grp == 0)
            Km = sum(1 for (grp, _li) in remap.values() if grp == 1)
            if E == 256 and Kn == 64 and Km == 192:
                hyb["_hetero_mapped_state"] = "map_pending"
                hyb["_hetero_mapped_failure"] = None
                try:
                    tier_local_map = torch.tensor(
                        _combined_tier_local_descriptors(remap),
                        dtype=torch.int32, device=dev).contiguous()
                    if (tuple(tier_local_map.shape) != (256,)
                            or tier_local_map.dtype != torch.int32
                            or tier_local_map.device != dev
                            or not tier_local_map.is_contiguous()):
                        raise RuntimeError(
                            "combined tier-local map must be contiguous "
                            "int32[256] on the prepared-weight device")
                    # This per-layer owner outlives capture and every graph replay.
                    hyb["tier_local_map"] = tier_local_map
                    hyb["_hetero_mapped_state"] = "map_ready"
                except Exception as _me:
                    hyb["tier_local_map"] = None
                    detail = str(_me).splitlines() or ["<no detail>"]
                    detail = detail[0]
                    hyb["_hetero_mapped_state"] = "map_failed"
                    hyb["_hetero_mapped_failure"] = (
                        f"{type(_me).__name__}: {detail}")
            emap_a = torch.full((E,), -1, dtype=torch.int32, device=dev)
            emap_b = torch.full((E,), -1, dtype=torch.int32, device=dev)
            for g, (grp, li) in remap.items():
                (emap_a if grp == 0 else emap_b)[g] = li
            fc1_tn, fc2_tn = _B12X_NF3_TILES[1], _B12X_NF3_TILES[3]

            # ---- object B: NF3 -> "nf3_2p1" (chunked over experts) ----
            # (built FIRST: object A's production build frees the kept originals)
            prep_b = None
            if Km > 0:
                chunk = 16   # bound transient VRAM (codes int32 = ~400MB/16 w13 experts)
                w13_planes, w2_planes = [], []
                for e0 in range(0, Km, chunk):
                    codes = nf3_replan.unpack_837_codes(
                        layer.w13_weight_packed[e0:e0 + chunk], H)
                    w13_planes.append(_nf3_pack_code_experts(
                        codes, size_k=H, size_n=2 * I, tile_n=fc1_tn))
                    del codes
                for e0 in range(0, Km, chunk):
                    codes = nf3_replan.unpack_837_codes(
                        layer.w2_weight_packed[e0:e0 + chunk], I)
                    w2_planes.append(_nf3_pack_code_experts(
                        codes, size_k=I, size_n=H, tile_n=fc2_tn))
                    del codes
                w13_nf3 = torch.cat(w13_planes, 0).contiguous(); del w13_planes
                w2_nf3 = torch.cat(w2_planes, 0).contiguous(); del w2_planes
                w13_ns = _nf3_pack_scale_experts(
                    layer.w13_nf_s.float(), size_k=H, size_n=2 * I)
                w2_ns = _nf3_pack_scale_experts(
                    layer.w2_nf_s.float(), size_k=I, size_n=H)
                nf3_global = torch.full(
                    (Km,), 2.0 ** 116, dtype=torch.float32, device=dev)
                prep_b = PreparedNF3MoeWeights(
                    w13=w13_nf3, w13_scale=w13_ns, w13_global_scale=nf3_global,
                    w2=w2_nf3, w2_scale=w2_ns,
                    w2_global_scale=nf3_global.clone(),
                    workspace=_make_workspace(dev),
                    hidden_size=H, intermediate_size=I, num_experts=Km,
                    is_gated=True, params_dtype=torch.bfloat16,
                    fc1_tile_n=fc1_tn, fc2_tile_n=fc2_tn)

            hyb["prepB"] = prep_b
            hyb["prepA"] = None  # filled by the kept build below
            hyb["emap_a"], hyb["emap_b"] = emap_a, emap_b
            hyb["Kn"], hyb["Km"] = Kn, Km
            keep_dbg = _HYBRID_DEBUG and _dbg_first_build[0]
            if keep_dbg:
                _dbg_first_build[0] = False
            # ---- object A: kept NVFP4 -> manual "packed" composition.
            # REHABILITATED 2026-07-01: byte-identical to the stock prepare entry
            # (verified) and PASSES the FIXED (swizzle=False) dequant reference on
            # ranks 0/1/2 at rel 0.006. The earlier condemnation used a broken
            # reference. Produces weight_layout="packed" -> TC-decode launches
            # compile; NO modular kernel / workspace manager in the forward.
            if Kn > 0:
                g13 = layer.w13_weight_scale_2[:Kn, 0].contiguous()
                g2 = layer.w2_weight_scale_2[:Kn].contiguous()
                w13_packed = _repack_weight(
                    layer.w13_weight.contiguous(), size_k=H, size_n=2 * I)
                w2_packed = _repack_weight(
                    layer.w2_weight.contiguous(), size_k=I, size_n=H)
                w13_ps, w13_pg = _permute_nvfp4_scales(
                    layer.w13_nv_s, g13, size_k=H, size_n=2 * I,
                    a_dtype=torch.bfloat16)
                w2_ps, w2_pg = _permute_nvfp4_scales(
                    layer.w2_nv_s, g2, size_k=I, size_n=H,
                    a_dtype=torch.bfloat16)
                hyb["prepA"] = W4A16PackedWeights(
                    w13=w13_packed, w13_scale=w13_ps, w13_global_scale=w13_pg,
                    w2=w2_packed, w2_scale=w2_ps, w2_global_scale=w2_pg,
                    workspace=_make_workspace(dev),
                    hidden_size=H, intermediate_size=I, num_experts=Kn,
                    is_gated=True, params_dtype=torch.bfloat16,
                    source_format="modelopt_nvfp4", w13_layout="w13",
                    weight_layout="packed", scale_format="e4m3_k16")
                if not keep_dbg:
                    for nm in ("w13_weight", "w2_weight", "w13_nv_s", "w2_nv_s"):
                        p_ = getattr(layer, nm, None)
                        if p_ is not None and getattr(p_, "data", None) is not None:
                            p_.data = p_.data.new_empty((0,))
            if not keep_dbg:
                for nm in ("w13_weight_packed", "w2_weight_packed",
                           "w13_nf_s", "w2_nf_s"):
                    p = getattr(layer, nm, None)
                    if p is not None and getattr(p, "data", None) is not None:
                        p.data = p.data.new_empty((0,))
            print(f"[hybrid] {getattr(layer, 'prefix', '?')}: b12x_nf3 built "
                  f"(Kn={Kn} via production chain + Km={Km} nf3_2p1, "
                  f"tiles={_B12X_NF3_TILES})", flush=True)

        def _ensure_b12x_nf3_runtime(self, layer, m, topk):
            """First-apply init: pinned-tile preplanned launches (per object) +
            ONE module-level scratch/buffer set. First apply = vLLM's eager
            profile run at max_num_batched_tokens, so max_m sizes itself to the
            real serving ceiling and nothing compiles during graph capture."""
            import dataclasses
            import torch
            from b12x.moe.fused.w4a16.kernel import compile_w4a16_fused_moe
            from b12x.moe.fused.w4a16.host import (
                make_w4a16_packed_buffers, max_packed_route_slots)
            hyb = layer.hyb
            st = _b12x_nf3_rt
            E, H, I = hyb["E"], hyb["H"], hyb["I"]
            dev = hyb["emap_a"].device
            if st["max_m"] is None:
                st["max_m"] = max(_B12X_NF3_MAX_TOKENS, int(m))
                st["topk"] = int(topk)
            if int(topk) != st["topk"]:
                raise RuntimeError(
                    f"b12x_nf3: topk changed {st['topk']} -> {topk}")
            props = torch.cuda.get_device_properties(dev)
            sms = int(props.multi_processor_count)
            max_shared_mem = int(getattr(
                props, "shared_memory_per_block_optin", 101_376))

            def launches(prepared):
                key = (prepared.num_experts, prepared.weight_layout,
                       prepared.scale_format, st["topk"], st["max_m"], H, I)
                got = st["launches"].get(key)
                if got is not None:
                    return got
                common = dict(
                    hidden_size=H, intermediate_size=I,
                    num_experts=prepared.num_experts, top_k=st["topk"],
                    activation="silu", apply_router_weight_on_input=False,
                    element_dtype="bf16", fast_math=True, sms=sms,
                    max_shared_mem=max_shared_mem,
                    weight_layout=prepared.weight_layout,
                    scale_format=prepared.scale_format,
                    force_tile_config=_B12X_NF3_TILES)
                # Universal packed-route fallback for every m (1..max_m):
                # spec-0, block-64, expert_map, and zero_fc2_output=True.
                # The optional direct launch below is admitted only for mixed
                # layers; runtime selects it only at exact m=4. Uniform E256,
                # prefill, and every other runtime m stay on this launch.
                cap_slots = max_packed_route_slots(st["max_m"] * st["topk"], 64, E)
                pre = compile_w4a16_fused_moe(
                    size_m=st["max_m"], zero_fc2_output=True,
                    moe_block_size=64, max_m_blocks=(cap_slots + 63) // 64,
                    direct_topk_routes=False, tc_decode_fused_sum=False,
                    **common)
                assert (int(pre.fc1_tile_n), int(pre.fc2_tile_n)) == (
                    _B12X_NF3_TILES[1], _B12X_NF3_TILES[3]), "tile pin failed"
                dec = pre
                if _admit_tc_decode_compile(
                        prepared.weight_layout, hyb["Km"] != 0,
                        _HYBRID_TC_DECODE, _HYBRID_NF3_TC_DECODE):
                    # SMALL size_m (8) direct-topk + fused-sum launch at the
                    # production PIN tiles. The op-boundary tile-config
                    # passthrough fix is required: stock re-resolution can swap
                    # PIN->auto and read the 256-pack with the wrong geometry.
                    # The master flag admits mixed E64 packed; the secondary
                    # flag admits mixed E192 nf3_2p1. Corrected real-weight
                    # CUDA-graph m4/r6/D24 for full E192: packed
                    # 131.136-131.968 us vs direct 94.208-95.072 us
                    # (1.379-1.401x); D15 is only 1.045x, hence opt-in.
                    # Run side maps global route IDs through the per-tier LUT
                    # and passes no expert_map. Each tier keeps its own output.
                    try:
                        cand = compile_w4a16_fused_moe(
                            size_m=_B12X_NF3_DECODE_M, zero_fc2_output=False,
                            moe_block_size=8,
                            max_m_blocks=_B12X_NF3_DECODE_M * st["topk"],
                            direct_topk_routes=True, tc_decode_fused_sum=True,
                            **common)  # carries the production tile tuple
                        assert (int(cand.fc1_tile_n), int(cand.fc2_tile_n)) == (
                            _B12X_NF3_TILES[1], _B12X_NF3_TILES[3]), "tc tile pin failed"
                        dec = cand
                        print(f"[hybrid] tc-decode launch armed "
                              f"layout={prepared.weight_layout} "
                              f"experts={prepared.num_experts} "
                              f"tiles={int(cand.fc1_tile_n)},"
                              f"{int(cand.fc2_tile_n)}", flush=True)
                    except Exception as _te:
                        print(f"[hybrid] tc-decode compile failed; packed fallback "
                              f"layout={prepared.weight_layout} "
                              f"experts={prepared.num_experts}: {_te}", flush=True)
                st["launches"][key] = (dec, pre)
                return st["launches"][key]

            if hyb.get("prepA") is not None:
                hyb["launchA"] = launches(hyb["prepA"])
            if hyb.get("prepB") is not None:
                hyb["launchB"] = launches(hyb["prepB"])
            if st["buffers"] is None:
                prep_any = hyb.get("prepA") or hyb.get("prepB")
                if prep_any is None:
                    # uniform-NVFP4 layer (MTP) first: no NF3 object anywhere yet;
                    # the kept kernel manages its own workspace -> no buffers needed.
                    hyb["_rt_ready"] = True
                    return
                buf = make_w4a16_packed_buffers(
                    prep_any, m=st["max_m"], topk=st["topk"],
                    dtype=torch.bfloat16, device=dev, route_num_experts=E)
                # the preplanned prefill launch validates route capacity at
                # moe_block_size=64; the plan's own block choice can be smaller
                # for small max_m -> upsize the route buffers if needed.
                need_slots = max_packed_route_slots(st["max_m"] * st["topk"], 64, E)
                need_blocks = (need_slots + 63) // 64
                if (buf.packed_route_indices.numel() < need_slots
                        or buf.block_expert_ids.numel() < need_blocks):
                    buf = dataclasses.replace(
                        buf,
                        packed_route_indices=torch.empty(
                            (need_slots,), dtype=torch.int32, device=dev),
                        block_expert_ids=torch.empty(
                            (need_blocks,), dtype=torch.int32, device=dev))
                # Finish every fallible allocation before publishing any shared
                # serial state, so an allocation failure leaves a clean retry.
                out_a = buf.output
                out_b = torch.empty_like(buf.output)
                st["buffers"], st["out_a"], st["out_b"] = buf, out_a, out_b
                print(f"[hybrid] b12x_nf3 runtime ready: max_m={st['max_m']} "
                      f"topk={st['topk']} sms={sms}", flush=True)
            hetero_gate = (
                _HYBRID_HETERO_DECODE
                and _HYBRID_TC_DECODE
                and _HYBRID_NF3_TC_DECODE
                and _HYBRID_TIER == "both"
                and not _HYBRID_DEBUG
                and not _HYBRID_PROFILE
                and hyb.get("Km", 0) != 0
            )
            if (hetero_gate and hyb.get("_hetero_direct") is None
                    and not st["hetero_direct_failed"]):
                def hetero_fallback(reason):
                    if not st["hetero_direct_warned"]:
                        st["hetero_direct_warned"] = True
                        print("[hybrid] heterogeneous one-grid unavailable; "
                              f"serial-direct fallback: {reason}", flush=True)
                    st["hetero_direct_failed"] = True

                exact_mixed = (
                    E == 256
                    and H == 6144
                    and I == 512
                    and hyb.get("Kn") == 64
                    and hyb.get("Km") == 192
                    and st["topk"] == 8
                )
                if not exact_mixed:
                    hetero_fallback(
                        "requires m4/topk8/E64-NVFP4/E192-NF3/E256/"
                        f"H6144/I512 (got topk={st['topk']}, "
                        f"E{hyb.get('Kn')}/E{hyb.get('Km')}/E{E}, H{H}/I{I})")
                elif torch.cuda.is_current_stream_capturing():
                    hetero_fallback("resources were not prepared before capture")
                else:
                    try:
                        prep_a, prep_b = hyb["prepA"], hyb["prepB"]
                        prepared_contract = (
                            int(prep_a.num_experts) == 64
                            and prep_a.weight_layout == "packed"
                            and prep_a.scale_format == "e4m3_k16"
                            and int(prep_a.hidden_size) == 6144
                            and int(prep_a.intermediate_size) == 512
                            and int(prep_b.num_experts) == 192
                            and prep_b.weight_layout == "nf3_2p1"
                            and prep_b.scale_format == "e4m3_k32"
                            and int(prep_b.hidden_size) == 6144
                            and int(prep_b.intermediate_size) == 512
                        )
                        if not prepared_contract:
                            raise RuntimeError(
                                "prepared tier layouts do not match the "
                                "E64 packed/e4m3_k16 + E192 nf3_2p1/e4m3_k32 ABI")

                        def prepared_views(prepared):
                            return (
                                prepared.w13.view(torch.int32).view(-1),
                                prepared.w2.view(torch.int32).view(-1),
                                prepared.w13_scale.view(torch.uint8)
                                .view(torch.int32).view(-1),
                                prepared.w2_scale.view(torch.uint8)
                                .view(torch.int32).view(-1),
                                prepared.w13_global_scale.view(-1),
                                prepared.w2_global_scale.view(-1),
                            )

                        weight_views = (
                            *prepared_views(prep_a),
                            *prepared_views(prep_b),
                        )
                        shared = st["hetero_direct"]
                        scratch_diagnostic = None
                        if shared is None:
                            from b12x.moe.fused.w4a16.kernel import (
                                compile_w4a16_hybrid_direct)

                            launch = compile_w4a16_hybrid_direct(
                                size_m=4,
                                hidden_size=H,
                                intermediate_size=I,
                                nv_num_experts=64,
                                nf_num_experts=192,
                                top_k=8,
                                activation="silu",
                                element_dtype="bf16",
                                fast_math=True,
                                sms=sms,
                                max_shared_mem=max_shared_mem,
                                force_tile_config=_B12X_NF3_TILES)
                            actual_tiles = (
                                int(launch.fc1_tile_k),
                                int(launch.fc1_tile_n),
                                int(launch.fc2_tile_k),
                                int(launch.fc2_tile_n),
                            )
                            if (
                                int(launch.size_m) != 4
                                or int(launch.hidden_size) != 6144
                                or int(launch.intermediate_size) != 512
                                or int(launch.nv_num_experts) != 64
                                or int(launch.nf_num_experts) != 192
                                or int(launch.top_k) != 8
                                or actual_tiles != _B12X_NF3_TILES
                                or int(launch.grid_x) != 128
                                or int(launch.cta_threads) != 256
                                or int(launch.shared_memory_bytes) != 45_184
                                or int(launch.route_slots) != 32
                                or int(launch.scratch_elements) <= 0
                                or int(launch.workspace_words) <= 0
                            ):
                                raise RuntimeError(
                                    "compiled heterogeneous launch failed exact "
                                    "geometry/resource admission")
                            if not hasattr(
                                    torch.ops.b12x,
                                    "w4a16_hybrid_direct_launch"):
                                raise RuntimeError(
                                    "b12x heterogeneous custom op is unavailable")
                            scratch_elements = int(launch.scratch_elements)
                            # Serial and one-grid branches are mutually exclusive
                            # in apply(), and every layer launch is ordered on the
                            # current stream.  Borrow the four smallest compatible
                            # retained serial fields with distinct storage instead
                            # of reserving a second 16.09375 MiB scratch pool.
                            borrowed, borrowed_bytes, retained_bytes = (
                                _borrow_one_grid_scratch(
                                    torch, st["buffers"], dev, scratch_elements))
                            dedicated_bytes = (
                                32 * 1024 * 2
                                + 32 * 512 * 2
                                + 2 * scratch_elements * 4)
                            if borrowed_bytes != dedicated_bytes:
                                raise RuntimeError(
                                    "borrowed one-grid byte accounting mismatch")
                            # Workspace remains a dedicated, zeroed owner. Complete
                            # this sole shared CUDA allocation before publication.
                            workspace = torch.zeros(
                                (int(launch.workspace_words),),
                                dtype=torch.int32, device=dev)
                            shared = {
                                "launch": launch,
                                "sms": sms,
                                "max_shared_mem": max_shared_mem,
                                **borrowed,
                                "workspace": workspace,
                            }
                            scratch_diagnostic = (
                                borrowed_bytes, dedicated_bytes, retained_bytes)
                        elif shared["fc1"].device != dev:
                            raise RuntimeError(
                                "shared heterogeneous scratch is on another device")

                        # Keep the output address unique and stable for every
                        # mixed layer in the captured full-model graph.
                        layer_resource = {
                            "weight_views": weight_views,
                            "output": torch.empty(
                                (4, H), dtype=torch.bfloat16, device=dev),
                        }
                        # Both RHS objects and all tensors are complete here;
                        # tuple publication contains no remaining fallible work.
                        hyb["_hetero_direct"], st["hetero_direct"] = (
                            layer_resource, shared)
                        if scratch_diagnostic is not None:
                            borrowed_bytes, dedicated_bytes, retained_bytes = (
                                scratch_diagnostic)
                            print(
                                "[hybrid] heterogeneous one-grid scratch reuse: "
                                f"borrowed_bytes={borrowed_bytes} "
                                f"dedicated_bytes_avoided={dedicated_bytes} "
                                f"retained_source_bytes={retained_bytes} "
                                "ranges=intermediate_cache13[0:65536],"
                                "intermediate_cache2[0:32768],"
                                f"fc1_c_tmp[0:{scratch_elements * 4}],"
                                f"fc2_c_tmp[0:{scratch_elements * 4}]",
                                flush=True)
                    except Exception as _he:
                        detail = str(_he).splitlines()[0]
                        hetero_fallback(
                            f"{type(_he).__name__}: {detail}")
            # The mapped-global-ID specialization is additive. Its shared
            # scratch/output are the already-published local-ID resources, so a
            # mapped-only failure cannot disarm the proven local one-grid path.
            if hyb.get("Km", 0) != 0:
                hyb["_hetero_decode_branch"] = "serial"
            if hyb.get("_hetero_direct") is not None:
                hyb["_hetero_decode_branch"] = "local_one_grid"
                tier_local_map = hyb.get("tier_local_map")
                mapped_gate = (
                    hetero_gate
                    and E == 256
                    and H == 6144
                    and I == 512
                    and hyb.get("Kn") == 64
                    and hyb.get("Km") == 192
                    and st["topk"] == 8
                    and tier_local_map is not None
                    and not st["hetero_mapped_failed"]
                )
                if mapped_gate:
                    try:
                        if (tier_local_map.dtype != torch.int32
                                or tuple(tier_local_map.shape) != (256,)
                                or not tier_local_map.is_contiguous()
                                or tier_local_map.device != dev
                                or tier_local_map.data_ptr() % 16):
                            raise RuntimeError(
                                "combined tier-local map failed the mapped ABI")
                        local_shared = st["hetero_direct"]
                        local_layer_resource = hyb["_hetero_direct"]
                        mapped_buffers = (
                            ("fc1", local_shared["fc1"], torch.bfloat16,
                             32 * 1024, True),
                            ("activated", local_shared["activated"],
                             torch.bfloat16, 32 * 512, True),
                            ("output", local_layer_resource["output"],
                             torch.bfloat16, 4 * 6144, True),
                            ("fc1_c_tmp", local_shared["fc1_c_tmp"],
                             torch.float32, 2_097_152, False),
                            ("fc2_c_tmp", local_shared["fc2_c_tmp"],
                             torch.float32, 2_097_152, False),
                            ("workspace", local_shared["workspace"],
                             torch.int32, 754, False),
                        )
                        for name, tensor, dtype, elements, exact in mapped_buffers:
                            actual = tensor.numel()
                            if (tensor.dtype != dtype
                                    or tensor.device != dev
                                    or not tensor.is_contiguous()
                                    or (actual != elements if exact
                                        else actual < elements)
                                    or tensor.data_ptr() % 16):
                                raise RuntimeError(
                                    f"mapped {name} resource failed exact admission")
                        mapped_shared = st["hetero_mapped"]
                        if mapped_shared is None:
                            from b12x.moe.fused.w4a16.kernel import (
                                compile_w4a16_hybrid_mapped_grid188)

                            mapped_launch = compile_w4a16_hybrid_mapped_grid188(
                                size_m=4,
                                hidden_size=H,
                                intermediate_size=I,
                                nv_num_experts=64,
                                nf_num_experts=192,
                                top_k=8,
                                activation="silu",
                                element_dtype="bf16",
                                fast_math=True,
                                sms=sms,
                                max_shared_mem=max_shared_mem,
                                force_tile_config=_B12X_NF3_TILES)
                            actual_tiles = (
                                int(mapped_launch.fc1_tile_k),
                                int(mapped_launch.fc1_tile_n),
                                int(mapped_launch.fc2_tile_k),
                                int(mapped_launch.fc2_tile_n),
                            )
                            compiled_abi = tuple(mapped_launch.compiled_abi)
                            resource_metadata = dict(
                                mapped_launch.codegen_resource_metadata)
                            try:
                                route_arg = compiled_abi.index(
                                    "global_topk_ids_i32")
                            except ValueError as exc:
                                raise RuntimeError(
                                    "mapped compile ABI lacks global route IDs") from exc
                            try:
                                registers_per_thread = resource_metadata[
                                    "registers_per_thread_actual"]
                                register_bytes_per_cta = resource_metadata[
                                    "register_bytes_per_cta"]
                                register_budget_bytes = resource_metadata[
                                    "register_budget_bytes"]
                                register_capacity_ctas = resource_metadata[
                                    "register_capacity_ctas"]
                            except KeyError as exc:
                                raise RuntimeError(
                                    "mapped compile metadata lacks exact "
                                    "register admission") from exc
                            if any(type(value) is not int for value in (
                                    registers_per_thread,
                                    register_bytes_per_cta,
                                    register_budget_bytes,
                                    register_capacity_ctas)):
                                raise RuntimeError(
                                    "mapped compile register admission is not "
                                    "integer-valued")
                            expected_register_bytes = (
                                int(mapped_launch.registers_per_thread)
                                * int(mapped_launch.cta_threads) * 4)
                            expected_register_capacity = (
                                register_budget_bytes
                                // expected_register_bytes
                                if expected_register_bytes > 0 else 0)
                            if (
                                int(mapped_launch.size_m) != 4
                                or int(mapped_launch.hidden_size) != 6144
                                or int(mapped_launch.intermediate_size) != 512
                                or int(mapped_launch.nv_num_experts) != 64
                                or int(mapped_launch.nf_num_experts) != 192
                                or int(mapped_launch.top_k) != 8
                                or actual_tiles != _B12X_NF3_TILES
                                or int(mapped_launch.grid_x) != 188
                                or int(mapped_launch.cta_threads) != 256
                                or int(mapped_launch.blocks_per_sm) != 1
                                or int(mapped_launch.shared_memory_bytes) != 45_184
                                or int(mapped_launch.workspace_words) != 754
                                or int(mapped_launch.scratch_elements) != 2_097_152
                                or int(mapped_launch.route_slots) != 32
                                or int(mapped_launch.map_slots) != 256
                                or int(mapped_launch.moe_block_size) != 8
                                or int(mapped_launch.abi_version) != 1
                                or not 1 <= int(
                                    mapped_launch.registers_per_thread) <= 255
                                or int(mapped_launch.local_memory_bytes) != 0
                                or int(mapped_launch.spill_store_bytes) != 0
                                or int(mapped_launch.spill_load_bytes) != 0
                                or int(mapped_launch.generated_code_bytes) <= 0
                                or sms != 188
                                or resource_metadata.get("candidate")
                                != "w4a16_hybrid_mapped_grid188"
                                or resource_metadata.get("architecture") != "sm120"
                                or resource_metadata.get("abi_version") != 1
                                or resource_metadata.get("target_sms") != 188
                                or resource_metadata.get("grid_x") != 188
                                or resource_metadata.get("blocks_per_sm") != 1
                                or resource_metadata.get("cta_threads") != 256
                                or resource_metadata.get(
                                    "min_blocks_per_mp") != 1
                                or resource_metadata.get("shared_words") != 11_296
                                or resource_metadata.get(
                                    "workspace_words") != 754
                                or resource_metadata.get(
                                    "scratch_elements") != 2_097_152
                                or resource_metadata.get("route_slots") != 32
                                or resource_metadata.get("map_slots") != 256
                                or resource_metadata.get(
                                    "resident_capacity_ctas") != 188
                                or resource_metadata.get(
                                    "whole_grid_resident") is not True
                                or resource_metadata.get(
                                    "one_cta_per_sm") is not True
                                or registers_per_thread != int(
                                    mapped_launch.registers_per_thread)
                                or resource_metadata.get(
                                    "registers_per_thread") != (
                                        registers_per_thread)
                                or resource_metadata.get(
                                    "local_memory_bytes") != 0
                                or resource_metadata.get(
                                    "spill_store_bytes") != 0
                                or resource_metadata.get(
                                    "spill_load_bytes") != 0
                                or resource_metadata.get(
                                    "generated_code_bytes") != int(
                                        mapped_launch.generated_code_bytes)
                                or register_bytes_per_cta != (
                                    expected_register_bytes)
                                or register_budget_bytes != 255 * 1024
                                or register_capacity_ctas != (
                                    expected_register_capacity)
                                or register_capacity_ctas < 1
                                or resource_metadata.get(
                                    "shared_memory_bytes_exact") != 45_184
                                or resource_metadata.get(
                                    "shared_memory_bytes") != 45_184
                                or resource_metadata.get(
                                    "shared_memory_guard_bytes") != 1536
                                or (45_184 + 1536 > max_shared_mem)
                                or resource_metadata.get(
                                    "task_schedule") != "task=cta+wave*188"
                                or resource_metadata.get(
                                    "fc1_schedule") != "128x1+60x0"
                                or resource_metadata.get(
                                    "fc2_schedule") != "16x5+172x4"
                                or compiled_abi[route_arg:route_arg + 2] != (
                                    "global_topk_ids_i32",
                                    "tier_local_map_i32")
                                or "nv_local_ids_i32" in compiled_abi
                                or "nf_local_ids_i32" in compiled_abi
                            ):
                                raise RuntimeError(
                                    "mapped launch failed exact ABI/resource admission")
                            if not hasattr(
                                    torch.ops.b12x,
                                    "w4a16_hybrid_mapped_grid188_launch"):
                                raise RuntimeError(
                                    "b12x mapped grid188 custom op is unavailable")
                            mapped_shared = {
                                "launch": mapped_launch,
                                "device": dev,
                            }
                        elif mapped_shared["device"] != dev:
                            raise RuntimeError(
                                "mapped heterogeneous launch belongs to another device")
                        # The layer resource owns the immutable map explicitly;
                        # weight views/output and shared scratch remain owned by
                        # the local resource so either branch has stable addresses.
                        mapped_layer_resource = {
                            "tier_local_map": tier_local_map,
                        }
                        hyb["_hetero_mapped"], st["hetero_mapped"] = (
                            mapped_layer_resource, mapped_shared)
                        hyb["_hetero_mapped_state"] = "armed"
                        hyb["_hetero_decode_branch"] = "mapped"
                    except Exception as _hme:
                        detail = str(_hme).splitlines() or ["<no detail>"]
                        detail = detail[0]
                        reason = f"{type(_hme).__name__}: {detail}"
                        st["hetero_mapped_failed"] = True
                        st["hetero_mapped_failure"] = reason
                        hyb["_hetero_mapped_state"] = "demoted"
                        hyb["_hetero_mapped_failure"] = reason
                elif (hetero_gate
                      and E == 256 and H == 6144 and I == 512
                      and hyb.get("Kn") == 64 and hyb.get("Km") == 192
                      and st["topk"] == 8
                      and tier_local_map is not None
                      and st["hetero_mapped_failed"]):
                    hyb["_hetero_mapped_state"] = "globally_demoted"
                    hyb["_hetero_mapped_failure"] = st[
                        "hetero_mapped_failure"]
            hyb["_rt_ready"] = True

        def _apply_b12x_nf3(self, layer, x, topk_weights, topk_ids):
            import torch
            from b12x.moe.fused.w4a16.kernel import run_w4a16_moe
            hyb = layer.hyb
            st = _b12x_nf3_rt
            m = int(x.shape[0])
            topk = int(topk_ids.shape[1])
            if not hyb.get("_rt_ready"):
                self._ensure_b12x_nf3_runtime(layer, m, topk)
            if m > st["max_m"]:
                raise RuntimeError(
                    f"b12x_nf3: m={m} exceeds planned capacity {st['max_m']}; "
                    "set HYBRID_B12X_MAX_TOKENS >= max_num_batched_tokens")
            decode = m == 4
            if _HYBRID_ACT_CAPTURE and m > 8:
                try:
                    from vllm.distributed import get_tensor_model_parallel_rank
                    if get_tensor_model_parallel_rank() == 0:
                        pfx = str(hyb.get("lname") or getattr(layer, "prefix", None)
                                  or id(layer)).replace("/", "_").replace(".", "_")
                        b = _act_store["buf"].setdefault(pfx, {"x": [], "ids": []})
                        if _act_store["flushed"].get(pfx, 0) < 32768:  # cap tokens/layer
                            b["x"].append(x.detach().to(torch.float16).cpu())
                            b["ids"].append(topk_ids.detach().to(torch.int32).cpu())
                            if sum(t.shape[0] for t in b["x"]) >= 8192:
                                import os as _os
                                _os.makedirs(_HYBRID_ACT_CAPTURE, exist_ok=True)
                                fn = f"{_HYBRID_ACT_CAPTURE}/{pfx}.pt"
                                prev = torch.load(fn) if _os.path.exists(fn) else {"x": [], "ids": []}
                                prev["x"].append(torch.cat(b["x"]))
                                prev["ids"].append(torch.cat(b["ids"]))
                                torch.save(prev, fn)
                                _act_store["flushed"][pfx] = _act_store["flushed"].get(pfx, 0) + sum(t.shape[0] for t in b["x"])
                                b["x"], b["ids"] = [], []
                except Exception as _ce:
                    if not _act_store.get("_warned"):
                        _act_store["_warned"] = True
                        print("[hybrid][act-capture] failed:", repr(_ce), flush=True)
            tw = (topk_weights if topk_weights.dtype == torch.float32
                  else topk_weights.float())
            if not tw.is_contiguous():
                tw = tw.contiguous()
            ids_i32 = (topk_ids if topk_ids.dtype == torch.int32
                       else topk_ids.to(torch.int32))
            if not ids_i32.is_contiguous():
                ids_i32 = ids_i32.contiguous()
            buf = st["buffers"]

            def run(prepared, launch_pair, emap, out, buffers=buf):
                use_dec = (_HYBRID_TC_DECODE and decode
                           and launch_pair[0] is not launch_pair[1])
                launch = launch_pair[0] if use_dec else launch_pair[1]
                route_ids = ids_i32
                if use_dec:
                    # direct-topk path: kernel reads FLAT local ids and skips
                    # negatives itself; expert_map must NOT be used. emap doubles
                    # as the global->local LUT (graph-safe gather).
                    route_ids = emap[ids_i32.long()].to(
                        torch.int32).contiguous()
                    em = None
                else:
                    em = emap   # kernel translates global->local + drops -1 (zero=True pairs)
                return run_w4a16_moe(
                    x, prepared, tw, route_ids,
                    activation="silu",
                    intermediate_cache13=buffers.intermediate_cache13,
                    intermediate_cache2=buffers.intermediate_cache2,
                    output=out,
                    fc1_c_tmp=buffers.fc1_c_tmp,
                    fc2_c_tmp=buffers.fc2_c_tmp,
                    packed_route_indices=buffers.packed_route_indices,
                    block_expert_ids=buffers.block_expert_ids,
                    packed_route_count=buffers.packed_route_count,
                    expert_offsets=buffers.expert_offsets,
                    expert_map=em,
                    fused_launch=launch)

            if hyb["Km"] == 0:
                # uniform-NVFP4 layer (MTP/nextn): single tier through OUR launcher
                out = torch.empty((m, hyb["H"]), dtype=x.dtype, device=x.device)
                out = run(hyb["prepA"], hyb["launchA"], hyb["emap_a"], out, buf)
                if _HYBRID_DEBUG and not hyb.get("_dbg"):
                    hyb["_dbg"] = True
                    print(f"[dbg-nf3] {getattr(layer, 'prefix', '?')} m={m} "
                          f"A-only={out.float().norm():.1f}", flush=True)
                return out
            out_a = out_b = None
            hetero = (
                _HYBRID_HETERO_DECODE
                and _HYBRID_TC_DECODE
                and _HYBRID_NF3_TC_DECODE
                and m == 4 and topk == 8
                and tuple(topk_ids.shape) == (4, 8)
                and hyb.get("Kn") == 64 and hyb.get("Km") == 192
                and hyb.get("E") == 256
                and hyb.get("H") == 6144 and hyb.get("I") == 512
                and _HYBRID_TIER == "both"
                and not _HYBRID_DEBUG
                and not _HYBRID_PROFILE
                and x.dtype == torch.bfloat16
                and x.is_contiguous()
                and tw.numel() == 32
                and st.get("hetero_direct") is not None
                and not st["hetero_direct_failed"]
                and hyb.get("_hetero_direct") is not None
                and torch.cuda.is_current_stream_capturing()
            )
            mapped = (
                hetero
                and topk_ids.dtype == torch.int32
                and topk_ids.numel() == 32
                and topk_ids.is_contiguous()
                and topk_ids.is_cuda
                and topk_ids.device == x.device
                and topk_ids.data_ptr() % 16 == 0
                and st.get("hetero_mapped") is not None
                and not st["hetero_mapped_failed"]
                and hyb.get("_hetero_mapped") is not None
            )
            mapped_succeeded = False
            if mapped:
                mapped_shared = st["hetero_mapped"]
                mapped_resource = hyb["_hetero_mapped"]
                shared = st["hetero_direct"]
                layer_resource = hyb["_hetero_direct"]
                launch = mapped_shared["launch"]
                out = layer_resource["output"]
                try:
                    # Pass the router-owned compact storage directly. The
                    # grid188 op consumes its 32 elements as a flat route list.
                    torch.ops.b12x.w4a16_hybrid_mapped_grid188_launch(
                        x,
                        *layer_resource["weight_views"],
                        topk_ids,
                        mapped_resource["tier_local_map"],
                        shared["fc1"],
                        shared["activated"],
                        out,
                        tw,
                        shared["fc1_c_tmp"],
                        shared["fc2_c_tmp"],
                        shared["workspace"],
                        m,
                        int(launch.size_m),
                        int(launch.hidden_size),
                        int(launch.intermediate_size),
                        int(launch.nv_num_experts),
                        int(launch.nf_num_experts),
                        int(launch.top_k),
                        launch.activation,
                        launch.element_dtype,
                        bool(launch.fast_math),
                        int(shared["sms"]),
                        int(shared["max_shared_mem"]),
                        int(launch.fc1_tile_k),
                        int(launch.fc1_tile_n),
                        int(launch.fc2_tile_k),
                        int(launch.fc2_tile_n),
                        int(launch.grid_x),
                        int(torch.cuda.current_stream(x.device).cuda_stream),
                    )
                    mapped_succeeded = True
                except Exception as _hme:
                    detail = str(_hme).splitlines() or ["<no detail>"]
                    detail = detail[0]
                    reason = f"{type(_hme).__name__}: {detail}"
                    st["hetero_mapped_failed"] = True
                    st["hetero_mapped_failure"] = reason
                    hyb["_hetero_mapped_state"] = "runtime_demoted"
                    hyb["_hetero_mapped_failure"] = reason
                    hyb["_hetero_decode_branch"] = "local_one_grid"
            if mapped_succeeded:
                pass
            elif hetero:
                shared = st["hetero_direct"]
                layer_resource = hyb["_hetero_direct"]
                launch = shared["launch"]
                # One normalized global-ID tensor feeds both graph-safe LUT
                # gathers.  The kernel selects NV when nonnegative, otherwise
                # NF, while the flattened route ordinal is the original slot.
                ids64 = ids_i32.long()
                nv_local_ids_i32 = hyb["emap_a"][ids64].reshape(-1)
                nf_local_ids_i32 = hyb["emap_b"][ids64].reshape(-1)
                if not nv_local_ids_i32.is_contiguous():
                    nv_local_ids_i32 = nv_local_ids_i32.contiguous()
                if not nf_local_ids_i32.is_contiguous():
                    nf_local_ids_i32 = nf_local_ids_i32.contiguous()
                out = layer_resource["output"]
                torch.ops.b12x.w4a16_hybrid_direct_launch(
                    x,
                    *layer_resource["weight_views"],
                    nv_local_ids_i32,
                    nf_local_ids_i32,
                    shared["fc1"],
                    shared["activated"],
                    out,
                    tw,
                    shared["fc1_c_tmp"],
                    shared["fc2_c_tmp"],
                    shared["workspace"],
                    m,
                    int(launch.size_m),
                    int(launch.hidden_size),
                    int(launch.intermediate_size),
                    int(launch.nv_num_experts),
                    int(launch.nf_num_experts),
                    int(launch.top_k),
                    launch.activation,
                    launch.element_dtype,
                    bool(launch.fast_math),
                    int(shared["sms"]),
                    int(shared["max_shared_mem"]),
                    int(launch.fc1_tile_k),
                    int(launch.fc1_tile_n),
                    int(launch.fc2_tile_k),
                    int(launch.fc2_tile_n),
                    int(launch.grid_x),
                    int(torch.cuda.current_stream(x.device).cuda_stream),
                )
            elif _HYBRID_TIER == "a":
                out = run(hyb["prepA"], hyb["launchA"], hyb["emap_a"],
                          st["out_a"][:m], buf).clone()
            elif _HYBRID_TIER == "b":
                out = run(hyb["prepB"], hyb["launchB"], hyb["emap_b"],
                          st["out_b"][:m], buf).clone()
            else:
                prof = _HYBRID_PROFILE and m > 64
                if prof:
                    import time as _time
                    _w0 = _time.perf_counter()
                    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                    ev[0].record()
                out_a = run(hyb["prepA"], hyb["launchA"], hyb["emap_a"],
                            st["out_a"][:m], buf)
                if prof:
                    ev[1].record()
                out_b = run(hyb["prepB"], hyb["launchB"], hyb["emap_b"],
                            st["out_b"][:m], buf)
                if prof:
                    ev[2].record()
                    _prof_store["wall_ms"] = _prof_store.get("wall_ms", 0.0) + \
                        (_time.perf_counter() - _w0) * 1000.0
                    _prof_store["pend"].append((m, ev))
                    _prof_store["n"] += 1
                    if _prof_store["n"] % 624 == 0:  # ~8 full 78-layer prefill passes
                        try:
                            torch.cuda.synchronize()
                            for pm, pe in _prof_store["pend"]:
                                _prof_store["a_ms"] += pe[0].elapsed_time(pe[1])
                                _prof_store["b_ms"] += pe[1].elapsed_time(pe[2])
                                _prof_store["rows"] += pm
                            _prof_store["pend"] = []
                            from vllm.distributed import get_tensor_model_parallel_rank
                            if get_tensor_model_parallel_rank() == 0:
                                n = _prof_store["n"]
                                a = _prof_store["a_ms"] / n
                                b = _prof_store["b_ms"] / n
                                mavg = _prof_store["rows"] / n
                                w = _prof_store.get("wall_ms", 0.0) / n
                                print(f"[hybrid-prof] n={n} avg_m={mavg:.0f} "
                                      f"A(nvfp4)={a:.3f}ms B(nf3)={b:.3f}ms "
                                      f"wall={w:.3f}ms /layer-chunk "
                                      f"-> MoE-gpu {78*(a+b):.0f}ms MoE-wall {78*w:.0f}ms "
                                      f"per {mavg:.0f}-tok chunk",
                                      flush=True)
                        except Exception as _pe:
                            print("[hybrid-prof] harvest failed:", repr(_pe), flush=True)
                out = out_a + out_b
            if _HYBRID_DEBUG and not hyb.get("_dbg"):
                hyb["_dbg"] = True
                na = "-" if out_a is None else f"{out_a.float().norm():.1f}"
                nb = "-" if out_b is None else f"{out_b.float().norm():.1f}"
                print(f"[dbg-nf3] {getattr(layer, 'prefix', '?')} m={m} "
                      f"decode={decode} tier={_HYBRID_TIER} x={x.float().norm():.1f} "
                      f"A={na} B={nb} out={out.float().norm():.1f} "
                      f"nan={bool(torch.isnan(out).any())}", flush=True)
                if getattr(layer, "w13_weight", None) is not None and \
                        layer.w13_weight.numel() > 0:
                    try:
                        ra = self._apply_ref(layer, x, topk_weights, topk_ids, 0)
                        rb = self._apply_ref(layer, x, topk_weights, topk_ids, 1)
                        ea = float((out_a.float() - ra.float()).abs().sum()
                                   / (ra.float().abs().sum() + 1e-9))
                        eb = float((out_b.float() - rb.float()).abs().sum()
                                   / (rb.float().abs().sum() + 1e-9))
                        print(f"[dbgcmp-nf3] A_rel={ea:.4f} B_rel={eb:.4f}",
                              flush=True)
                    except Exception as _e:
                        print("[dbgcmp-nf3] ref failed:", repr(_e), flush=True)
            return out

        def process_weights_after_loading(self, layer):
            if not hasattr(layer, "hyb"):
                return super().process_weights_after_loading(layer)
            if _HYBRID_KEPT == "b12x_nf3":
                try:
                    self._build_b12x_nf3(layer)
                    layer.hyb["_b12x_nf3"] = True
                    layer.hyb["_b12x"] = False
                    layer.hyb["_kept_triton"] = False
                    layer.hyb["_fast"] = False
                    return
                except Exception as e:
                    import traceback
                    print("[hybrid] b12x_nf3 build FAILED -> triton/ref "
                          "fallback:", e, flush=True)
                    traceback.print_exc()
                    layer.hyb["_b12x_nf3"] = False
                    # originals are freed only on success -> the stock flow
                    # below still has everything it needs.
            E = layer.hyb["E"]
            dev = layer.w13_weight.device
            # --- NF3 tier: expert map (demoted global id -> local NF3 index, else -1) ---
            emap = torch.full((E,), -1, dtype=torch.int32, device=dev)
            for g, (grp, li) in layer.hyb["remap"].items():
                if grp == 1:
                    emap[g] = li
            layer.hyb["emap_nf3"] = emap
            if _HYBRID_NF3 == "fast":
                try:
                    import nf3_kernel  # noqa: F401 (mounted alongside loader; warm import)
                    layer.hyb["_fast"] = True
                except Exception as e:
                    print("[hybrid] nf3_kernel import failed -> reference:", e, flush=True)
                    layer.hyb["_fast"] = False
            else:
                layer.hyb["_fast"] = False
            # --- NVFP4 kept tier ---
            layer.hyb["_b12x"] = False
            layer.hyb["_kept_triton"] = False
            if _HYBRID_KEPT in ("triton", "b12x_nf3"):
                # Custom Triton NVFP4 grouped-GEMM: reads the checkpoint weights DIRECTLY
                # (no convert/prepare/copy), graph-safe. emap: kept global -> local, else -1.
                # ("b12x_nf3" reaches here ONLY after a failed b12x_nf3 build ->
                #  fall back to the proven triton kept tier; originals intact.)
                kmap = torch.full((E,), -1, dtype=torch.int32, device=dev)
                for gid, (grp, li) in layer.hyb["remap"].items():
                    if grp == 0:
                        kmap[gid] = li
                layer.hyb["emap_nvfp4"] = kmap
                layer.hyb["g13"] = layer.w13_weight_scale_2[:, 0].contiguous()
                layer.hyb["g2"] = layer.w2_weight_scale_2.contiguous()
                try:
                    import nvfp4_kernel  # noqa: F401
                    layer.hyb["_kept_triton"] = True
                except Exception as e:
                    print("[hybrid] nvfp4_kernel import failed -> reference:", e, flush=True)
            elif _HYBRID_KEPT == "b12x":
                try:
                    self._build_kept_b12x(layer)
                    layer.hyb["_b12x"] = True
                except Exception as e:
                    import traceback
                    print("[hybrid] kept b12x build FAILED -> reference NVFP4:", e, flush=True)
                    traceback.print_exc()

        def _apply_ref(self, layer, x, topk_weights, topk_ids, only_grp):
            H, I, gs = layer.hyb["H"], layer.hyb["I"], layer.hyb["gs"]
            remap = layer.hyb["remap"]
            out = torch.zeros_like(x)
            for e in torch.unique(topk_ids).tolist():
                if e < 0 or e not in remap:
                    continue
                grp, li = remap[e]
                if grp != only_grp:
                    continue
                if grp == 0:
                    w13 = _deq_nv(layer.w13_weight[li], layer.w13_nv_s[li], layer.w13_weight_scale_2[li], gs, dequantize_to_dtype)
                    w2 = _deq_nv(layer.w2_weight[li], layer.w2_nv_s[li], layer.w2_weight_scale_2[li], gs, dequantize_to_dtype)
                else:
                    w13 = _unpack_nf3(layer.w13_weight_packed[li], layer.w13_nf_s[li], H)
                    w2 = _unpack_nf3(layer.w2_weight_packed[li], layer.w2_nf_s[li], I)
                sel = (topk_ids == e)
                tok = sel.any(-1)
                if not tok.any():
                    continue
                g, u = (x[tok] @ w13.t()).chunk(2, -1)
                y = (F.silu(g) * u) @ w2.t()
                wgt = (topk_weights * sel).sum(-1)[tok].unsqueeze(-1).to(y.dtype)
                out[tok] += wgt * y
            return out

        def apply(self, layer, x, topk_weights, topk_ids,
                  shared_experts=None, shared_experts_input=None):
            # routed experts only; shared experts + routing handled by the runner.
            if not hasattr(layer, "hyb"):
                return super().apply(layer, x, topk_weights, topk_ids,
                                     shared_experts, shared_experts_input)
            if layer.hyb.get("_b12x_nf3"):
                return self._apply_b12x_nf3(layer, x, topk_weights, topk_ids)
            H, I, E = layer.hyb["H"], layer.hyb["I"], layer.hyb["E"]
            # --- NVFP4 kept tier ---
            if layer.hyb.get("_kept_triton"):
                import nvfp4_kernel
                kept_out = nvfp4_kernel.nvfp4_moe_layer(
                    x, layer.w13_weight, layer.w13_nv_s, layer.hyb["g13"],
                    layer.w2_weight, layer.w2_nv_s, layer.hyb["g2"],
                    topk_ids, topk_weights, layer.hyb["emap_nvfp4"], E, H, I, 16)
            elif layer.hyb.get("_b12x"):
                kk = layer.hyb["kept_kernel"]
                KL = layer.hyb["kept_KL"]
                Kn = layer.hyb["Kn"]
                kept_ids = layer.hyb["nvfp4_remap"][topk_ids]   # kept->[0,Kn), else Kn (b12x skips)
                kept_out = kk.apply(
                    x, KL.w13_weight, KL.w2_weight, topk_weights, kept_ids,
                    activation=KL.activation, global_num_experts=Kn, expert_map=None,
                    apply_router_weight_on_input=False,
                    shared_experts=None, shared_experts_input=None)
            else:
                kept_out = self._apply_ref(layer, x, topk_weights, topk_ids, 0)
            # --- NF3 tier (192 experts) ---
            if layer.hyb.get("_fast"):
                import nf3_kernel
                nf3_out = nf3_kernel.nf3_moe_layer(
                    x, layer.w13_weight_packed, layer.w13_nf_s,
                    layer.w2_weight_packed, layer.w2_nf_s,
                    topk_ids, topk_weights, layer.hyb["emap_nf3"], E, 32, H, I)
            else:
                nf3_out = self._apply_ref(layer, x, topk_weights, topk_ids, 1)
            out = kept_out + nf3_out
            if _HYBRID_DEBUG and not layer.hyb.get("_dbg"):
                layer.hyb["_dbg"] = True
                print(f"[dbg] {getattr(layer, 'prefix', '?')} "
                      f"x={x.float().norm():.1f} kept={kept_out.float().norm():.1f} "
                      f"nf3={nf3_out.float().norm():.1f} out={out.float().norm():.1f} "
                      f"nan={bool(torch.isnan(out).any())}", flush=True)
                if layer.hyb.get("_b12x") and hasattr(layer, "w13_weight"):
                    try:
                        rk = self._apply_ref(layer, x, topk_weights, topk_ids, 0)
                        ratio = float(kept_out.float().norm() / (rk.float().norm() + 1e-9))
                        erel = float((kept_out.float() - rk.float()).abs().sum()
                                     / (rk.float().abs().sum() + 1e-9))
                        print(f"[dbgcmp] {getattr(layer, 'prefix', '?')} "
                              f"b12x={kept_out.float().norm():.3f} ref={rk.float().norm():.3f} "
                              f"ratio={ratio:.3f} elem_rel={erel:.3f}", flush=True)
                    except Exception as _e:
                        print("[dbgcmp] ref failed:", repr(_e), flush=True)
            return out

    mod.ModelOptNvFp4Config.FusedMoEMethodCls = HybridNvFp4MoE
    print("[hybrid_loader] HybridNvFp4MoE installed", flush=True)

    # ---- fp8 non-expert dequant-on-load (weight_scale_fp8 modules are in `ignore` -> bf16) ----
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        if not getattr(DefaultModelLoader, "_hyb_wrapped", False):
            _oga = DefaultModelLoader.get_all_weights
            def get_all_weights(self, model_config, model):
                w = _fp8_ne_transform(_oga(self, model_config, model))
                try:
                    import kv_hadamard
                    w = kv_hadamard.maybe_wrap(w)
                except ImportError:
                    pass
                return w
            DefaultModelLoader.get_all_weights = get_all_weights
            DefaultModelLoader._hyb_wrapped = True
            print("[hybrid_loader] fp8 non-expert dequant-on-load installed", flush=True)
    except Exception as e:
        print("[hybrid_loader] wrap get_all_weights failed:", e, flush=True)

    # ---- 3. HYBRID_MXFP8_NATIVE: serve the F8 ne-tier via online-mxfp8 ----
    # Instead of dequant-to-bf16 (2 B/param resident), excluded LinearBase
    # modules whose disk tensors are F8 get Mxfp8OnlineLinearMethod: loader
    # still feeds them our dequanted bf16, the method re-quantizes e8m0/32 at
    # load (bit-exact round trip vs rev-3 disk) and serves through the B12X
    # fp8 GEMM (VLLM_USE_B12X_FP8_GEMM=1 forces B12xMxfp8LinearKernel).
    # Net: ~-3.65 GiB/GPU weights. Activations on these linears become
    # dynamic-A8 (Festr serving convention) — KLD re-gate before publishing.
    if os.environ.get("HYBRID_MXFP8_NATIVE", "0") == "1":
        try:
            import json as _json
            from vllm.model_executor.layers.quantization.online.mxfp8 import (
                Mxfp8OnlineLinearMethod)
            from vllm.model_executor.layers.linear import (
                LinearBase, UnquantizedLinearMethod)
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                ParallelLMHead)
            _tier_path = os.environ.get(
                "HYBRID_MXFP8_TIER_JSON",
                "/opt/venv/lib/python3.12/site-packages/mxfp8_tier.json")
            _allow = set(_json.load(open(_tier_path))["module_prefixes"])

            def _mx_norm(p):
                i = p.find("layers.")
                return p[i:] if i >= 0 else p

            _base = mod.ModelOptQuantConfigBase
            if not getattr(_base, "_hyb_mxfp8_overlay", False):
                _orig_gqm = _base.get_quant_method
                _n_over = [0]

                def get_quant_method(self, layer, prefix):
                    m = _orig_gqm(self, layer, prefix)
                    if (type(m) is UnquantizedLinearMethod
                            and isinstance(layer, LinearBase)
                            and not isinstance(layer, ParallelLMHead)
                            and _mx_norm(prefix) in _allow):
                        _n_over[0] += 1
                        if _n_over[0] <= 4 or _n_over[0] % 128 == 0:
                            print(f"[hybrid_loader] mxfp8 overlay #{_n_over[0]}: "
                                  f"{prefix}", flush=True)
                        return Mxfp8OnlineLinearMethod()
                    return m

                _base.get_quant_method = get_quant_method
                _base._hyb_mxfp8_overlay = True
                print(f"[hybrid_loader] mxfp8 ne-tier overlay armed "
                      f"({len(_allow)} module prefixes)", flush=True)
        except Exception as e:
            import traceback
            print("[hybrid_loader] mxfp8 overlay FAILED:", e, flush=True)
            traceback.print_exc()


class _Hook(importlib.abc.MetaPathFinder):
    T = "vllm.model_executor.layers.quantization.modelopt"
    def find_spec(self, name, path, target=None):
        if name == self.T and not getattr(self, "_d", False):
            self._d = True
            spec = importlib.util.find_spec(name)
            if spec:
                real_exec = spec.loader.exec_module
                def ex(module):
                    real_exec(module)
                    try:
                        _patch(module)
                    except Exception as e:
                        import traceback
                        print("[hybrid_loader] patch FAILED:", e, flush=True)
                        traceback.print_exc()
                spec.loader.exec_module = ex
            return spec
        return None


if not any(isinstance(f, _Hook) for f in sys.meta_path):
    sys.meta_path.insert(0, _Hook())
    print("[hybrid_loader] import hook armed", flush=True)
