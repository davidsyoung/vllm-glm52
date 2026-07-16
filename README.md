# GLM-5.2 serving on four Blackwell GPUs

Ready-to-run vLLM+B12X configuration for
[madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid):
the full 753B model, all 256 experts, TP4+DCP4, on four 96 GB SM120 GPUs such as the RTX PRO 6000 Blackwell.

The v1.4 image is a clean delta on the immutable public v1.3 release. It keeps the measured fast647 sparse-MLA and guarded-A2A profile, then adds exact-shape heterogeneous W4A16 decode, MTP/DCP synchronization fixes, and lower-overhead MXFP8/runtime memory handling.

## Requirements

- Linux with Docker Engine and Docker Compose v2.
- Four NVIDIA SM120 GPUs with 96 GB each.
- NVIDIA open driver 580 or newer and a CUDA 13.2-capable runtime.
- Working GPU peer access. The measured host uses a Microchip Switchtec Gen5 P2P switch; Gen4 root-port layouts will have slower TP/DCP collectives.
- Approximately 331 GB of checkpoint storage and enough host RAM to load the model.

## Quickstart

```bash
git clone https://github.com/davidsyoung/vllm-glm52.git
cd vllm-glm52

hf download madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  --local-dir ./glm52-hybrid

docker compose up -d
docker compose logs -f glm52
curl -fsS http://127.0.0.1:5001/v1/models
```

To keep the checkpoint elsewhere:

```bash
MODEL_DIR=/absolute/path/to/glm52-hybrid docker compose up -d
```

The OpenAI-compatible endpoint is `http://127.0.0.1:5001/v1`, with model name `GLM-5.2`.

## Default serving profile

| Setting | Value |
|---|---|
| Image | `davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.4` |
| Tensor / decode-context parallelism | TP4 / DCP4, interleave 1 |
| KV cache | `nvfp4_ds_mla`, 368 bytes/token/layer |
| Attention / MoE | `B12X_MLA_SPARSE` / `b12x` |
| Model length | 480,000 tokens |
| KV allocation | 2,490 blocks / 637,440 tokens |
| Max batched tokens / sequences | 3,072 / 8 |
| CUDA graph capture | 32 tokens |
| Speculative decoding | MTP-3, probabilistic draft sampling |
| GPU memory utilization | 0.986 |
| Small / large DCP transport | B12X A2A through 16 rows / AG-RS above 16 rows |
| Project-before-merge | BF16 for actual prefill rows above 1,024 |
| Prefix caching / chunked prefill | enabled / enabled |

## Optimizations

### Exact heterogeneous Grid188 decode

The validated decode shape is `M=4`, top-8 routing, hidden size 6,144, TP-local intermediate size 512, with 64 packed NVFP4 experts and 192 NF3 experts. The primary kernel launches exactly 188 CTAs, one per SM:

1. zero the final output;
2. execute both tiers' FC1 work;
3. cross a whole-grid barrier;
4. apply gated SiLU to the 32 routed rows;
5. cross a second whole-grid barrier; and
6. execute FC2 with route weights and top-k accumulation fused into the output.

The kernel consumes the router's global expert IDs through a 256-entry tier/local-expert map. It bypasses generic route packing and a separate top-k-sum launch. Admission is fail-closed: exact geometry, SM120/188-SM device identity, register use, zero local memory, shared-memory capacity, tensor layout, alignment, and aliases are checked before use.

Fallback order is mapped Grid188, local-ID direct128, then the established serial per-tier path. Unsupported row counts and prefill keep the existing generic implementation. The FC1/FC2 tile tuple is also carried through the registered custom-op boundary so fallback compilation cannot silently select incompatible packed geometry.

`HYBRID_TC_DECODE`, `HYBRID_NF3_TC_DECODE`, and `HYBRID_HETERO_DECODE` default to enabled in both source and image. Setting any one to `0` remains available for diagnosis; normal operation no longer depends on a Compose-only gate.

### MTP and DCP synchronization removal

Two metadata fixes remove recurring host/device serialization:

- The autoregressive speculator computes a CPU-resident optimistic sequence-length upper bound once per proposal and passes it to each draft-step attention-metadata rebuild. The B12X sparse path no longer needs a lazy exact `seq_lens.to("cpu")` in that loop.
- `get_dcp_local_seq_lens` caches the constant `[[dcp_rank]]` tensor by device and rank. Subsequent calls avoid rebuilding the same CUDA tensor and its pageable host-to-device copy.

The sequence-length arithmetic and device-side exact lengths are unchanged.

### MXFP8 allocation cleanup

Online-MXFP8 input quantization now allocates its row and physical scale storage without filling both buffers with unity first. The quantizer overwrites every logical scale consumed by the dense GEMM. Tensor shapes, layouts, quantization math, weights, and logical outputs are unchanged; unused physical M-tail padding remains unspecified.

### Compilation and memory lifecycle

The image keeps AOT compilation enabled but bypasses direct load and save of the serialized standalone-AOT function for this profile. Lower-level compiler caches remain reusable, so this is not a fully cold build on every boot.

After compilation, kernel warmup, CUDA graph capture, and sampler warmup, each GPU worker synchronizes once and releases unused allocator-cache blocks. Live weights, KV cache, graphs, and retained buffers are not freed. `B12X_EMPTY_CACHE_AFTER_WARMUP` defaults on and accepts an explicit false override.

Route-pack kernels keep `live_numel` as a runtime argument instead of creating value-specialized Triton variants. Native small-batch top-k/top-p fallback uses bounded Triton filtering when available. A 144 MiB-bounded BF16 projection helper is included for callers that need chunked DCP projection; the current serving dispatch still uses the established projection path.

## Measured results

| Metric | Result |
|---|---:|
| KV pool | **637,440 tokens** |
| Exact-32k prefill repeats | **2,128 / 2,128 tok/s** |
| Deterministic C1/ctx0, five 60 s cells | **105.703 aggregate / 112.153 per-user median tok/s** |
| Stochastic C1/ctx0 | **102.895 aggregate / 107.575 per-user tok/s** |
| Stochastic C1 aggregate at 32k / 64k / 128k | **93.414 / 89.777 / 89.762 tok/s** |
| Stochastic C8 aggregate at 0 / 32k / 64k | **318.535 / 307.744 / 293.774 tok/s** |

Client aggregate throughput includes request turnover and idle gaps. Per-user throughput is derived from active inter-token latency. The fields are not interchangeable, and deterministic and stochastic request payloads are reported separately rather than compared as an A/B.

See [BENCHMARKS.md](BENCHMARKS.md) for run-level values, metric definitions, and optimization-specific validation.

## Build and source lineage

`Dockerfile` builds v1.4 directly from:

```text
davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.3@sha256:99ae7b28bb7069b9f7a96f75ea815be56266d2cccf7808d4c497340bb8658bd5
```

The image copies and syntax-checks the nine source overlays that differ from public v1.3. Runtime behavior is contained in the image; no host source bind mounts are required.

The model weights retain their original license. The serving code is Apache-2.0 lineage.
