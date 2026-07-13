# GLM-5.2 serving on four Blackwell GPUs

Ready-to-run vLLM+B12X configuration for
[madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid):
the full 753B model, all 256 experts, TP4+DCP4, on **4× 96 GB SM120 GPUs** such as the RTX PRO 6000 Blackwell.

The default Compose file is the exact v1.3 production profile measured on 2026-07-13. It adds the
fast647 sparse-MLA workspace path and guarded B12X A2A decode transport to the compact 368-byte
NVFP4 MLA cache and BF16 project-before-merge path inherited from v1.2.

## Requirements

- Linux with Docker Engine and Docker Compose v2.
- Four NVIDIA SM120 GPUs with 96 GB each.
- NVIDIA open driver 580 or newer and a CUDA 13.2-capable runtime.
- Working GPU peer access. The measured host uses a Microchip Switchtec Gen5 P2P switch; ordinary
  Gen4 root-port layouts will be slower for TP/DCP collectives.
- Enough disk for the approximately 331 GB checkpoint and enough host RAM for model loading.

## Quickstart

```bash
# 1. Clone this repository and enter it.
git clone https://github.com/davidsyoung/vllm-glm52.git
cd vllm-glm52

# 2. Download the checkpoint. This is the default MODEL_DIR used by Compose.
hf download madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  --local-dir ./glm52-hybrid

# 3. Pull v1.3 and start the server.
docker compose up -d

# 4. Cold boot can take about 15 minutes; warm boot is normally about 5 minutes.
docker compose logs -f glm52

# 5. Verify the OpenAI-compatible endpoint.
curl -fsS http://127.0.0.1:5001/v1/models
```

To keep the checkpoint elsewhere:

```bash
MODEL_DIR=/absolute/path/to/glm52-hybrid docker compose up -d
```

The API is served on `http://127.0.0.1:5001/v1`, with model name `GLM-5.2`.

## Exact production profile

| Setting | Value |
|---|---|
| Image | `davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.3` |
| Tensor parallelism | TP4 |
| Decode context parallelism | DCP4, interleave 1 |
| KV cache | `nvfp4_ds_mla`, 368 bytes/token/layer |
| Attention / MoE | `B12X_MLA_SPARSE` / `b12x` |
| Max model length | 480,000 |
| Explicit KV allocation | 2,490 blocks / 637,440 tokens |
| Max batched tokens | 3,072 |
| Max sequences | 8 |
| CUDA graph capture | 32 |
| Speculative decoding | MTP-3, probabilistic |
| GPU utilization | 0.986 |
| Small DCP transport | B12X one-shot A2A through 16 rows |
| Large DCP transport | AG/RS above 16 rows |
| Project-before-merge | BF16, actual prefill rows strictly above 1,024 |
| MLA workspace | Persistent aliases, workspace DCP gather, preallocated eager reduce-scatter |
| Dense / DMA paths | split-K turbo, FP8 DMA wire (`ag`) |
| Prefix caching | enabled |

The 1,024-row project-before-merge threshold is deliberately greater than the 32-token CUDA graph
capture size. Decode graphs therefore keep the original merge-then-project path and never capture the
per-call NCCL weight gather.

## Measured results

Hardware: 4× RTX PRO 6000 Blackwell 96 GB, EPYC 7713, TP4+DCP4, Microchip Switchtec Gen5 GPU P2P.
Every cell used the exact Compose profile above on the measured peer-switch configuration.

| Metric | Result |
|---|---:|
| KV pool | **637,440 tokens** |
| Reproduced prefill at 8k, N=20 | **1,987 tok/s** |
| Reproduced prefill at 32k, N=6 | **2,131 tok/s** |
| Reproduced prefill at 64k, N=3 | **2,109 tok/s** |
| Reproduced prefill at 128k, N=2 | **2,141 tok/s** |
| C1 decode at 0 / 32k / 64k / 128k | **79.4 / 71.6 / 69.0 / 71.4 tok/s** |
| C8 aggregate decode at 0 / 32k / 64k | **311.5 / 290.4 / 276.5 tok/s** |
| 128k needle retrieval inherited from v1.2 | **4/4 depths** |

The same exact fast647 profile previously measured 1,987 / 2,108 / 2,117 / 2,136 tok/s; the
2026-07-13 release reproduction landed within 1.1% at every context. A normalized clean-v1.2 control
measured 1,912 / 2,034 / 2,009 / 2,032 tok/s, so the adopted workspace profile improved prefill by
3.9–5.4% while giving up 10,240 KV tokens (1.58%). The 2,490-block cap is required: attempts to use
2,530 or automatically admitted 2,592 blocks with the workspace overlay failed during startup with a
2 MiB CUDA allocation at only 2–3 MiB free.

The isolated guarded-A2A A/B improved C1 decode by 8.7–11.2% at 0–32k versus the matched AG/RS
return control. C8 changed by +3.6% at zero context and −0.8% at 32k. The production cutoff therefore
keeps A2A on small DCP rows and routes larger rows through AG/RS.

See [BENCHMARKS.md](BENCHMARKS.md) for the comparison and validation details.

## What changed in v1.3

- Persistent sparse-MLA query/workspace aliases and direct DCP gather into the borrowed workspace.
- Preallocated eager DCP reduce-scatter buffers across vLLM's collective stack.
- Paged-indexer carry-fold mode paired with the MLA workspace path.
- Guarded B12X A2A for DCP rows up to 16; AG/RS remains the large-row backend.
- Explicit 2,490-block KV allocation and CUDA graph cap 32, matching the reproduced safe profile.
- Clean cutover: all six runtime overlays are copied into the immutable image; no host bind mounts.

## What changed in v1.2

- Compact NVFP4 MLA records: 432 → 368 bytes by storing the 64-value RoPE lane as E4M3 with one FP32
  scale while preserving the 288-byte E2M1 latent/group-scale region.
- Guarded on-demand BF16 project-before-merge for qualifying B12X sparse prefill calls: project each
  DCP partial from 512 to 256 channels before the natural-LSE merge, then exchange the narrower output.
- Call-local gathered `W_UV`: no persistent gathered BF16 or MXFP8 weights and no KV-pool loss.
- Empty-shard sanitization and strict valid-count ownership for exact natural-LSE reduction.
- Projection helper scratch bounded to 144 MiB with production-size chunking.
- CUDA graph safety invariant: prefill threshold must be at least the maximum graph capture size.
- The DSA indexer block-table width fix is baked into the image; no host bind mount is required.

## Image and source lineage

- Public image: `davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.3`, built from the v1.2
  digest plus six source-visible fast647 runtime overlays.
- Base image: `v1.2@sha256:994fb9dfa20ea37544fb5454d076a25edb3947a6553d55c13c2ddea60adbd18d`.
- Model: [madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid).
- vLLM lineage: `davidsyoung/vllm`, CUDA 13.2 eldritch base, upstream batch-A ports, semantic PR 48196
  port, guarded BF16 projection, then the v1.3 workspace/DCP overlays.
- B12X lineage: compact NVFP4 KV, F16-RoPE prefill, sparse MLA, hybrid NF3/NVFP4 MoE, and paged-indexer
  carry-fold.

The model weights retain their original license. The serving code is Apache-2.0 lineage.
