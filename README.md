# GLM-5.2 low-bit KV serving on four Blackwell GPUs

Ready-to-run vLLM+B12X configuration for
[madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid):
the full 753B model, all 256 experts, TP4+DCP4, on **4× 96 GB SM120 GPUs** such as the RTX PRO 6000 Blackwell.

The default Compose file is the exact production profile measured on 2026-07-12. It uses the compact
368-byte NVFP4 MLA cache and guarded BF16 project-before-merge prefill path from image
`davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.2`.

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

# 3. Build the small ACS preflight image, pull v1.2, and start the server.
docker compose up -d --build

# 4. Cold boot can take about 15 minutes; warm boot is normally about 5 minutes.
docker compose logs -f glm52

# 5. Verify the OpenAI-compatible endpoint.
curl -fsS http://127.0.0.1:5001/v1/models
```

To keep the checkpoint elsewhere:

```bash
MODEL_DIR=/absolute/path/to/glm52-hybrid docker compose up -d --build
```

The API is served on `http://127.0.0.1:5001/v1`, with model name `GLM-5.2`.

## Exact production profile

| Setting | Value |
|---|---|
| Image | `davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.2@sha256:994fb9dfa20ea37544fb5454d076a25edb3947a6553d55c13c2ddea60adbd18d` |
| Tensor parallelism | TP4 |
| Decode context parallelism | DCP4, interleave 1 |
| KV cache | `nvfp4_ds_mla`, 368 bytes/token/layer |
| Attention / MoE | `B12X_MLA_SPARSE` / `b12x` |
| Max model length | 480,000 |
| Max batched tokens | 3,072 |
| Max sequences | 8 |
| CUDA graph capture | 64 |
| Speculative decoding | MTP-3, probabilistic |
| GPU utilization | 0.975 |
| Small DCP transport | B12X one-shot A2A through 64 tokens |
| Large DCP transport | AG/RS above 64 tokens |
| Project-before-merge | BF16, actual prefill rows strictly above 1,024 |
| Dense / DMA paths | split-K turbo, FP8 DMA wire (`ag`) |
| Prefix caching | enabled |

The 1,024-row project-before-merge threshold is deliberately greater than the 64-token CUDA graph
capture size. Decode graphs therefore keep the original merge-then-project path and never capture the
per-call NCCL weight gather.

## Why Compose runs an ACS preflight

PCIe ACS Request Redirect, Completion Redirect, or Upstream Forwarding on any GPU-path bridge can force
peer traffic upstream instead of through the GPU switch. On the measured host that reduced 32k prefill
from approximately 1,680 tok/s to approximately 1,100 tok/s.

`docker-compose.yml` first builds and runs `Dockerfile.acs` as a privileged one-shot service. The guard:

1. walks every bridge in every NVIDIA GPU's complete upstream path;
2. clears only those three redirect bits;
3. preserves Source Validation and all unrelated ACS controls;
4. reads every control word back and fails the Compose dependency if verification fails.

The vLLM service starts only after `acs-preflight` exits successfully. Inspect it with:

```bash
docker compose logs acs-preflight
# Expected on an already-correct four-GPU host:
# ACS redirect clear on 7 GPU-path bridges; changed=0 GPUs=4
```

## Measured results

Hardware: 4× RTX PRO 6000 Blackwell 96 GB, EPYC 7713, TP4+DCP4, Microchip Switchtec Gen5 GPU P2P.
Every cell used the exact Compose profile above and a clean ACS preflight.

| Metric | Result |
|---|---:|
| KV pool | **647,680 tokens** |
| Prefill at 8k | **1,835 tok/s** |
| Prefill at 32k | **1,981 tok/s** |
| Prefill at 64k | **2,036 tok/s** |
| Prefill at 128k | **2,038 tok/s** |
| C1 decode at 8k, 30-second cell | **67.7 tok/s** |
| C8 decode at 8k | **292.2 aggregate tok/s** |
| 128k needle retrieval | **4/4 depths** |

The matched prior profile produced 1,615 / 1,730 / 1,809 / 1,807 prefill tok/s at
8k / 32k / 64k / 128k. The guarded on-demand BF16 path is 12.5–14.5% faster without giving up KV
capacity. A rejected persistent MXFP8 gathered-weight variant reduced the pool to 554,240 tokens and
was slower; it is not present in `v1.2` or this Compose file.

See [BENCHMARKS.md](BENCHMARKS.md) for the comparison and validation details.

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

## Other Compose files

`docker-compose.dcp1.yml` and `docker-compose.dcp2.yml` preserve the older short-context profiles from
`v1.1`. They are not the measured production setup described above. Use the default
`docker-compose.yml` to reproduce the current best balanced configuration.

## Image and source lineage

- Public image: `davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.2`, pinned by Compose to digest `sha256:994fb9dfa20ea37544fb5454d076a25edb3947a6553d55c13c2ddea60adbd18d`.
- Model: [madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid).
- vLLM lineage: `davidsyoung/vllm`, CUDA 13.2 eldritch base, upstream batch-A ports, semantic PR 48196
  port, then guarded on-demand BF16 projection.
- B12X lineage: compact NVFP4 KV, F16-RoPE prefill path, sparse MLA, hybrid NF3/NVFP4 MoE.

The model weights retain their original license. The serving code is Apache-2.0 lineage.
