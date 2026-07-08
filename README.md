# vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv

Ready-to-run vLLM+b12x image for [madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid)
(full 753B, all 256 experts, no pruning) on **4× 96 GB SM120 GPUs** (RTX PRO 6000 Blackwell) — with
**low-bit MLA KV-cache formats** that stock stacks don't have:

| `--kv-cache-dtype` | Bytes/token/layer | Pool (DCP4, mnbt 2048, util .96) | Single-stream decode |
|---|---|---|---|
| `fp8` | 656 | ~360k tokens | 68.9 t/s |
| **`nvfp4_ds_mla` (4-bit)** | **432** | **545,906 tokens** | **70.8-73.2 t/s** |
| `nf3_ds_mla` (3-bit NoPE + fp8 RoPE) | 304 | 781,740 (util .972) → 750k max context | ~60 t/s |

**The 4-bit KV dominates fp8 on this stack — faster decode AND +50% pool** (measured single-variable
A/B, 2026-07-08). The 3-bit format buys the maximum context window at a ~15% decode cost.

Retrieval validated: needle at **720k depth** (nf3, DCP4) and 300k (DCP2). LAVD ledger-consistency:
10/10 within tolerance (6 exact). GPQA-Diamond on the 3-bit KV: **177/198 (89.39%)** — equal to
NVIDIA's full-model NVFP4 reference. See [BENCHMARKS.md](BENCHMARKS.md).

## What's in the image (on top of the 2026-07-06 eldritch cu132 base)

vLLM (`davidsyoung/vllm` @ [`glm52-hybrid-lowbit-kv`](https://github.com/davidsyoung/vllm/tree/glm52-hybrid-lowbit-kv)):
- `nvfp4_ds_mla` + `nf3_ds_mla` KV-cache dtypes (write kernels + cache ops; the 4-bit format is upstream PR [local-inference-lab/vllm#82](https://github.com/local-inference-lab/vllm/pull/82))
- Hybrid a2a/ag_rs DCP dispatch (upstream PR #78 cherry) — a2a for decode-sized batches, ag_rs for prefill
- SM120 sparse-MLA DCP head-chunking; MTP/spec-decode cherries (vllm#47238, #47198, #47381)
- Hybrid runtime loader for the NF3-hybrid checkpoint (mxfp8 overlay, NF3 expert routing)

b12x (`davidsyoung/b12x` @ [`nf3-ds-mla`](https://github.com/davidsyoung/b12x/tree/nf3-ds-mla)):
- `nvfp4_ds_mla` read path + b12x-native CuTe-DSL write kernel
- `nf3_ds_mla` paired read/write kernels (NF3 3-bit NoPE lane + fp8-E4M3 RoPE lane)

## Quickstart

```bash
hf download madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid --local-dir ./glm52-hybrid
MODEL_DIR=./glm52-hybrid docker compose up -d          # balanced profile (DCP4, 500k ctx, 73 t/s)
# or: docker compose -f docker-compose.dcp2.yml up -d  # prefill/interactive (~2.4k pf, 260k ctx)
# or: docker compose -f docker-compose.dcp1.yml up -d  # max-prefill (~3.3k pf, 130k ctx)
```

## Profiles

All three composes share the same tuned core (4-bit KV, MTP-5, mnbt 2048, seqs 8, capture 64, util .96,
a2a decode dispatch) and differ **only in DCP** — decode speed is roughly constant; DCP trades KV pool
against prefill speed:

| | `docker-compose.yml` (DCP4, default) | `docker-compose.dcp2.yml` | `docker-compose.dcp1.yml` |
|---|---|---|---|
| Max context / KV pool | **500,000 / 545,906** *(measured)* | ~260,000 / ~277k *(projected)* | ~130,000 / ~140k *(projected)* |
| Prefill @32k | 1,650 t/s *(measured)* | ~2,400 t/s *(projected)* | ~3,300 t/s *(projected)* |
| Single-stream decode | **73.2 t/s @ctx0, 67-70 @32k** *(measured)* | ~66-71 t/s | ~66-73 t/s |
| TTFT @64k | ~40s | ~27s | ~20s |

DCP2/DCP1 numbers are projected from the measured DCP prefill-scaling curve on this stack
(DCP1/2/4 = 3,276 / 2,597 / 1,766 t/s) and pool ratios (×1 / ×1.97 / ×3.85); the DCP4 column is fully
benchmarked (2026-07-08).

**In-place variants of the default profile (both measured):**
- *Prefill-priority:* `--max-num-batched-tokens 4096` while **keeping** `HYBRID_B12X_MAX_TOKENS: "2048"`
  → prefill 1,890 t/s, decode ~68, pool 446,139 (set `--max-model-len` ≤ 430000).
- *Max-context:* `--kv-cache-dtype nf3_ds_mla`, `--max-model-len 750000`, util 0.972, seqs 4,
  capture 16, MTP-3 → pool 781,740, needle-verified @720k, decode ~55-60 t/s. (This is the profile the
  GPQA/LAVD quality results in BENCHMARKS.md were run on.)

## Lever table — prefill vs context

Every knob that trades one axis against the other, with measured effects on this stack
(4× RTX PRO 6000, TP4, PCIe Gen4 host):

| Lever | Prefill effect | Context/pool effect | Decode effect | Basis |
|---|---|---|---|---|
| **DCP 1 → 2 → 4** | 3,276 → 2,597 → 1,766 t/s (~−21-26% per doubling; redundant cross-rank attention, SM-bound) | pool ×1 → ×1.97 → ×3.85 | ≈ flat | measured (all three) |
| **KV dtype fp8 → nvfp4 → nf3** | −5-7% total across the range | 656→432→304 B/tok/layer: pool ×1 → ×1.5 → ×2.15 | **nvfp4 fastest (70.8), fp8 68.9, nf3 60.3** — 3-bit dequant costs grow with context | measured (single-variable, 2026-07-08) |
| **mnbt 2048 → 4096** | +15% (1,650 → 1,900) | −0.66 GiB/GPU profile peak = **−100k pool tokens @DCP4 (measured), −~55k @DCP2** | −2-4 t/s (partly recoverable: keep `HYBRID_B12X_MAX_TOKENS` at 2048) | measured |
| **`HYBRID_B12X_MAX_TOKENS` (MoE workspace)** | ≈ none (double-pass per chunk is free) | none | smaller = slightly faster; **do not go below 2048** (NF3 graph capture breaks) | measured |
| **a2a decode dispatch** (`VLLM_USE_B12X_DCP_A2A=1` + fp8 DMA wire) | ≈ none | none | **+2.4 t/s** | measured |
| **util .96 → .968 → .972** | none | +~16k tokens per +0.001 util @DCP4 (half @DCP2); risk = transient-alloc OOM under large prefills | measured (incl. 2 OOMs @.972) |
| **max-model-len itself** | none | block tables charge against pool: ~0.19 GiB per 50k ml — set ml ≈ pool − 30-45k margin or boot fails with vLLM printing the true ceiling | measured (boot failures) |
| **capsize / seqs** | none | ≈ none (graphs allocate after pool sizing) — capsize must cover seqs×(MTP+1) | coverage cliff only | measured (A/B) |
| **MTP depth 3 → 5** | none | ~few k tokens (draft KV) | +tokens/step (acceptance ~3.4 @MTP-5 on this model rev) | measured |
| **prefix caching** (on) | repeat-prefix TTFT → ~0 | small block overhead | none | measured |

Rule of thumb: **DCP picks your quadrant; nvfp4 KV is the default (it wins both axes vs fp8);
mnbt fine-tunes prefill-vs-pool inside the quadrant; nf3 KV only when you need >550k context.**
`--max-cudagraph-capture-size` must cover `max_num_seqs × (spec_tokens + 1)` or high-concurrency
decode falls off the graphs. Needs ≥64 GB host RAM; boot ~15 min cold (JIT), ~5 min warm (cache volume).

### Hardware note — PCIe fabric

Measured numbers come from GPUs attached via a [C-Payne PCIe Gen5 MCIO switch (Microchip Switchtec PM50100)](https://c-payne.com/products/pcie-gen5-mcio-switch-100-lane-microchip-switchtec-pm50100): GPU-to-GPU peer traffic switches at Gen5 even though the EPYC 7713 host uplink is Gen4. The TP/DCP collectives (allreduce, a2a) are fabric-sensitive — plain Gen4 root-port topologies will land somewhat lower on prefill/decode.

Requires the NVIDIA open driver ≥580 and CUDA 13.2-capable runtime. Everything is Apache-2.0 lineage;
model weights are madeby561's (see model card). Base image lineage: local-inference-lab eldritch cu132.
