# vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv

Ready-to-run vLLM+b12x image for [madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid)
(full 753B, all 256 experts, no pruning) on **4× 96 GB SM120 GPUs** (RTX PRO 6000 Blackwell) — with
**low-bit MLA KV-cache formats** that stock stacks don't have:

| `--kv-cache-dtype` | Bytes/token/layer | Measured pool (DCP4, util .972) |
|---|---|---|
| `fp8` | 656 | ~430k tokens |
| `nvfp4_ds_mla` (4-bit) | 432 | ~590k tokens |
| **`nf3_ds_mla` (3-bit NoPE + fp8 RoPE)** | **304** | **781,740 tokens → 750k max context** |

Retrieval validated: needle at **720k depth** (nf3, DCP4) and 300k (DCP2). LAVD ledger-consistency:
10/10 within tolerance (6 exact). Logprob drift of nf3 vs fp8 KV: mean |Δlogp| 0.0188 on matched tokens —
under the 0.0195 bar that GPQA-validated as neutral for the 4-bit format.

## What's in the image (on top of the 2026-07-06 eldritch cu132 base)

vLLM (`davidsyoung/vllm` @ [`glm52-hybrid-lowbit-kv`](https://github.com/davidsyoung/vllm/tree/glm52-hybrid-lowbit-kv)):
- `nvfp4_ds_mla` + `nf3_ds_mla` KV-cache dtypes (write kernels + cache ops; the 4-bit format is upstream PR [local-inference-lab/vllm#82](https://github.com/local-inference-lab/vllm/pull/82))
- Hybrid a2a/ag_rs DCP dispatch (upstream PR #78 cherry) — a2a for decode-sized batches, ag_rs for prefill
- SM120 sparse-MLA DCP head-chunking; MTP/spec-decode cherries (vllm#47238, #47198, #47381)
- Hybrid runtime loader for the NF3-hybrid checkpoint (mxfp8 overlay, NF3 expert routing)

b12x (`davidsyoung/b12x` @ [`nf3-ds-mla`](https://github.com/davidsyoung/b12x/tree/nf3-ds-mla)):
- `nvfp4_ds_mla` read path + b12x-native CuTe-DSL write kernel
- `nf3_ds_mla` paired read/write kernels (NF3 3-bit NoPE lane + fp8-E4M3 RoPE lane)

> **Benchmarks:** GPQA-Diamond **177/198 (89.39%)** on the 3-bit KV — equal to NVIDIA's full-model NVFP4 reference — plus LAVD context-consistency and 720k-needle results: see [BENCHMARKS.md](BENCHMARKS.md).

## Quickstart

```bash
hf download madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid --local-dir ./glm52-hybrid
MODEL_DIR=./glm52-hybrid docker compose up -d          # max-context profile (750k ctx)
# or: MODEL_DIR=./glm52-hybrid docker compose -f docker-compose.dcp2.yml up -d   # interactive profile
```

## Profiles

| | `docker-compose.yml` (DCP4 max-context) | DCP4 prefill-priority *(not tested — expected)* | `docker-compose.dcp2.yml` (DCP2 interactive) |
|---|---|---|---|
| Max context / KV pool | **750,000 / 781,740** | ~630,000 / ~670,000 *(expected)* | 325,000 / 343,041 |
| Prefill @32k | ~1,350 t/s (TTFT ~24s) | ~1,600-1,700 t/s *(expected)* | **2,397 t/s (TTFT 13.5s)**, flat to 131k |
| Single-stream decode | ~65 t/s (MTP-3) | ~65 t/s | ~63 t/s (MTP-3) |
| Concurrency | 4 (all CUDA-graphed) | 4 | 10 (all CUDA-graphed) |

The prefill-priority variant is `docker-compose.yml` with three edits — **not measured, derived** from the
DCP2 result ÷ the measured DCP2/DCP4 prefill ratio (1.47×):

```yaml
      HYBRID_B12X_MAX_TOKENS: "4096"
      # command: --max-num-batched-tokens 4096  --max-model-len 630000
```

## Lever table — prefill vs context

Every knob that trades one axis against the other, with measured effects on this stack
(4× RTX PRO 6000, TP4, PCIe Gen4):

| Lever | Prefill effect | Context/pool effect | Basis |
|---|---|---|---|
| **DCP 1 → 2 → 4** | 3,276 → 2,597 → 1,766 t/s (~−21-26% per doubling; redundant cross-rank attention, SM-bound) | pool ×1 → ×1.97 → ×3.85 | measured (all three) |
| **mnbt 2048 → 4096** (+`HYBRID_B12X_MAX_TOKENS`) | +~20% prefill (fewer chunked passes over past KV) | −0.66 GiB/GPU profile peak = **−~110k tokens @DCP4, −~55k @DCP2** | measured @DCP2; DCP4 pool cost derived |
| **KV dtype fp8 → nvfp4 → nf3** | fp8→nvfp4 ≈ free; nvfp4→nf3 ≈ −10-16% (chunked re-reads dequant NF3) | 656→432→304 B/tok/layer: pool ×1 → ×~1.5 → ×~2.15 | measured |
| **util .96 → .968 → .972** | none | +~16k tokens per +0.001 util @DCP4 (half @DCP2); risk = graph-capture OOM margin | measured |
| **max-model-len itself** | none | block tables charge against pool: ~0.19 GiB per 50k ml — set ml ≈ pool − 20-30k margin or boot fails with vLLM printing the true ceiling | measured (two boot failures) |
| **capsize / seqs** | none | ≈ none (graphs allocate after pool sizing) — decode-only: capsize must cover seqs×(MTP+1) | measured (A/B) |
| **MTP depth** | none | ~few k tokens (draft KV) | measured |
| **prefix caching** (on) | repeat-prefix TTFT → ~0 | small block overhead | measured |

Rule of thumb: **DCP picks your quadrant, mnbt fine-tunes inside it, KV dtype sets the pool ceiling,
util/ml squeeze the last 5%.** `--max-cudagraph-capture-size` must cover `max_num_seqs × (spec_tokens + 1)`
or high-concurrency decode falls off the graphs. Needs ≥64 GB host RAM; boot ~15 min cold (JIT), ~5 min
warm (cache volume).

### Hardware note — PCIe fabric

Measured numbers come from GPUs attached via a [C-Payne PCIe Gen5 MCIO switch (Microchip Switchtec PM50100)](https://c-payne.com/products/pcie-gen5-mcio-switch-100-lane-microchip-switchtec-pm50100): GPU-to-GPU peer traffic switches at Gen5 even though the EPYC 7713 host uplink is Gen4. The TP/DCP collectives (allreduce, a2a) are fabric-sensitive — plain Gen4 root-port topologies will land somewhat lower on prefill/decode.

Requires the NVIDIA open driver ≥580 and CUDA 13.2-capable runtime. Everything is Apache-2.0 lineage;
model weights are madeby561's (see model card). Base image lineage: local-inference-lab eldritch cu132.
