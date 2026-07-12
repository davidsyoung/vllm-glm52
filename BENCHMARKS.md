# Benchmarks

All results measured on **4× NVIDIA RTX PRO 6000 Blackwell Workstation 96 GB (SM120), TP4, no NVLink,
EPYC 7713** — GPUs attached via a
[C-Payne PCIe Gen5 MCIO switch (100-lane, Microchip Switchtec PM50100)](https://c-payne.com/products/pcie-gen5-mcio-switch-100-lane-microchip-switchtec-pm50100),
so GPU↔GPU peer traffic (allreduce/a2a collectives) switches at **Gen5** while the EPYC host uplink is Gen4.
This fabric is load-bearing for the collective-heavy numbers here — GPUs on plain Gen4 host root ports
will see slower TP/DCP collectives. Serving [madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid)
(2026-07-09 bf16-uplift revision). The current production profile uses image
`davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.2`.

The older quality runs below used the DCP2 `nf3_ds_mla` profile. The first section records the
current default `docker-compose.yml` profile.

---

## Current production profile — guarded BF16 project-before-merge

Measured 2026-07-12 with DCP4, MNBT 3,072, MTP-3, utilization 0.975, max length 480k,
and 368-byte `nvfp4_ds_mla`.

| Path | KV pool | 8k prefill | 32k prefill | 64k prefill | 128k prefill |
|---|---:|---:|---:|---:|---:|
| prior production | 647,680 | 1,615 | 1,730 | 1,809 | 1,807 |
| rejected persistent MXFP8 `W_UV` | 554,240 | 1,438 | 1,645 | 1,682 | 1,692 |
| **v1.2 guarded on-demand BF16** | **647,680** | **1,835** | **1,981** | **2,036** | **2,038** |

The adopted path improves matched prefill by 12.5–14.5% without losing KV capacity. It gathers BF16
`W_UV` only for B12X sparse prefill calls above 1,024 actual rows, projects DCP attention partials
from 512 to 256 channels before the natural-LSE merge, and preserves the original decode path.

Additional gates:

- C1 decode at 8k, 30-second sustained cell: **67.7 tok/s**.
- C8 decode at 8k: **292.2 aggregate tok/s**.
- 128k needle retrieval at depths 0.10 / 0.35 / 0.65 / 0.90: **4/4 HIT**.
- Focused route, metadata, natural-LSE, empty-shard, chunking, warmup, and graph-threshold tests:
  **22/22 passed**.
- Runtime errors, CUDA errors, OOMs, and NaNs after validation: **0**.

The persistent MXFP8 gathered-weight implementation was rejected: it consumed 93,440 KV tokens,
regressed decode, was slower in the matched run, and its 3.755–3.795% projection relative-L2 error
exceeded the 1% numerics gate.


---

## GPQA-Diamond — 3-bit KV holds full-model accuracy

**177/198 = 89.39%**, zero API errors, 198/198 completed.

| Model revision | KV cache | Bytes/tok/layer | GPQA-Diamond |
|---|---|---|---|
| rev 1 | `fp8` | 656 | 175/198 — 88.38% |
| rev 1 | `nvfp4_ds_mla` (4-bit) | 432 | 174/198 — 87.88% |
| **bf16-uplift** | **`nf3_ds_mla` (3-bit)** | **304** | **177/198 — 89.39%** |

Reference points (NVIDIA, full GLM-5.2, no 4-card constraint): **FP8 89.52 / NVFP4 89.39**.
This run **equals the full-model NVFP4 reference** while holding the KV cache in less than half
the bytes of fp8. The +2/+3-question spread over the rev-1 baselines is within binomial noise at
n=198 (±4 at 1σ) — the conservative claim is *parity with fp8-KV quality*; the load-bearing finding
is **no measurable quality cost from 3-bit KV**.

Run facts:
- Protocol: 198 questions, concurrency 8, `reasoning_effort: max`, temperature 1.0, max 100k tokens/answer
  ([madeby561/reap-bench](https://github.com/madeby561/reap-bench) `gpqa_bench.py`)
- Wall time 9h26m; 8 requests in flight end-to-end; **zero preemptions** on a 343k-token pool
- Of the 21 misses, 9 were final-letter parse failures ("None"), an artifact of temp-1.0 free-form
  endings that affects all runs in this table equally

## LAVD — long-structured-context consistency

[LAVD](https://github.com/local-inference-lab/llm-inference-bench) (`--test-profile lavd`) embeds a
167-row work ledger (~29k-token prompt) containing planted data-entry errors; the model must keep the
structure consistent, find and repair the errors, and return the final ticket count and hours
(expected `72, 46.0`; NEAR = within ±4). It is the most KV-noise-sensitive quality probe here —
the failure mode it hunts is losing track of corrections buried deep in context.

| Run | Score | Exact rate | In-tolerance |
|---|---|---|---|
| 10 runs @ conc 10 | EXACT 6 / NEAR 4 / FAIL 0 | 60% | 100% |
| 30 runs @ conc 10 | EXACT 19 / NEAR 9 / FAIL 2 | 63% | 93% |
| **Combined (n=40)** | **EXACT 25 / NEAR 13 / FAIL 2** | **62.5%** | **95%** |

- Completions averaged ~14.4k reasoning tokens (p99 ~23k, none hit a cap) — the model reasons at
  length over 3-bit-quantized context without losing the ledger. Zero unparseable answers in 40 runs.
- The 2 failures answered nearly the same wrong pair (`65, 40.75` and `66, 40.75`) — both missed the
  same repair rule, a reasoning slip on one planted error rather than context degradation.
- Side data: per-request decode held ~35 t/s with 10 concurrent streams (~300 t/s system aggregate);
  TTFT averaged 12.1s cold and 3.96s once the shared prompt prefix was cache-resident.

## Long-context retrieval (needle)

| Profile | KV | Needle depth | Result |
|---|---|---|---|
| Historical DCP4 max-context | `nf3_ds_mla` | **720,000 tokens** (50% depth) | HIT (exact string, 718,643-token prompt) |
| Historical DCP2 | `nf3_ds_mla` | 300,000 tokens (50% depth) | HIT |

## Notes

- Speculative decoding (MTP) is lossless — it never changes outputs, only speed. Quality results are
  independent of the `num_speculative_tokens` setting in the serving profile.
- Raw logs/JSON for these runs are kept out of the repo for size; open an issue if you want them.
