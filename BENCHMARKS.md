# Benchmarks

All results measured on **4× NVIDIA RTX PRO 6000 Blackwell Workstation 96 GB (SM120), TP4, PCIe Gen4
(no NVLink), EPYC 7713** serving [madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid)
(2026-07-07 bf16-uplift revision) with image `davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1`.

Serving profile for the quality runs below: the **DCP2 profile** (`docker-compose.dcp2.yml`) with
`--kv-cache-dtype nf3_ds_mla` — the 3-bit KV cache (304 B/token/layer: NF3 NoPE + fp8-E4M3 RoPE).

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
| DCP4 max-context (`docker-compose.yml`) | `nf3_ds_mla` | **720,000 tokens** (50% depth) | HIT (exact string, 718,643-token prompt) |
| DCP2 (`docker-compose.dcp2.yml`) | `nf3_ds_mla` | 300,000 tokens (50% depth) | HIT |

## Notes

- Speculative decoding (MTP) is lossless — it never changes outputs, only speed. Quality results are
  independent of the `num_speculative_tokens` setting in the composes.
- Raw logs/JSON for these runs are kept out of the repo for size; open an issue if you want them.
