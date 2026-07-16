# Benchmarks

## Test setup

All results were measured on four NVIDIA RTX PRO 6000 Blackwell Workstation 96 GB GPUs (SM120), TP4+DCP4, with an EPYC 7713 host. GPU peer traffic uses a Microchip Switchtec Gen5 P2P switch; systems limited to Gen4 root-port paths should expect slower collective-heavy results.

The served checkpoint is [madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid), bf16-uplift revision: 78 main layers plus one MTP layer, hidden size 6,144, 256 routed experts with top-8 routing, and 75 hybrid MoE layers containing 64 NVFP4 plus 192 NF3 experts.

| Setting | Value |
|---|---|
| Release | v1.4, built on immutable public v1.3 |
| Parallelism | TP4 + DCP4, interleave 1 |
| Speculation | MTP3, probabilistic draft sampling |
| Attention / MoE | `B12X_MLA_SPARSE` / `b12x` |
| KV cache | `nvfp4_ds_mla`, 368 bytes/token/layer |
| KV pool | 2,490 blocks / 637,440 tokens |
| Max model length | 480,000 |
| Max sequences / batched tokens | 8 / 3,072 |
| CUDA graph cap | 32 |
| GPU memory utilization | 0.986 |
| Scheduling | async, chunked prefill, prefix caching |

## Metric definitions

- **Client aggregate tok/s** is total completion tokens divided by the sustained measured interval. It includes request turnover and idle gaps.
- **Per-user active-decode tok/s** is derived from inter-token latency while generation is active. It excludes time to first token and inter-request gaps.
- **Client prefill tok/s** is prompt tokens divided by client-observed time to first token for standalone prefill.

Aggregate and per-user throughput are not interchangeable, even at C1. C8 aggregate throughput is a capacity result, not single-user speed.

Two decode request protocols are shown separately. The deterministic campaign used a fixed deterministic request payload while the serving stack retained probabilistic MTP3 draft sampling. The stochastic campaign used a stochastic request payload. They are characterizations of the same release, not a matched A/B between protocols.

## Current release results

### Exact-32k prefill

| Run | Client prefill tok/s | TTFT |
|---:|---:|---:|
| Public v1.3 reference | **2,131** | 15.377 s |
| v1.4 repeat 1 | **2,128** | 15.398 s |
| v1.4 repeat 2 | **2,128** | 15.400 s |

The release keeps AOT mode but bypasses direct load and save of the serialized standalone-AOT function for this profile. Lower-level compiler caches remain writable and reusable. The two v1.4 cells effectively match the public v1.3 reference.

### Deterministic C1/ctx0 repeats

| Run | Window | Client aggregate tok/s | Per-user active-decode tok/s | Errors |
|---:|---:|---:|---:|---:|
| 1 | 60 s | **108.708** | **114.402** | 0 |
| 2 | 60 s | **105.703** | **112.153** | 0 |
| 3 | 60 s | **106.856** | **112.598** | 0 |
| 4 | 60 s | **105.642** | **111.892** | 0 |
| 5 | 60 s | **105.276** | **111.055** | 0 |
| **Median** | five cells | **105.703** | **112.153** | **0** |

Every cell reached exact effective concurrency 1, was not capacity-limited, and completed without request errors.

### Stochastic serving matrix

| Cell | Client aggregate tok/s | Per-user active-decode tok/s | Status |
|---|---:|---:|---|
| C1 / ctx0 | **102.895** | **107.575** | 0 errors |
| C1 / ctx32k | **93.414** | **101.243** | 0 errors |
| C1 / ctx64k | **89.777** | **97.563** | 0 errors |
| C1 / ctx128k | **89.762** | **98.210** | 0 errors |
| C8 / ctx0 | **318.535** | **36.751** | 0 errors |
| C8 / ctx32k | **307.744** | not retained | 0 errors |
| C8 / ctx64k | **293.774** | not retained | 0 errors |
| C8 / ctx128k | excluded | excluded | capacity-limited by construction |

Five additional stochastic C1/ctx0 repeats measured:

| Run | Client aggregate tok/s | Per-user active-decode tok/s |
|---:|---:|---:|
| 1 | **102.063** | **109.254** |
| 2 | **100.189** | **106.255** |
| 3 | **98.602** | **104.525** |
| 4 | **101.727** | **110.415** |
| 5 | **100.044** | **107.949** |
| **Median** | **100.189** | **107.949** |

All five repeats completed without request errors.

## Optimization-specific validation

### Grid188 heterogeneous W4A16

The exact `M=4` kernel combines both expert tiers, activation, and weighted FC2 accumulation in one 188-CTA grid. It uses global expert IDs directly and falls back to direct128 or serial per-tier launches when exact admission fails.

The historical release-reference cell for the mapped Grid188 path was:

| Cell | Window | Client aggregate tok/s | Per-user active-decode tok/s | Server generation tok/s | Acceptance | Errors |
|---|---:|---:|---:|---:|---:|---:|
| MTP3 probabilistic, C1/ctx0 | 60 s | **98.4165** | **105.0030** | **98.4048** | **0.5766** | 0 |

The cell reached effective concurrency 1, was not underfilled, and was not capacity-limited. It is a release reference rather than a current-image lineage attestation.

### Speculator upper bound and DCP rank cache

The speculator supplies a CPU-resident optimistic sequence-length bound to each draft-step metadata rebuild. The DCP helper reuses a constant rank-offset tensor instead of rebuilding it on CUDA. Together they remove two recurring synchronization paths without changing partition arithmetic.

| Validation cell | Result |
|---|---:|
| C1/ctx0 per-user, five 60 s cells | 110.35 / 110.87 / 110.70 / 110.32 / 113.56; median **110.70 tok/s** |
| C1/ctx32k per-user | **99.98 tok/s** |
| C2/ctx32k aggregate | **130.80 tok/s** |
| C8/ctx32k aggregate | **271.95 tok/s** |

These were adopted matched per-cell validations. They should not be described as a formally passed six-cell promotion matrix because the orchestration validity rule rejected one faster sequential-turnover cell despite a healthy server, zero errors, and no queue.

### MXFP8 no-init allocation

The online quantizer now skips two unity fills for scale buffers that it immediately overwrites for every logical row consumed by GEMM.

| Validation cell | Client aggregate tok/s | Per-user active-decode tok/s | Acceptance |
|---|---:|---:|---:|
| C1/ctx0, five 60 s cells | 105.623–108.413; median **107.688** | 111.695–115.019; median **113.629** | 0.66–0.83 |
| C1/ctx32k | not retained | **105.97** | not retained |
| C8/ctx32k | **278.94** | not retained | not retained |

The five displayed C1 per-user observations were 113.63, 113.05, 115.02, 113.84, and 111.70 tok/s. Missing metric fields were not reconstructed.

## Interpretation limits

- No current result establishes single-user throughput above 125 tok/s. Larger C2/C8 values are aggregate capacity measurements.
- Aggregate throughput must not be compared with per-user active-decode throughput.
- Deterministic and stochastic request payloads must not be compared as a performance A/B.
- C8/128k is excluded because the requested working set exceeds the 637,440-token pool.
- The AOT setting does not force fully cold compilation; lower-level caches remain reusable.
- MXFP8 no-init leaves only unused physical M-tail padding unspecified. Logical scales are overwritten before GEMM reads them.

The raw benchmark JSON, launch attestations, and repeated-run records are retained by the maintainers outside the repository because of size and machine-specific metadata. They are available on request through a public issue.
