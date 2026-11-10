# Research brief: ModernBERT-base encoder forward, batch 1, 512 tokens, Torch-TensorRT

## Objective
Minimize milliseconds per forward of `answerdotai/ModernBERT-base` (`ModernBertModel`:
22 layers, hidden 768, 12 heads of 64, GeGLU MLP 1152, RoPE) on one RTX 5090.
Batch 1, a fixed 512-token sequence of public-domain prose ([CLS], the first 510
of 973 text tokens, [SEP]). Output: `last_hidden_state` [1, 512, 768] in FP16.
The masked-LM head is not part of the benchmark. Workload:
`tools/llm/agentic_encoder.py:create_case`, args
`{"model": "answerdotai/ModernBERT-base", "length": 512}`.

## Accuracy limits
The reference is FP32 eager (HF SDPA, no TF32). Every token's hidden vector is
compared with it. A few tokens' final hidden state is ill-conditioned: two of
the 512 here (positions 271 and 287, both " the"). Any perturbation of the
forward moves them to a token cosine anywhere between 0.17 and 0.99. That
includes FP16 rounding, FP16 instead of FP32 GEMM accumulation, a different
attention kernel and, in FP32, a local window one token wider. Recompiling the
same Inductor config gave a smallest cosine of 0.77 to 0.997. The check is
therefore in two parts:

- **Bulk and local (tight).**
  - `rel_l2_error_trimmed <= 0.0075`: relative L2 error with the worst 1% of
    tokens (5) left out.
  - `block_cosine_median_min >= 0.9997`: median token cosine of every 64-token
    block.
- **Outliers (loose).**
  - `outlier_token_fraction <= 0.01`: share of tokens below cosine 0.9.
  - `token_cosine_min >= 0`.
  - `rel_l2_error <= 0.10` over all tokens.
- `token_cosine_mean` and `max_abs_error` are reported only.

Calibration (smoke measurements):

| What | Trimmed rel L2 | Other |
|---|---|---|
| Correct FP16 builds (the TensorRT baseline, eager, Inductor) | 0.0019 - 0.0062 (the largest: eager with `bench.fp16_accumulation`) | block medians >= 0.99995, at most 2 tokens below 0.9, cosine >= 0.17 |
| FP32 with the local window one token narrower or wider | 0.0087 / 0.0090 | **misses** |
| One 64x64 tile of the band mask dropped in every local layer | 0.031 - 0.045 | block median 0.9982 - 0.9989, **misses** |
| BF16 eager | 0.0165 | **misses** (also not in the catalog) |
| No local mask | rel L2 0.62 | **misses** |
| Global RoPE base in local layers | rel L2 0.30 | **misses** |

A build that loses only the ill-conditioned tokens (one or two tokens with a
low cosine while the bulk and block metrics stay tight) is not broken: any
perturbation does that, and the loose outlier limits allow it. The broken
variants in the table miss a tight limit or the overall rel L2.

## Precision policy (read before comparing with PyTorch)
- TensorRT FP16 engines run their GEMMs with **FP16 accumulation** by default
  (`sm80_xmma_gemm_f16f16_f16f16_f16_*` kernels).
- PyTorch accumulates FP16 GEMMs in FP32 unless
  `torch.backends.cuda.matmul.allow_fp16_accumulation` is set (bench key
  `bench.fp16_accumulation`). cuBLAS speed on this GPU:
  - 512x2304x768: 156 TFLOP/s (FP32 accumulation) vs 192 (FP16).
  - 8192-row GEMMs: 169-186 vs 276-292.
- TensorRT tools that change it:
  - `engine.use_fp32_acc`: FP32 casts around every matmul, so every GEMM
    becomes `f16f32`. With `decompose_attention` the attention matmuls are cast
    as well and run unfused in TF32.
  - `engine.build_route=-cask:fp16_acc=off`: 88 of the 110 GEMM launches
    accumulate in FP32; one fused 64x64 GEMM per layer stays FP16. It inserts
    no casts, so attention stays fused.
  - `-cask:fp16_acc=off -cask_fusion:fp16_acc=off`: at 512 tokens it builds
    the same engine as `-cask:fp16_acc=off` alone. At 8192 it moves every GEMM
    to FP32.
- Compare like with like: TensorRT's default FP16 engines against PyTorch with
  `bench.fp16_accumulation`, and TensorRT with FP32 accumulation against
  PyTorch's default. Both PyTorch references are under "Reference points".

## The model's attention
- Layers 0, 3, ..., 21 (8 of them) are global.
- The other 14 are sliding-window layers: token i sees j only if |i - j| <= 64
  (129 of 512 keys). Hugging Face builds a dense [512, 512] boolean band mask
  for them.
- The global layers run mask-free in this workload, as Hugging Face does in
  eager mode. Exported, Hugging Face would add an all-true mask, and TensorRT
  11.4 FP16 engines crash at enqueue when two attention layers read it.

## What bounds the answer
- Work per forward:
  - Linear layers 113 GFLOP.
  - Attention 17.7 GFLOP as computed with the dense band mask (9.3 needed).
    `bandedB` computes B + 128 keys per local query instead of 512 (129 are
    needed): 10.7 / 12.1 / 14.9 GFLOP of attention for B = 64 / 128 / 256.
    Each trial's `structure` reports `linear_gflop`, `attention_gflop` and
    `attention_gflop_needed`.
  - Weights 0.22 GB in FP16 (non-embedding): 0.14 ms at the measured 1.62 TB/s.
    The forward is compute/latency bound, not memory bound (no
    `bytes_per_unit`).
- cuBLAS at 512 rows does 156 TFLOP/s with FP32 accumulation, 192 with FP16.
  The needed work is 0.6-0.8 ms at those rates. Small GEMMs (M=512) do not
  fill 170 SMs, so ~1.0-1.1 ms is a realistic single-engine target.

## What profiling showed (baseline; smoke numbers)
- **TensorRT baseline** (the documented compile: FP16, attention as TensorRT
  attention layers, TensorRT's default FP16-accumulation GEMMs, optimization
  level 3, no CUDA graphs): 1.414 - 1.432 ms in smoke runs, build ~20 s.
  1 engine, 0 PyTorch partitions, 356 layers, 354 kernels, host overhead ~0%.
  - GEMMs ~0.75 ms (52%: 64x64 / 64x128-tile FP16-accumulation xmma kernels).
  - Local attention 0.15 ms (14 `_gemm_mha_v2` calls); global attention
    0.05 ms (8). Per call, a masked local layer (~11 us) costs more than a
    mask-free global one (~6 us), though it needs only 129 of the 512 keys.
  - ~0.25 ms of small memory-bound kernels: QKV split transposes/casts
    0.093 ms (44 calls), LayerNorm + residual 0.092 ms (43), RoPE 0.048 ms (22).
  - Accuracy: trimmed rel L2 0.0049, block median cosine 0.99998, no token
    below cosine 0.9 (smallest 0.91), rel L2 0.020.
- **Inductor max-autotune + FlexAttention + `bench.fp16_accumulation`**, the
  fastest PyTorch configuration (1.143 ms, 273 kernels, 1.117 ms of GPU time):
  - GEMMs: CUTLASS `h16816gemm` 128x64 + 64x64 tiles, 0.675 ms.
  - Flex attention 0.196 ms.
  - Its fused Triton pointwise kernels do the work of most of the casts,
    transposes, LayerNorm and RoPE kernels that TensorRT runs separately.

  Against the TensorRT baseline (both with FP16 accumulation): GEMMs 0.675 vs
  0.75 ms, attention about equal (0.196 vs 0.20 ms), and fused Triton
  kernels in place of most of the baseline's ~0.25 ms of small kernels.

## Tools: what each changes and what it costs
Every TensorRT recipe built while calibrating was one engine with 0 PyTorch
partitions. A build takes ~10-20 s; a trial is ~15-25 s end to end.

Attention (0.20 ms of the baseline, 0.15 ms of it in the 14 local layers):
- `workload.local_attention` (`dense` | `banded128` | `banded64` |
  `banded256`; default `dense`): `dense` masks a full [512, 512] attention with
  the band mask. `bandedB` splits the queries into blocks of B tokens that
  attend only to the B + 128 keys around them, with a small static mask, as
  one batched SDPA. It is exact (unit-tested equal to the dense band-masked
  attention) and changes only the local layers. Smaller blocks compute fewer
  masked scores but run smaller attention batches.
- `graph.fold_attention_masks` (False | True): folds the attention masks into
  engine constants, so TensorRT sees the mask pattern while it builds the
  engine. By default Torch-TensorRT keeps the arange ops that build the masks
  in the engine and computes them on every call. Folded, the dense band mask
  costs length^2 bytes of engine weights (256 KiB here). The global layers
  have no mask.
- `engine.decompose_attention` (False | True): Torch-TensorRT lowers SDPA to
  matmul, masked softmax and matmul instead of a TensorRT attention layer.
  TensorRT can re-fuse that pattern into its MHA kernel, and then it sees the
  band-mask pattern. It changes the attention kernels and their rounding, and
  the engine around them can come out with other GEMM tiles. Combined with
  `engine.use_fp32_acc`, attention runs unfused (see Precision policy).

GEMM precision and tactics (~0.75 ms of the baseline):
- `engine.build_route` (`''` | `'-cask:fp16_acc=off'` |
  `'-cask:fp16_acc=off -cask_fusion:fp16_acc=off'` | `'-tactic:splitk=off'`):
  TensorRT Global Performance Tuning routes (TensorRT 11.1+, not TensorRT-RTX
  or Windows).
  - `-cask:fp16_acc=off`: FP32 accumulation for most GEMMs without casts (see
    Precision policy). It trades GEMM rate for bulk accuracy.
  - `-cask:fp16_acc=off -cask_fusion:fp16_acc=off`: the same engine as
    `-cask:fp16_acc=off` alone at 512 tokens, so a duplicate trial here.
  - `-tactic:splitk=off`: drops split-K GEMM tactics. It gives other GEMM
    tiles in the dense baseline engine; on an engine built with
    `decompose_attention` it is an exact no-op (bitwise-identical outputs).
- `engine.use_fp32_acc` (False | True; applies to FP16 only): FP32 casts
  around every matmul, so every GEMM accumulates in FP32 (see Precision
  policy). It trades GEMM rate for bulk accuracy and adds the casts; with
  `decompose_attention` it also unfuses attention.
- `engine.precision` (`fp16` | `fp32`; default `fp16`): `fp32` runs without
  TF32 and is exact, but gives up FP16 GEMMs. BF16 is not offered: it misses
  the bulk limit (eager 0.0165).
- `engine.optimization_level` (3 | 5 | 4; default 3): TensorRT builder
  optimization level; higher levels search more tactics and take longer to
  build.

Host side (host overhead ~0% in the baseline):
- `runtime.cudagraphs` (False | True): replays the compiled module from a CUDA
  graph, removing per-call launch and host overhead.
- `runtime.pre_allocated_outputs` (False | True): reuses output buffers between
  calls instead of allocating them on each call.

Known failures and duplicates (not in the catalog):
- `engine.tiling_optimization_level` `fast`: the TensorRT 11.4 builder
  segfaults on this model, so the tool is left out.
- Route `-kgen:codegen:cuda_tile=2`: no engine (a Myelin check fails).
- Routes `-cask_fusion:fp16_acc=off` (alone), `-late_h_fusion=off` and
  `-mha_fusion:v2:force_flash_tiled=on`: build the same kernels with identical
  outputs, with and without `decompose_attention`.
- No tool targets the small memory-bound kernels (QKV split transposes/casts,
  LayerNorm + residual, RoPE) directly, and none sets GEMM tiles directly.

The `bench.*` keys select the PyTorch reference configurations below; they are
not tools of the TensorRT search.

## Reference points (PyTorch, same input and FP16; smoke numbers, shared GPU)
- TensorRT baseline (FP16 accumulation): 1.414 - 1.432 ms.
- Best PyTorch with FP16 accumulation (as TensorRT's default engines):
  Inductor max-autotune + FlexAttention + `bench.fp16_accumulation`, 1.143 ms.
- Best PyTorch with FP32 accumulation (PyTorch's default; on the TensorRT side
  `use_fp32_acc`, or mostly with the `-cask:fp16_acc=off` routes): Inductor
  max-autotune + FlexAttention, 1.285 - 1.297 ms (with and without
  coordinate-descent tuning).
- PyTorch eager: 4.80 ms.

All PyTorch configurations measured. `flex` is
`bench.attn="flex_attention"`, `banded128` is
`workload.local_attention="banded128"` with SDPA, `-cdt` is
`bench.inductor={"coordinate_descent_tuning": true}`, `-fp16acc` is
`bench.fp16_accumulation`, and `default`, `max-autotune`,
`max-autotune-no-cudagraphs` and `reduce-overhead` are `bench.mode` values.
All of these compile and warm up inside build, so build s includes
compilation:

| config | ms | trimmed rel L2 | block median cos | tokens < 0.9 | cos min | rel L2 | build s |
|---|---|---|---|---|---|---|---|
| inductor-max-autotune-flex-fp16acc | 1.143 | 0.0039 | 0.99999 | 0.00% | 0.91 | 0.019 | 27 |
| inductor-max-autotune-flex-cdt-fp16acc | 1.147 | 0.0046 | 0.99998 | 0.20% | 0.47 | 0.045 | 27 |
| inductor-max-autotune-flex-cdt | 1.285 | 0.0020 | 0.99999 | 0.00% | 0.92 | 0.018 | 27 |
| inductor-max-autotune-flex | 1.297 | 0.0021 | 1.00000 | 0.00% | 0.91 | 0.020 | 27 |
| inductor-default-flex-fp16acc | 1.308 | 0.0040 | 0.99999 | 0.20% | 0.36 | 0.045 | 11 |
| inductor-max-autotune-banded128-fp16acc | 1.321 | 0.0041 | 0.99999 | 0.00% | 0.91 | 0.020 | 20 |
| inductor-max-autotune-no-cudagraphs-flex | 1.325 | 0.0020 | 0.99999 | 0.00% | 0.94 | 0.016 | 27 |
| inductor-default-flex | 1.438 | 0.0020 | 1.00000 | 0.00% | 0.92 | 0.018 | 12 |
| inductor-max-autotune-banded128 | 1.485 | 0.0020 | 0.99999 | 0.00% | 1.00 | 0.004 | 20 |
| inductor-max-autotune-fp16acc | 1.517 | 0.0047 | 0.99998 | 0.39% | 0.53 | 0.051 | 17 |
| inductor-default-fp16acc | 1.542 | 0.0042 | 0.99999 | 0.20% | 0.77 | 0.031 | 8 |
| inductor-reduce-overhead | 1.670 | 0.0021 | 0.99999 | 0.00% | 0.92 | 0.019 | 8 |
| inductor-max-autotune | 1.672 | 0.0019 | 1.00000 | 0.00% | 1.00 | 0.003 | 17 |
| inductor-max-autotune-cdt | 1.672 | 0.0021 | 1.00000 | 0.00% | 0.97 | 0.013 | 17 |
| inductor-default | 1.693 | 0.0021 | 0.99999 | 0.00% | 0.92 | 0.019 | 8 |
| inductor-max-autotune-no-cudagraphs | 1.699 | 0.0020 | 1.00000 | 0.00% | 0.93 | 0.018 | 17 |
| eager | 4.800 | 0.0026 | 0.99999 | 0.00% | 0.91 | 0.020 | 1 |
| eager-fp16acc | 4.893 | 0.0062 | 0.99995 | 0.39% | 0.17 | 0.070 | 1 |

## How to work
- One hypothesis per trial, motivated by the baseline profile (GEMMs ~0.75 ms,
  attention 0.20 ms, small memory-bound kernels ~0.25 ms, host overhead ~0%).
  Name the kernel or signal it targets and what you expect it to change.
- A precision change (`-cask:fp16_acc=off`, `use_fp32_acc`) trades speed for
  bulk accuracy. Check it against the limits, remembering that the two
  ill-conditioned tokens move with any kernel change.
- The duplicates and no-ops listed under Tools need no trial.
- Builds take ~10-20 s; a trial is ~15-25 s end to end.
- Differences below the session's noise threshold are not real. The GPU is
  shared, so re-measure before believing a small gain. Revisit a rejected
  option only when the incumbent changed in a way that could change its
  effect.
- Stop when no remaining option can plausibly clear the threshold.
