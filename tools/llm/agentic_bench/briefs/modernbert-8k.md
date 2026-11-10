# Research brief: ModernBERT-base encoder forward, batch 1, 8192 tokens, Torch-TensorRT

## Objective
Minimize milliseconds per forward of `answerdotai/ModernBERT-base` (`ModernBertModel`:
22 layers, hidden 768, 12 heads of 64, GeGLU MLP 1152, RoPE) on one RTX 5090.
Batch 1, a fixed 8192-token sequence: [CLS], 973 tokens of public-domain prose
repeated 8.4 times, [SEP]. Output: `last_hidden_state` [1, 8192, 768] in FP16.
The masked-LM head is not part of the benchmark. Workload:
`tools/llm/agentic_encoder.py:create_case`, args
`{"model": "answerdotai/ModernBERT-base", "length": 8192}`.

## Accuracy limits
The reference is FP32 eager (HF SDPA, no TF32). Every token's hidden vector is
compared with it. A few tokens' final hidden state is ill-conditioned (" the",
","; 1 to 17 of the 8192 below cosine 0.9 in correct FP16 builds). Any
perturbation of the forward moves them: FP16 rounding, FP16 GEMM accumulation,
other attention kernels, or a local window one token wider in FP32. Which of
them a correct FP16 build loses changes with every kernel choice, so losing a
few of them is not a sign of a wrong output. The check is therefore in two
parts:

- **Bulk and local (tight).**
  - `rel_l2_error_trimmed <= 0.0095`: relative L2 error with the worst 1% of
    tokens (81) left out.
  - `block_cosine_median_min >= 0.9997`: median token cosine of every 64-token
    block.
- **Outliers (loose).**
  - `outlier_token_fraction <= 0.01`: share of tokens below cosine 0.9.
  - `token_cosine_min >= 0`.
  - `rel_l2_error <= 0.10` over all tokens.
- `token_cosine_mean` and `max_abs_error` are reported only.

Measured:

| What | Trimmed rel L2 | Block median | Result |
|---|---|---|---|
| Baseline (dense TensorRT FP16 engine) | 0.0082 | 0.99988 | pass; 0.11% of tokens below 0.9, cosine min 0.20, rel L2 0.031 |
| Correct FP16 builds, TensorRT and PyTorch | 0.0019 - 0.0082 (the baseline at the top) | >= 0.99987 | pass; outliers <= 0.21%, cosine >= 0.199, rel L2 <= 0.040 |
| FP32 with the local window one token narrower or wider | 0.0123 / 0.0109 | | **misses** (bulk) |
| FP32 with one 64x64 tile of the band mask dropped in every local layer | 0.0071 / 0.0035 (passes) | 0.9987 / 0.9994 | **misses** (local) |
| BF16 eager | 0.0186 | 0.99948 | **misses** (also not in the catalog) |
| No local mask | rel L2 0.98 | | **misses** |
| Global RoPE base in the local layers | rel L2 0.38 | | **misses** |

The bulk margin is thin. The dense baseline is at 0.0082 and window+1 at
0.0109. A candidate within ~10% of a limit deserves a repeat run before its
verdict is trusted.

## Precision policy (read before comparing with PyTorch)
- TensorRT FP16 engines run their GEMMs with **FP16 accumulation** by default
  (`sm80_xmma_gemm_f16f16_f16f16_f16_*`). In the baseline the 1.81 TFLOP of
  linear layers take 6.2 ms (~290 TFLOP/s). Whether TensorRT's fused MHA
  kernel accumulates in FP16 internally is not visible from its name.
- cuBLAS on these shapes does 169-186 TFLOP/s with FP32 accumulation (PyTorch's
  default) and 276-292 with FP16 accumulation (`bench.fp16_accumulation`).
- Three catalog options give TensorRT's GEMMs FP32 accumulation (details under
  Tools): `engine.use_fp32_acc` (casts around every matmul) and the build
  routes `-cask:fp16_acc=off` (the GEMMs outside fusions) and
  `-cask:fp16_acc=off -cask_fusion:fp16_acc=off` (all GEMMs).
- The strongest PyTorch configurations (smoke numbers, shared GPU; full table
  under Reference points):
  - PyTorch's default FP32 accumulation: Inductor max-autotune +
    FlexAttention, 19.53 ms. This is also the comparison for a TensorRT recipe
    whose GEMMs all accumulate in FP32.
  - FP16 accumulation, as TensorRT's default GEMMs: Inductor max-autotune +
    FlexAttention + `bench.fp16_accumulation`, 17.68 ms with coordinate-descent
    tuning, 17.70 ms without.

Kernel by kernel at 8192 tokens (the TensorRT figures are the baseline's):
- Global attention: TensorRT's fused MHA takes 6.04 ms for the 8 global layers
  (~273 TFLOP/s) vs 7.97 ms for FlexAttention or flash in PyTorch.
- Local attention: FlexAttention in PyTorch takes 0.77 ms for the 14 local
  layers; the baseline's dense masked MHA takes 13.56 ms.
- GEMMs: TensorRT 6.2 ms vs 6.9 ms in PyTorch (CUTLASS `h16816`, FP16
  accumulation) or ~8.5 ms (Triton, FP32 accumulation).

## The model's attention
- Layers 0, 3, ..., 21 (8 of them) are global.
- The other 14 are sliding-window ("local") layers: token i sees j only if
  |i - j| <= 64 (129 keys). Hugging Face builds a dense [8192, 8192] boolean
  band mask for them.
- The global layers run mask-free in this workload, as Hugging Face does in
  eager mode. Exported, Hugging Face would add an all-true 8192x8192 mask.
  TensorRT 11.4 FP16 engines crash at enqueue when two attention layers read
  it, so the workload drops it; that is exact.

## What bounds the answer
- Work per forward:
  - Linear layers 1.81 TFLOP.
  - Attention 4.54 TFLOP as the baseline computes it (dense local layers);
    1.69 TFLOP is needed (global 1.65 + local 0.04).
  - Needed total: 3.50 TFLOP.
  - Each trial's structure reports `linear_gflop`, `attention_gflop` (what the
    graph's attention computes: only `workload.local_attention` changes it,
    tiles a kernel skips are not counted) and `attention_gflop_needed`.
- Rates on this GPU:
  - TensorRT fused MHA ~273 TFLOP/s; TensorRT FP16-accumulation GEMMs
    ~290 TFLOP/s.
  - Flash SDPA and FlexAttention ~207 TFLOP/s.
  - cuBLAS GEMMs 169-186 TFLOP/s (FP32 accumulation) or 276-292 (FP16).
  - DRAM read 1.62 TB/s. Memory is not the bound: weights 0.3 GB, no
    `bytes_per_unit`.
- Floor with TensorRT's FP16-accumulation GEMMs and its MHA: 6.2 + 6.2 =
  ~12.4 ms (the 1.81 TFLOP of GEMMs at ~290 TFLOP/s plus the needed 1.69 TFLOP
  of attention at ~273 TFLOP/s). With FP32-accumulation GEMMs: ~16 ms. The
  baseline (29.24 ms) takes 2.4x the FP16 floor (roofline fraction ~42%).

## The baseline and its profile (smoke numbers)
The baseline is the plain Torch-TensorRT compile in FP16: `torch.export` and
`torch_tensorrt.dynamo.compile` with default settings (`min_block_size` 5,
builder optimization level 3, attention as TensorRT attention layers, GEMMs
with TensorRT's default FP16 accumulation), no CUDA graphs. Every catalog tool
is at its default.

- **29.24 ms:** 1 engine, 356 layers, 354 kernels, host overhead 4%, no syncs.
  - Local (masked) attention 13.56 ms: 46%, 14 `_gemm_mha_v2` calls that
    compute all 8192x8192 scores.
  - Global attention 6.04 ms (8 calls).
  - GEMMs 6.23 ms: 128x128 tiles x66 and 256x128 tiles x44, FP16
    accumulation.
  - Small kernels ~1.4 ms: QKV split transposes/casts 0.62 ms, LayerNorm +
    residual 0.51 ms, RoPE ~0.4 ms.
- The local layers compute 2.89 TFLOP of attention where 0.04 TFLOP is needed.
  Global attention and the GEMMs already run at about the best rates measured
  for those kernels on this GPU (see the rates above).

## Tools: what each changes and what it costs
No catalog option splits the model: every build is one TensorRT engine with no
PyTorch partitions. Builds take 10-30 s; a trial is ~20-30 s end to end.

Attention (they target the attention kernels; in the baseline the local
layers' attention is the dominant cost):
- `engine.decompose_attention` (False | True): Torch-TensorRT decomposes
  `scaled_dot_product_attention` into matmul, masked softmax and matmul
  instead of emitting a TensorRT attention layer. TensorRT re-fuses the
  pattern into its MHA kernel and then sees the band-mask pattern; Myelin can
  skip fully masked tiles. It changes the attention kernels and their
  rounding.
- `workload.local_attention` (`dense` | `banded64` | `banded128` |
  `banded256`): the exact block-banded rewrite of the 14 local layers. Each
  block of B queries attends only to the B + 128 keys around it, with a small
  static mask, instead of all 8192 keys under the dense band mask. The local
  layers' computed attention work drops 21-43x (B = 256 to 64), and the total
  computed attention from 4.54 TFLOP to 1.72 / 1.74 / 1.78 TFLOP for B = 64 /
  128 / 256. Smaller blocks compute fewer masked scores but run smaller
  attention batches. The rewrite adds kernels that gather each block's K/V
  window (TensorRT fuses them with RoPE). `dense` is the baseline.
- `graph.fold_attention_masks` (False | True): by default Torch-TensorRT keeps
  the arange ops that build the attention masks inside the engine and computes
  the masks on every call. Folding makes them engine constants, so TensorRT
  sees the mask pattern while it builds the engine; the dense band mask then
  costs 8192^2 bytes (64 MiB) of engine weights.

Precision:
- `engine.precision` (`fp16` | `fp32`): `fp32` runs without TF32 and is exact
  against the reference, but gives up the FP16 GEMM and MHA rates above. BF16
  is not offered: it misses the bulk limit (see Accuracy limits).
- `engine.use_fp32_acc` (False | True; acts with `fp16`): inserts FP32 casts
  around every matmul, so every GEMM runs as `f16f32` with FP32 accumulation
  (FP32-accumulation GEMMs run at a lower rate on this GPU; see the cuBLAS
  rates). With `decompose_attention` the attention matmuls are cast too, and
  attention runs unfused in TF32 instead of in the fused MHA kernel.
- `engine.build_route`, TensorRT Global Performance Tuning routes (TensorRT
  11.1+, not TensorRT-RTX or Windows):
  - `-cask:fp16_acc=off` forbids FP16-accumulation GEMM kernels outside
    fusions: 44 of the 109 GEMM launches switch to FP32 accumulation, the
    fused ones stay FP16.
  - `-cask:fp16_acc=off -cask_fusion:fp16_acc=off`: all 109 GEMM launches
    accumulate in FP32.
  - Neither inserts casts, so attention stays fused (unlike `use_fp32_acc`).
  - `-tactic:splitk=off` drops split-K GEMM tactics. In the dense baseline
    engine it selects other GEMM tiles; engines built with
    `decompose_attention` come out bitwise unchanged (an exact no-op there).
- The FP32-accumulation options trade GEMM speed for bulk accuracy; the
  baseline already passes the limits with FP16 accumulation.

Runtime and builder:
- `runtime.cudagraphs` (replay the engine from a CUDA graph) and
  `runtime.pre_allocated_outputs` (reuse output buffers between calls): they
  target launch and host overhead. The baseline is one engine with 4% host
  overhead and no syncs.
- `engine.optimization_level` (3 | 5 | 4): higher levels search more tactics
  and take longer to build.

Not in the catalog:
- BF16: misses the limits.
- `engine.tiling_optimization_level`: `fast` crashes the TensorRT 11.4 builder
  (segmentation fault) on this model.
- Build route `-kgen:codegen:cuda_tile=2`: no engine (a Myelin check fails).
- Build routes `-cask_fusion:fp16_acc=off` (alone), `-late_h_fusion=off` and
  `-mha_fusion:v2:force_flash_tiled=on`: bitwise-equal engines (exact no-ops).
- No catalog tool targets the small kernels (~1.4 ms in the baseline: QKV
  split transposes/casts, LayerNorm + residual, RoPE).

## Reference points: PyTorch
PyTorch on the same input and precision (FP16), smoke numbers on a shared GPU.
`-fp16acc` means `bench.fp16_accumulation`, `-cdt` coordinate-descent tuning;
all of these compile and warm up inside build:

| config | ms | trimmed rel L2 | block median cos | tokens < 0.9 | cos min | rel L2 | build s |
|---|---|---|---|---|---|---|---|
| inductor-max-autotune-flex-cdt-fp16acc | 17.680 | 0.0043 | 0.99996 | 0.04% | 0.76 | 0.015 | 28 |
| inductor-max-autotune-flex-fp16acc | 17.704 | 0.0046 | 0.99997 | 0.07% | 0.63 | 0.020 | 29 |
| inductor-default-flex-fp16acc | 18.283 | 0.0046 | 0.99996 | 0.02% | 0.67 | 0.016 | 12 |
| inductor-max-autotune-banded128-fp16acc | 18.674 | 0.0045 | 0.99994 | 0.04% | 0.51 | 0.016 | 25 |
| inductor-max-autotune-flex | 19.533 | 0.0019 | 1.00000 | 0.02% | 0.81 | 0.010 | 29 |
| inductor-max-autotune-flex-cdt | 19.616 | 0.0020 | 0.99998 | 0.01% | 0.61 | 0.011 | 31 |
| inductor-max-autotune-no-cudagraphs-flex | 19.938 | 0.0021 | 0.99998 | 0.02% | 0.58 | 0.013 | 33 |
| inductor-max-autotune-banded128 | 20.290 | 0.0020 | 0.99999 | 0.01% | 0.61 | 0.011 | 22 |
| inductor-default-flex | 21.304 | 0.0024 | 0.99998 | 0.01% | 0.65 | 0.011 | 12 |
| inductor-max-autotune-fp16acc | 50.854 | 0.0046 | 0.99993 | 0.04% | 0.52 | 0.018 | 22 |
| inductor-default-fp16acc | 50.909 | 0.0044 | 0.99998 | 0.05% | 0.72 | 0.016 | 8 |
| inductor-max-autotune-cdt | 52.529 | 0.0020 | 0.99998 | 0.02% | 0.54 | 0.012 | 21 |
| inductor-max-autotune | 52.575 | 0.0020 | 0.99999 | 0.02% | 0.72 | 0.012 | 22 |
| inductor-max-autotune-no-cudagraphs | 52.593 | 0.0021 | 0.99999 | 0.01% | 0.60 | 0.012 | 20 |
| inductor-reduce-overhead | 53.433 | 0.0020 | 0.99999 | 0.01% | 0.86 | 0.008 | 8 |
| inductor-default | 53.933 | 0.0020 | 0.99999 | 0.01% | 0.86 | 0.008 | 8 |
| eager-fp16acc | 58.931 | 0.0047 | 0.99997 | 0.05% | 0.58 | 0.021 | 1 |
| eager | 61.576 | 0.0028 | 0.99998 | 0.02% | 0.77 | 0.012 | 1 |

## How to work
- One hypothesis per trial, motivated by the profile: name the kernel or
  signal it targets and what the profile predicts.
- In the baseline, local attention is 46% of the time; global attention and
  the GEMMs run at TensorRT's FP16 kernel rates; the ~1.4 ms of small kernels
  has no catalog tool.
- The precision tools trade speed for bulk accuracy.
- Differences below the session's noise threshold are not real. Within-run
  IQR is ~1% for the dense engines and ~2.5% for engines with decomposed or
  banded attention on the shared GPU, so re-measure before believing a small
  gain.
- Revisit a rejected option only when the incumbent changed in a way that
  could change its effect.
- Stop when no remaining option can plausibly clear the noise threshold.
