# Research brief: Qwen3.5-0.8B batch-1 prefill (2k tokens) with Torch-TensorRT

## Objective
Minimize milliseconds per prefill of the text model of `Qwen/Qwen3.5-0.8B` on
one RTX 5090: one forward pass at batch 1 over a fixed 2,048-token prompt (the
start of CPython's language reference, `pydoc_data.topics`, never repeating),
from an empty state, returning the logits at 64 evenly spaced positions (31,
63, ..., 2047; the LM head runs on those rows only).
Workload: `tools/llm/qwen35_prefill.py:create_case` with
`{"model": "Qwen/Qwen3.5-0.8B", "length": 2048, "positions": 64}`.

## Accuracy
Candidates are checked against the FP32 eager Hugging Face model on the same
tokens (SDPA, Hugging Face's PyTorch Gated DeltaNet fallback, TF32 off). The
limits are the workload defaults: top-1 agreement >= 0.95, mean KL <= 0.002,
and largest KL at any one position <= 0.01.

Measured values (top-1 / mean KL / largest KL):
- BF16 eager Hugging Face (the noise floor): 1.0 / 4.2e-4 / 1.2e-3. Correct
  BF16 TensorRT builds: 1.0 / 4.2e-4 to 4.9e-4 / up to 2.1e-3. FP16 with FP32
  accumulation: 1.0 / 1e-5; FP16 with FP16 accumulation: 1.0 / 1.6e-4 / 6e-4.
  FP32 TensorRT with the functional Gated DeltaNet: KL 3e-11.
- Legitimate approximations (eager BF16 variants of the functional rule): BF16
  operands in all chunk products, as fla does: 1.0 / 4.6e-4 / 1.7e-3; TF32 in
  all Gated DeltaNet products: 0.984 / 4.1e-4 / 1.5e-3.
- Errors the limits catch: one Gated DeltaNet layer without inter-chunk state
  (1.0-0.95 / 1.6e-3-1.3e-2 / 0.012-0.13); the last chunk's state correction
  skipped in every layer (0.98 / 9.8e-3 / 0.58); the last attention layer
  seeing future tokens (0.95 / 9.1e-3 / 0.084); the state one chunk late
  (0.91 / 0.071 / 0.62); a UT inverse truncated to 32-wide blocks
  (0.94 / 0.10 / 0.68). Gross errors (no gating, no inverse, reversed conv
  taps, no query scaling) give KL about 5-11 or infinite.
- Errors confined to the last Gated DeltaNet layer can hide in BF16 noise
  (its last chunk's state correction skipped: 1.0 / 5.3e-4 / 7.6e-3). A patch
  that changes the numerics should be an exact rewrite or a stated
  approximation; the unit tests (`tools/llm/tests/test_qwen35_gdn.py`) check
  the formulation exactly.
- Do not tighten top-1: six of the 64 reference positions have a top-2 logit
  gap of 0.08-0.32, within BF16's 0.2-0.3 logit error.

## The model
24 layers, hidden 1024, MLP 3584 (SiLU-gated), vocabulary 248,320 with tied
embeddings, 752M parameters (254M of them the embedding), BF16 native.
- 18 Gated DeltaNet (linear attention) layers: in_proj_qkv (1024 -> 6144),
  in_proj_z (-> 2048), in_proj_b/a (-> 16 each), a causal depthwise conv
  (width 4) + SiLU on q/k/v, 16 heads with key and value head dim 128, the
  gated delta rule (FP32), a gated RMSNorm and out_proj (2048 -> 1024).
- 6 gated full-attention layers (every 4th): 8 query and 2 KV heads of
  dimension 256, q/k RMSNorm, partial RoPE (64 of 256 dims), a sigmoid output
  gate.
- Without fla, Hugging Face runs Gated DeltaNet through its PyTorch fallback
  (`torch_chunk_gated_delta_rule`: a loop over 64-token chunks with in-place
  writes; `solve_triangular` in eager mode and a 63-step forward substitution
  under export). Production stacks (vLLM, SGLang, Hugging Face with fla) run
  fla's fused Triton chunk kernels instead; fla cannot be installed here.
- Under `torch.export` (and `torch.compile`) Hugging Face cannot run its
  data-dependent packed-sequence check, so it always materializes a dense
  2048 x 2048 attention mask; eager mode skips the mask and calls SDPA with
  `is_causal`.

## What bounds the answer
- One prefill needs 2.33 TFLOP of matrix products: 2.04 TFLOP of projections,
  0.10 TFLOP of causal attention (6 layers, masked half skipped), 0.15 TFLOP
  of Gated DeltaNet chunk math in the functional formulation (FP32; 0.09 at
  chunk 32, 0.39 at chunk 128) and 0.03 TFLOP of LM-head rows.
- Measured dense GEMM rates (cuBLAS 8192^3) and the floors they give:
  - BF16 with FP32 accumulation: 234 TFLOP/s, floor 9.9 ms
    (`compute_floor_ms`);
  - FP16 with FP16 accumulation: 357 TFLOP/s (1.5x), floor 6.5 ms
    (`compute_floor_ms_fp16_acc`).

  Every trial's `structure` reports `gflop_per_unit` and both floors. The
  FP32 Gated DeltaNet math runs on CUDA cores, or on TF32 tensor cores.
- Memory does not bound it: about 1.5 GB of BF16 weights per prefill, 0.93 ms
  at the measured 1.61 TB/s (no `bytes_per_unit` is declared, so trials have
  no `roofline` field).
- Host overhead is negligible: one engine, no host synchronizations, the GPU
  never waits (`host_overhead_fraction` below 1%).

## What profiling showed (smoke numbers: RTX 5090 shared with other jobs)
- Baseline (recipe `{}`: Hugging Face's fallback exported as is), 50.1 ms,
  6,724 kernels, about 5 minutes to build: 2,286 tiny FP32 GEMMs (32x32
  tiles, FFMA) of the forward substitution and the chunk loop take 8.1 ms
  (16%); BF16 projection GEMMs about 10.4 ms; the attention kernel 2.4 ms
  (6 calls, dense over the materialized mask); one fused elementwise kernel
  per Gated DeltaNet layer 1.6 ms; the rest are thousands of small
  elementwise kernels and copies of the in-place chunk loop.
- Functional formulation (`workload.gdn=functional`, everything else at its
  default), 43.4 ms, 4,258 kernels, 73 s to build: one fused Myelin kernel
  per Gated DeltaNet layer takes 16.5 ms (38%, 0.92 ms per layer). It fuses
  the q/k/v split and L2 norms with the transposed, channel-major output of
  Hugging Face's conv1d and reads across a 2048-element stride. Then BF16
  projections about 10.4 ms; 1,134 small FP32 GEMMs of the unrolled
  inter-chunk state recurrence (63 per layer, about 3 us each) 3.5 ms; 288
  batched FP32 chunk GEMMs (UT system, inverse, u/w, scores, outputs) 3.5 ms;
  attention 2.4 ms.
- Plugin-region builds (`kernels.gdn_scan=triton*`) fuse the elementwise work
  around each Gated DeltaNet layer differently. Loop builds run the conv,
  q/k/v split and SiLU as one kernel per layer (`__myl_SigmTranReshCastConc...
  Silu...`, 0.88 ms per prefill) and the L2 norms and decays as another
  (`__myl_AddSqrtDivMul...`, 1.18 ms). Scan builds fuse both into one kernel
  per layer that takes 2.82 ms, about 0.8 ms more (3% of such a build). A loop
  build through the same strict export path (`torch_tensorrt.compile(...,
  strict=True)`) keeps the two kernels at 0.87 and 1.19 ms, so the region
  boundary causes it, not the export path. In a build with every other tool
  this kernel is the largest single kernel (16%). This is an open lever: what
  enters the region and in which layout is set by `chunk_gated_delta_rule`
  and the layer forward, both in the patch slot.

## Tools and their tradeoffs
- `workload.gdn` (`hf` baseline, `functional`): the functional formulation
  (`qwen35_gdn.py`) does the same math as Hugging Face's fallback with batched
  matrix products (UT inverse by block-doubling forward substitution); in FP32
  it matches Hugging Face's eager logits to 3e-5 at chunk 64 (9e-5 at chunk
  128), and an FP32 engine matches them to KL 3e-11.
  Builds take about 1 minute instead of 5. Every other Gated DeltaNet tool
  requires it.
- `workload.conv` (`conv1d`, `shifts`): Hugging Face's channel-major conv1d,
  or four token-major shifted multiply-adds in FP32 fused with the SiLU.
- `workload.chunk_size` (64, 32, 128): shorter chunks make the per-chunk
  systems cheaper but the sequential recurrence longer (32 steps per layer at
  64), longer chunks the reverse; chunk 32 doubles the graph (build about
  160 s).
- `kernels.gdn_scan` (`loop`, `triton`, `triton-tf32`; requires
  `runtime.cudagraphs=True`): the recurrence as one Triton kernel per layer
  inside the engine (`torch_tensorrt.region.fused_kernel`), with IEEE-FP32 or
  TF32 dots. In isolation it takes 200 us (IEEE) or 137 us (TF32) per layer,
  against about 300 us for the unrolled loop. Works with every chunk size:
  the kernel walks a chunk in blocks of 64 rows and needs 32-36 KB of shared
  memory. The build compiles the kernel first and refuses one that the
  TensorRT 11.4 plugin launch cannot run (over 48 KB of shared memory, or
  Triton scratch buffers). Plugin builds take about 1 minute; see the
  profile note above about the fusion around the region.
- `workload.attention` (`mask`, `causal`) and `engine.decompose_attention`:
  the dense-mask attention kernel, or attention that skips the masked tiles.
- `engine.precision` (`bf16`, `fp16`, `fp32`) and `engine.use_fp32_acc`
  (FP16 only; True is the documented `tools/llm` setting): FP16 does not
  overflow for this model on this prompt and is more accurate than BF16.
  With `use_fp32_acc=False` TensorRT picks FP16-accumulation GEMMs
  (`f16f16_f16f16_f16`), which run faster on GeForce GPUs with less accurate
  sums. PyTorch accumulates FP16 GEMMs in FP32 unless
  `allow_fp16_accumulation` is set, so state the policy in any comparison.
- `engine.disable_tf32` (True baseline): False lets TensorRT run the FP32
  Gated DeltaNet math on TF32 tensor cores (10-bit-mantissa products).
- Generic: `runtime.cudagraphs` (required by the scan kernel; little host
  overhead to remove otherwise), `runtime.pre_allocated_outputs`,
  `engine.optimization_level` and `engine.tiling_optimization_level` (level 5
  and `fast` were checked on a functional recipe: they build in about the
  same time as the defaults and pass accuracy).
- Patch slot `gdn` (`qwen35_gdn.py` inside its EVOLVE-BLOCK fences): the
  functional Gated DeltaNet numerics, the Triton scan kernel and its launch
  table SCAN_LAUNCH, and the layer forward (conv, q/k/v split, gates). Ideas:
  BF16/FP16 operands with FP32 accumulation for the chunk GEMMs (what fla
  does), fusing the recurrence with the output products (`q S` and
  `scores v_new`), a cheaper inverse, a scan kernel that splits the 128-row
  state across warps so that pipelining fits in 48 KB, and a layout of the
  conv/split/norm work that fuses better around the scan region.

Build times: the `hf` baseline takes about 5 minutes (28k exported nodes,
6.7k engine layers), functional recipes 50-95 s (about 160 s at chunk 32).
Every race or confirmation against the baseline rebuilds it. A heuristic
smoke session (2 trials, 1 calibration run) took about 25 minutes, mostly
baseline builds. The exported artifact of a build is about 2 GB (the 1.5 GB
engine, base64-encoded by `torch_tensorrt.save`).

## How to work
- One hypothesis per trial; name the kernel group in the profile (and its
  launch count) that motivates it. The `hf` baseline costs 5 minutes per
  build; functional recipes about 1 minute.
- Differences below the session's calibrated noise threshold are not real.
- Lower-precision choices (FP16 accumulation, TF32, BF16 chunk operands)
  change rounding; report the mean and largest KL with every promotion.
- Revisit a rejected option only if the incumbent changed in a way that could
  change its effect.
- Read the whole kernel table, not just the GEMMs: the fused elementwise
  kernels around the Gated DeltaNet layers can be as large as the GEMMs, and
  their fusion changes around plugin regions. Stop when no remaining option
  or patch can plausibly beat the noise threshold.

## Hybrid tool: `kernels.inductor` (Inductor regions inside the TensorRT model)
`kernels.inductor` (default `none`) hands chosen operations to PyTorch Inductor
while the rest of the model stays in TensorRT engines. Each chosen scope runs
under `torch_tensorrt.region.execute_in_torch(backend="inductor")`, which the
strict `torch_tensorrt.compile` path (the one `kernels.gdn_scan` already uses)
keeps out of TensorRT; Inductor compiles it at the first call (default mode,
without its own CUDA graphs).
- `attention`: the attention core of the 6 full-attention layers (SDPA
  `is_causal` with native GQA, and the transpose of its output). Inductor calls
  PyTorch's fused SDPA kernel for it (FlashAttention-2 at head_dim 256, a
  split-KV kernel plus a combine kernel) and a generated Triton kernel for the
  transpose, instead of TensorRT's attention kernel. 6 regions, 7 engines, 12
  engine boundaries.
- `prologue`: each Gated DeltaNet layer's `in_proj_qkv` GEMM (cuBLAS), causal
  convolution + SiLU, q/k/v split and FP32 L2 norms (generated Triton kernels).
  18 regions, 19 engines, 36 boundaries. The gates and the chunk rule's
  elementwise work on q, k and v (scale, cumulative decays, beta and decay
  products) stay in TensorRT, which reads the region's FP32 q and k and its v
  back.
- `prologue-decay`: the same regions extended through the gates and the chunk
  rule's elementwise head, so TensorRT keeps the chunk matrix products, the UT
  inverse, the scan plugin and the outputs; the region writes all of their
  inputs (about 120 MB per layer at 2k tokens). It runs the workload's own copy
  of the chunk rule (`chunk_rule_head`/`chunk_rule_body` in `qwen35_gdn.py`,
  outside the fences), so a `gdn` patch of `chunk_gated_delta_rule` or
  `functional_forward` would not reach it; patches of the convolution, the
  inverse, the scan and its kernel do.
- `attention+prologue`, `attention+prologue-decay`: both kinds of region (24
  regions, 25 engines, 48 boundaries).

Requirements: `runtime.cudagraphs=True` (one CUDA graph then replays the
engines and the regions together, so a boundary adds no host launch),
`workload.gdn=functional` and `workload.attention=causal` (the values share one
set of requirements). The regions follow the recipe's matrix precision policy
(FP16 accumulation and TF32 when the recipe allows them). With a `gdn` patch,
the build refuses a prologue value when the patch edits `functional_forward` or
`chunk_gated_delta_rule`: those builds do not run these functions as written
(`prologue` replaces the layer forward and applies the q/k L2 norms itself;
`prologue-decay` also runs its own copy of the rule), so the trial would
otherwise measure a build without the patch. Hybrid builds are not exported:
`torch_tensorrt.save` inlines the regions and keeps no backend, so a reloaded
program would run them without Inductor; the candidate refuses the export, and
the recipe rebuilds it.

Costs and risks:
- Every boundary is an engine output plus an engine input: the tensors that
  cross it are written and read once more, and TensorRT's fusion and tactic
  choices stop at the cut, so kernels next to a cut can change in either
  direction. With `prologue`, the elementwise work left between the region and
  the chunk products is such a case; `prologue-decay` moves it into the region
  at the price of larger region outputs.
- With either prologue, `in_proj_qkv` leaves TensorRT's merged input
  projection (TensorRT otherwise merges the q/k/v, z, b and a projections into
  one GEMM); `in_proj_z/b/a` stay merged in TensorRT.
- The first call compiles the regions; the build keeps the regions' weights
  on the GPU next to the engines.
- GPU memory grows with the number of engines and regions, far beyond the
  single engine's: every engine reserves its own TensorRT activation memory for
  its largest layer, and every engine with a Gated DeltaNet layer needs about
  the reservation of the whole single engine; the first and the last engine
  both store the tied embedding and LM-head weight; and the tensors that cross
  the boundaries stay allocated between calls, because each engine keeps its
  bound inputs until its next call. The whole-model CUDA graph's memory pool
  therefore holds all boundary tensors at once instead of reusing memory, and
  the regions' eager warm-up before the capture leaves about as much again
  cached in PyTorch's allocator. The prologue values, whose regions write FP32
  tensors, cost the most, `prologue-decay` above all. Check the trial's
  `memory` field when GPU memory matters.
- Numerics: the regions compute the same operations; Inductor's fused kernels
  may skip intermediate roundings to FP16 (Inductor's `emulate_precision_casts`
  is off by default: the SiLU output that feeds the q/k L2 norms, for example,
  stays FP32 where the eager code rounds it), and FlashAttention-2 replaces
  TensorRT's attention kernel (both take FP16 inputs and accumulate the softmax
  in FP32).
- Splitting the model into several engines also changes how TensorRT
  schedules each engine (for example whether it runs parallel branches on
  auxiliary streams). Profiles of these builds trace one whole-model CUDA
  graph replay instead of one graph per engine; when kernels run on auxiliary
  streams, summed kernel time can exceed the latency.
