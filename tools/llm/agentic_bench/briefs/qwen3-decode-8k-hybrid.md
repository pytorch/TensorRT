# Research brief: Qwen3-0.6B batch-1 decode at 8k context with Torch-TensorRT

## Objective
Minimize milliseconds per generated token for batch-1 greedy decode of
Qwen/Qwen3-0.6B on one RTX 5090 with an 8192-token prompt. The prompt is
prefilled untimed; then 64 tokens are decoded, teacher-forced, at positions
8192..8255 (an 8256-slot KV cache). Accuracy limits, as at 128 tokens: top-1
agreement >= 0.97 and mean KL <= 0.005 against FP32 eager logits at every
decode position, checked on the first call and again after timing.

Workload: `tools/llm/agentic_decode.py:create_case` with
`{"model": "Qwen/Qwen3-0.6B", "isl": 8192, "osl": 64, "prompt": "recall", "prefill": "eager"}`.

- **Tokens (`prompt="recall"`).** The prompt opens with a paragraph written
  for this workload, continues with the opening of "A Tale of Two Cities"
  repeated as filler, and ends with the paragraph's first 16 tokens. The first
  28 decode tokens continue that paragraph, which the model can only predict
  by attending about 8k tokens back. The other 36 continue with text that
  appears nowhere in the prompt, so their predictions keep a natural
  uncertainty: mean entropy 1.3 nats over the 64 positions (2.3 over the 36),
  against 1.8 at 128 tokens. That text was chosen so that no position has two
  nearly tied best tokens (FP32 top-2 logit gaps of at least 0.15), which
  correct FP16 or BF16 rounding would flip.
- **What the check catches.** Simulated in FP32 against this reference, at
  the decode positions only: a 4096-token attention window, a dropped prefix
  (128 or 2048 tokens), lost or misplaced KV writes, and attention past the
  written slots all fail (KL 0.48-9.6). So do one skipped MLP or o_proj, the
  last 1/64 of the MLP rows missing in every layer, or a skipped final norm
  (KL 0.021-27; the limit is 0.005). Two bug classes pass: rotary positions
  off by one at the decode positions only (KL 6e-4 and 1.1e-3) and zeroed
  logits for the last 1/16 of the vocabulary (rare tokens, KL 4e-5). Correct
  builds measure top-1 1.0 with KL 1.2-1.7e-4 in FP16 (TensorRT or Inductor)
  and 1.8e-3 in BF16.
- **Unwritten cache slots.** Before the first call, a static_v3 engine's 64
  decode slots hold noise at the scale of the written ones. A correct engine
  never reads a slot before writing it. An engine that attends past position
  + 1 reads the noise and fails the first check (simulated: KL 9).
- **Prefill (`prefill="eager"`).** static_v3 candidates take the prompt's
  keys and values from one Hugging Face FP16 prefill of the same model,
  copied into the engine's cache in under 1 s. The alternative, 8192
  one-token engine calls, takes 13-16 s. Decode after either prefill agrees
  to top-1 1.0 and KL 3e-7. Prefill is never timed.

## What bounds the answer
- **Bytes per token.** All FP16 weights are read once: 1.192 GB, with the
  tied LM head counted once. Each cache slot holds 28 layers x 8 KV heads x
  128 x 2 (K and V) x 2 B = 115 KB, and a step reads the written slots: 8224
  on average, 0.943 GB. That is 2.135 GB per token.
- **The floor.** At the measured 1.61 TB/s read bandwidth, the floor is
  1.326 ms/token: 0.74 ms for weights and 0.59 ms for the KV cache. KV is 44%
  of the bytes; at 128 tokens it was 2%.
- **FLOPs per token.** About 1.2 GFLOP of GEMV plus 1.9 GFLOP of attention
  (QK and PV over 8224 keys, 16 heads, 28 layers): 3.1 GFLOP, or 1.5 FLOP per
  byte. Even on CUDA cores that is about 0.03 ms. Decode is memory-bound;
  tensor-core throughput does not matter.
- **The `roofline` field.** Each trial reports the fraction of the floor it
  reaches. The `roofline_gap` signal fires below 80%.

## A correctness finding (TensorRT 11.4, already handled)
For KV caches of about 1k slots and more (1088 measured), TensorRT 11.4 runs
decode attention with its `_gemv_mha_v2` kernel, which ignores the attention
mask, Boolean or additive, of any shape.

- **Effect on a masked lowering.** The original static_v3 lowering is a
  masked full-length SDPA. At 8256 slots, unwritten cache slots took part in
  the softmax. One-token engine prefill then corrupted the cache (decode
  top-1 0.58, KL 1.6), and garbage in masked slots changed logits at every
  position.
- **The older kernel.** `_gemv_mha_v1` honors the mask. TensorRT picks it for
  the 192-slot cache of the 128-token workload, for BF16 engines, and when the
  build route `-mha_fusion:gemv_mha_v2=off` forbids v2.
- **What the workload does.** Above 192 slots, static_v3 builds TensorRT's
  attention layer with `key_value_lengths = position + 1`
  (`static_cache_v3.configure(attention="auto")`, the default). It is
  correct at every position: KL against FP16 eager is at most 5e-5, and
  garbage in unwritten slots has no effect. It still runs `_gemv_mha_v2` and
  reads only the written slots. There is nothing to choose.

## What the profiles showed
Smoke numbers: single runs on a shared GPU, taken under an exclusive lock.

- **Baseline (static_v2, the documented usage).** 14.3 ms/token of GPU time,
  58 device-to-host copies per token, 10.3 GB of process memory:
  - 56%: the dynamic-shape engine runs the one-token attention with a
    GEMM-shaped kernel (`_gemm_mha_v2`, about 8 ms);
  - 21%: Move/Slice/Concat kernels copy the whole cache every layer and
    step (about 3 ms);
  - 19%: the projections as tensor-core GEMMs (about 2.7 ms).
- **A plain static_v3 engine (FP32 accumulation on, no CUDA graphs).** 1.93
  ms/token of GPU time in 339 kernels, host overhead about 2%, no host syncs:
  - GEMV: 0.92 ms (48%): cuBLAS `gemvx` with FP32 accumulation for the
    attention projections, and tensor-core `xmma` GEMMs for the MLP;
  - decode attention: 0.75 ms (39%): `_gemv_mha_v2`, 26.8 us per layer for
    the 33.7 MB of K/V it reads, about 1.26 TB/s (78% of the measured peak);
  - RMSNorm, RoPE, SiLU and the KV scatter: about 0.2 ms in many small
    Myelin kernels.
- **Inductor's flex-decoding attention is no faster.** Measured alone on 28
  distinct 8256-slot caches, FlexAttention compiled by Inductor (one split-KV
  kernel plus two small reduction kernels) takes 26.0 us per layer, about
  1.30 TB/s, against 26.8 us for `_gemv_mha_v2` in the engine. Inside a whole
  Inductor decode step its kernels also carry fused norm and RoPE work, so
  kernel names there overstate the attention share.
- **SDPA-based PyTorch paths.** One-query decode with a mask goes to the
  memory-efficient CUTLASS kernel: 575 us per layer, about 16 ms/token.

## Tools and tradeoffs
- `workload.cache`
  - `static_v3` removes the per-layer host syncs and the whole-cache copies,
    and makes CUDA graphs possible. `kernels.decode_blocks` needs it.
  - `static_v2` (the baseline) and `static_v1` slice and copy the cache every
    step and run the GEMM-shaped attention.
- `engine.use_fp32_acc`: on, FP16 GEMVs accumulate in FP32 through inserted
  casts, which selects `gemvx f16f16_f32_f32` and tensor-core GEMMs. Off,
  TensorRT uses `gemvx f16f16_f16_f16`.
- `runtime.cudagraphs`: removes launch overhead. Host overhead of a static_v3
  engine is about 2% at this length. `kernels.decode_blocks` requires it.
- `runtime.pre_allocated_outputs`: does nothing for static_v3 engines, whose
  caches are aliased inputs and outputs (the runtime disables pre-allocation
  for them); only static_v1/v2 engines can use it.
- `kernels.decode_blocks`: fused Triton RMSNorm+GEMV blocks (+SiLU or
  +residual) inside the engine: `mlp`, `mlp+attn`, `layers`, `all`. The KV
  write and attention stay in TensorRT. Needs static_v3, fp16 and CUDA
  graphs.
  - Patch slots `decode_launch` (BLOCK_N, BLOCK_K, num_warps and EVICT per
    block) and `decode_kernels` (the Triton code) are open. The accuracy check
    above catches dropped rows and skipped blocks.
- `engine.build_route`: Myelin knobs.
  - `-kgen:lower_fc:all=on` generates the GEMVs with kgen instead of cuBLAS
    `gemvx`.
  - `-mha_fusion:gemv_mha_v2=off` switches attention to `_gemv_mha_v1`.
  - On a static_v3 FP16 engine (use_fp32_acc off, CUDA graphs) at this
    length, `-peep:match_gemv_mha=on`, `-cask_fusion:fp16_acc=off`,
    `-tactic:splitk=off` and `-late_h_fusion=off` each build the same engine
    as no route: identical kernels and bit-identical logits.
    `-peep:match_gemv_mha=on` does the same on the static_v2 baseline.
  - `-kgen:codegen:cuda_tile=2` fails to build that engine (TensorRT 11.4
    Myelin internal check, `copy_vtx_analyzer.cpp:112`).
- `engine.precision`: in BF16, TensorRT 11.4 decodes this cache with
  `_gemv_mha_v1` instead of `_gemv_mha_v2`. FP32 doubles every byte.
- `engine.optimization_level`: levels 4 and 5 build the same engine as the
  default 3 for that static_v3 FP16 engine (identical kernels and logits).
- `engine.tiling_optimization_level`: every level other than `none` fails to
  build static_v3 engines. TensorRT 11.4's tiling tactics reject the aliased
  cache inputs and outputs ("should alias but memory addresses differ").

Build times at 8k, from model load to engine (eager prefill included):
static_v1/static_v2 22-32 s, static_v3 16-19 s, fused blocks 41 s (`mlp`),
62 s (`mlp+attn`) and 80 s (`layers`, `all`). A whole worker run, with the
accuracy checks and timing, takes 25-90 s.

## The open gap
Attention over the 8k cache is the largest single kernel and reads K/V at
about 78% of the bandwidth, as Inductor's flex-decoding does (81%). At the
full 1.61 TB/s a layer's 33.7 MB would take 20.9 us instead of 26.8 us. The
catalog has nothing that changes the attention kernel itself beyond the two
build routes.

- A split-K (flash-decoding) kernel at 90-95% of bandwidth would save about
  0.10-0.13 ms/token.
- It cannot be a `fused_kernel` region today: the cache only exists after
  export (the static_v3 pass adds it), and SDPA is not allowed inside
  regions.
- A feasible route is the one static_v3 already uses for `key_value_lengths`:
  a custom op emitted by the lowering pass, implemented by a Triton kernel
  plugin (partial plus combine, or one kernel with an in-kernel combine).
- The other big lever is fewer KV bytes (an FP8 KV cache), which is not in
  the catalog.

## How to work
- One hypothesis per trial; say which signal or measurement motivates it.
- Start from the structure (cache lowering, CUDA graphs), then kernels
  (`kernels.decode_blocks`), then knobs and patch slots. Prefer changes that
  remove whole kernels, copies or host syncs over sweeps.
- Differences below the session's calibrated noise threshold are noise.
- Stop when no remaining option can plausibly beat the noise threshold.

## Hybrid variants: Inductor's kernels in the decode step
Two tools, both off by default, let a recipe hand operations to Inductor's
kernels while the rest of the step stays the recipe's TensorRT engine (fused
blocks or plain static_v3, strictly exported). Existing recipes build as
before.

- `runtime.step_graph` (requires static_v3 and `runtime.cudagraphs`): one CUDA
  graph for the whole decode step, covering every engine, plugin and PyTorch
  region. The caches are bound in place, and the token and position are
  copied into static buffers before each replay; an engine prefill runs
  through replays of the same graph. `runtime.cudagraphs` alone captures each
  engine on its own and runs PyTorch regions eagerly between engines. With a
  single engine, the step graph also skips the Python runtime's per-call host
  work. The `kernels.inductor` values with regions (`lm-head`, `attention`,
  `attention-math`) require it; `attention-flex` does not. Measuring a recipe
  with and without it separates the capture change from the hybrid change.
- `kernels.inductor` (requires static_v3, fp16 and `runtime.cudagraphs`):
  - `attention-flex`: the kernels Inductor generates for FlexAttention decode,
    hosted unchanged inside the engine as three `fused_kernel` plugins per
    layer. No engine boundary, so it runs under the session's per-engine CUDA
    graphs or under the step graph.
    - What runs: the split-KV template kernel and its two combine reductions,
      generated at build time for this model's shapes with
      `mode="max-autotune-no-cudagraphs"` and the mask kv_idx <= position. They
      replace `_gemv_mha_v2` and its memset: 84 plugin launches per token
      instead of 56 kernels. The engine computes the block mask once per
      step, with a few integer ops.
    - Explicit caches: the step takes its 56 caches as inputs
      (`static_cache_v3.cache_decode_step`) and writes them with a functional
      `index_copy`, which TensorRT still lowers to its in-place cache update;
      attention then reads the written caches and masks slots past the
      position.
    - Alignment wrapper: the plugin compile drops the 16-byte
      pointer-alignment specialization that Inductor's own launch applies.
      Without it, the template compiles to unpipelined narrow loads. A wrapper
      per kernel restores the alignment and produces the same machine code as
      Inductor's launch (37 KB of shared memory, under the 48 KB QDP limit).
      The wrapper rounds pointers down, so every binding must be 16-byte
      aligned: TensorRT's own buffers are, and setup checks the cache inputs.
    - Build check: the hosted kernels must reproduce Inductor's own output on
      random inputs before the engine is built.
    - Risks: the hosting expects Inductor's code layout (one template and two
      reductions); shapes that Inductor compiles differently, such as fewer
      KV heads, are refused at build. It reads Inductor's chosen launch
      settings through private Inductor internals.
  - `lm-head`: the [residual add +] final RMSNorm + lm_head as one
    `execute_in_torch(backend="inductor")` region after the engine. It
    replaces the `final` fused block and adds 1 engine-to-region boundary per
    token. The region reads the final norm and the tied LM-head weight, so
    the build keeps those weights (and the embedding they share) on the GPU
    besides the engine; the decoder layers' weights are released.
  - `attention`: each layer's decode attention in an
    `execute_in_torch(backend="inductor")` region running Inductor's
    flex-decoding: 28 regions, 29 engines and 56 engine-region boundaries per
    token, same explicit caches and per-step block mask as `attention-flex`.
    Export captures `flex_attention` as a higher-order operator, which region
    capture rejects inside a scope, so the region calls the opaque, functional
    custom operator `qwen3_regions::flex_decode`; its CUDA implementation runs
    `flex_attention` compiled with `mode="max-autotune-no-cudagraphs"`, the
    same kernels `attention-flex` hosts, launched by Inductor's runtime
    instead of TensorRT.
  - `attention-math`: the same 28 regions with plain attention math as the
    body (a grouped QK^T matmul, a masked FP32 softmax and a PV matmul), which
    Inductor compiles with its max-autotune options (max_autotune and
    coordinate_descent_tuning). These are not flex-decoding kernels: Inductor
    chooses matmul templates or cuBLAS/CUTLASS for the matmuls.
  - Region costs: each region splits the engine, the step makes two
    transitions per region inside the graph, and TensorRT can no longer fuse
    across a region. Regions compile in the build, on one step over scratch
    caches, without Inductor's own CUDA graphs.
  - Variance risk: builds of the two attention region variants (29 engines
    in one graph) have run at two distinct inter-kernel gap levels from one
    process to the next, with the same kernels, kernel times and graph; a
    single build can mislead, so compare more than one.
- Accuracy: all variants keep FP16 inputs with FP32 accumulation. The flex
  variants keep the scores in FP32 through the softmax. `attention-math` rounds
  the QK^T scores to FP16 (the matmul's output type) before its FP32 softmax.
  `attention-math` and flex-decoding round the softmax probabilities to FP16
  before the PV product.
