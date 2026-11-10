# Research brief: Nemotron-3-Nano-4B batch-1 decode latency with Torch-TensorRT

## Objective
Minimize milliseconds per generated token for batch-1 greedy decode of
nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (128-token prompt prefilled untimed,
then 64 teacher-forced decode tokens) on one RTX 5090, without breaking
accuracy.
Workload: `tools/llm/nemotron_decode.py:create_case` (`NemotronDecodeCase`).

## Accuracy
- Limits: top-1 agreement >= 0.97 and mean KL <= 0.002 against an FP32 eager
  reference at every decode position (the Hugging Face model in FP32,
  prefilling with its chunked scan and decoding token by token with its own
  cache). Over the 64 positions one flipped top-1 token still passes (0.984);
  two fail (0.969).
- Correct BF16 engines measure a mean KL of 1e-4 to 7.4e-4 and FP16 ones
  2e-6 to 2e-5, so 0.002 still leaves a 2.7x margin.
- Broken engines fail it: a query that cannot attend to itself (mean KL
  3.9e-3), frozen conv or SSM states (1.8 to 4.5) or lost KV writes (0.7).
- BF16 noise can hide a small formulation error: a last Mamba layer whose SSM
  state never updates measures a mean KL of 8.6e-4, close to correct BF16
  engines and inside the limit. Patched builds are therefore checked in FP32
  first (see the patch slot below).
- The reference records its 192 token ids and `compare` refuses a reference
  computed for other tokens.

## The model and the decode step
- 42 layers: 21 Mamba-2 mixers, 17 squared-ReLU MLPs, 4 GQA attention layers
  (40 query / 8 KV heads, head_dim 128, no positional encoding); hidden 3136;
  Mamba: 96 heads x head_dim 80, 8 groups, state 128, conv kernel 4;
  vocab 131072 with an untied LM head. 3.97 B parameters, 7.95 GB in BF16.
- Every TensorRT candidate compiles `NemotronDecodeStep`: one shape-static
  decode step in plain PyTorch ops (exported with torch.export, compiled by
  `torch_tensorrt.dynamo.compile` into ONE engine, 0 PyTorch partitions).
  Inputs: token, position, 4x2 KV caches [1, 8, 192, 128] (written in place at
  the position by IKVCacheUpdateLayer, masked full-length attention), 21 conv
  states [1, 9728, 4] and 21 SSM states [1, 96, 80, 128]. The SSM update
  (softplus, exp(dt*A) decay, dt*B*x outer product, C.h + D*x) and the
  depthwise convolution run in FP32; gated RMSNorm per group of 960.
- The engine has no host synchronization and can be captured in a CUDA graph.
- The prompt is prefilled by Hugging Face eagerly (untimed) in the candidate's
  precision and its caches are copied into the static buffers.

## What bounds the answer
- Each token must move about 7.30 GB: every weight once except the untied
  input embedding (7.12 GB: Mamba in/out projections 3.32 GB, of which
  in_proj 2.31 GB; MLPs 2.67 GB; attention 0.31 GB; LM head 0.82 GB), the 21
  SSM states read and written (165 MB in FP32, 83 MB in BF16), the conv states
  (1.6 MB) and the whole static KV cache (3.1 MB). At this GPU's measured read
  bandwidth (~1.607 TB/s) the floor is 4.54 ms/token (4.49 ms with a BF16 SSM
  state).
- Each trial's `roofline` field reports the fraction reached (the bandwidth
  probe varies by about 1% between runs); `roofline_gap` fires below 80%.
- Compute is irrelevant at batch 1: ~7.1 GFLOP/token, about 35 us of
  tensor-core time.
- The baseline reaches 83.5% of the floor, about 0.9 ms/token above it.

## What profiling showed (baseline)
Smoke numbers, measured while other jobs shared the GPU.

Baseline (BF16, functional states, FP32 SSM state, no CUDA graphs, no build
route, optimization level 3): 5.425 ms/token, top-1 agreement 1.0 and mean KL
5.3e-4, 255 kernels/token, host overhead 0 (the GPU is never idle), no host
syncs, build 28 s.

| Kernel (per token) | ms | share | launches |
|---|---|---|---|
| xmma BF16 GEMM 32x32x64 tn (projections, MLPs) | 4.00 | 73% | 80 |
| xmma BF16 GEMM nn (LM head + 4) | 0.58 | 10.5% | 5 |
| Myelin Mamba step (conv + softplus + state update + C.h) | 0.46 | 8.5% | 21 (22 us each) |
| Myelin RMSNorm | 0.19 | 3.4% | 41 (4.5 us each, latency-bound) |
| split-K reduction | 0.12 | 2.3% | 55 |
| Myelin gated RMSNorm | 0.06 | 1.1% | 21 |
| attention (_gemv_mha_v1) | 0.02 | 0.4% | 4 |

- The two GEMM rows read the 7.12 GB of weights in 4.58 ms, about 1.55 TB/s:
  the baseline's GEMVs already stream close to the measured read bandwidth.
- The remaining ~0.85 ms per token, roughly the distance to the floor, is
  non-GEMM work: the fused Mamba step, the norms (62 small kernels), the
  split-K reductions and attention.

## Tools: what each changes and what it costs
Each offered value was validated to build on this engine. The signals a tool
targets are in brackets.

- `engine.precision` (`bf16` | `fp16`; default `bf16`) [`precision`]: weight
  and activation precision. BF16 is the checkpoint's. FP16 has 3 more mantissa
  bits but a narrower range (max 65504); the residual stream peaks at about
  1.5e3 here, so FP16 does not overflow. Both store 2-byte weights, so the
  byte floor is the same; the precision changes accuracy (see the KL ranges
  above) and the GEMM/GEMV tactics TensorRT chooses from. The SSM update and
  the convolution run in FP32 either way.
- `engine.use_fp32_acc` (bool; requires `engine.precision=fp16`) [`precision`,
  `dominant_kernel`]: accumulates FP16 matrix multiplications in FP32 by
  inserting casts around them (BF16 matrix multiplications already accumulate
  in FP32). The casts change the graph TensorRT selects tactics for and
  fuses. Builds with it take longer: about 36-41 s of export + build, against
  28 s for the baseline.
- `workload.ssm_state` (`fp32` | `bf16`; default `fp32`) [`roofline_gap`,
  `copy_overhead`]: storage precision of the 21 SSM states (21 x [96, 80,
  128]; 82.6 MB in FP32), read and written every step. BF16 halves that
  traffic (165 MB to 83 MB per token), which moves the floor from 4.54 to
  4.49 ms. The update itself stays FP32: each step upcasts the stored state,
  updates it and rounds the result back to the storage precision. Accuracy
  risk: the recurrent state is rounded to BF16 every step; watch top-1
  agreement (one flipped token of 64 passes, two fail).
- `workload.state_update` (`functional` | `inplace`; default `functional`)
  [`copy_overhead`, `roofline_gap`]: how the engine writes the new conv and
  SSM states.
  - `functional` returns them as 42 new engine outputs; in the baseline the
    fused Myelin Mamba-step kernel computes the update and emits them. Under
    CUDA graphs every state is then copied into and out of the graph's
    staging buffers each step (about 4x the state bytes of extra traffic).
  - `inplace` lays each state out as a one-slot cache (conv [1, 9728, 1, 4],
    SSM [1, 7680, 1, 128]; the same memory) and writes it into its input
    buffer with `index_copy`, which TensorRT lowers to the same in-place
    IKVCacheUpdateLayer as the KV cache; nothing is copied. The state writes
    become cache updates instead of outputs of the fused Mamba step, which
    changes the graph Myelin fuses around them.
- `engine.build_route` [`dominant_kernel`]: TensorRT Global Performance Tuning
  build routes (Myelin knobs), offered only when the installed TensorRT has
  the feature; one route per recipe.
  - `''` (default): no route.
  - `'-kgen:lower_fc:all=on'`: generates every GEMV with Myelin's kgen code
    generator instead of the cuBLAS-style xmma kernels, which lets it fuse the
    RMSNorms into the GEMVs. It targets the GEMV kernels and the separate norm
    kernels (62 launches, 0.25 ms per token in the baseline).
  - `'-tactic:splitk=off'`: forbids split-K GEMV tactics and with them the
    split-K reduction kernels (55 launches, 0.12 ms per token in the
    baseline).
- `runtime.cudagraphs` (bool; default false) [`launch_overhead`,
  `host_syncs`]: replays the engine from a CUDA graph, removing per-call
  launch and host overhead. This engine is capturable. The baseline profile
  shows no host overhead and no host syncs. With `functional` state updates
  the graph adds the staging copies described above.

### Not offered (known no-ops and build failures)
- Optimization levels 4 and 5: same kernels and same latency as level 3,
  which every candidate uses.
- The build routes `-cask_fusion:fp16_acc=off`, `-peep:match_gemv_mha=on`,
  `-mha_fusion:gemv_mha_v2=off` and `-late_h_fusion=off`: bit-identical
  engines (same 255 kernels and KL as the baseline).
- Split-K off on top of kgen: a no-op, because kgen GEMVs have no split-K.
- Tiling optimization: TensorRT 11.4 cannot build this engine at 'fast',
  'moderate' or 'full' (validateIOTensorAliasing on the aliased KV-cache
  I/O: Myelin's tactic timing does not bind the aliased inputs and outputs to
  the same memory).
- The CUDA Tile route `-kgen:codegen:cuda_tile=2`: fails a Myelin check
  (copy_vtx_analyzer CHECK on the KV-cache update copies).
- `runtime.pre_allocated_outputs`: ignored by engines with aliased KV outputs.

## PyTorch for reference
Hugging Face decode step with its hybrid StaticCache, same protocol and
inputs, BF16 unless stated; compile time is inside build:

| Config | ms/token | top-1 / mean KL | notes |
|---|---|---|---|
| eager | 8.264 | 1.0 / 3.7e-4 | 1302 kernels, 22% host overhead |
| Inductor default | 5.304 | 1.0 / 2.5e-4 | cuBLAS gemvx GEMVs, 5.5 us/layer fused SSM kernel |
| Inductor reduce-overhead | 5.260 | 1.0 / 2.5e-4 | |
| Inductor reduce-overhead + FlexAttention | 5.183 | 1.0 / 2.8e-4 | |
| Inductor max-autotune | 4.859 | 1.0 / 1.0e-4 | Triton GEMVs with norm, residual and relu^2 fused; 303 kernels; 52 s compile |
| Inductor max-autotune + coordinate-descent tuning | 4.856 | 1.0 / 9.8e-5 | |
| Inductor max-autotune + FlexAttention | **4.792** | 1.0 / 7.6e-5 | the fastest PyTorch configuration; 322 kernels; 13.3 GB GPU |
| Inductor reduce-overhead, fp16 | 5.301 | 1.0 / 4.0e-6 | |
| Inductor max-autotune, fp16 | 4.908 | 1.0 / 1.9e-6 | |

The FlexAttention numbers use a corrected decode mask: transformers 5.18's
FlexAttention mask under a StaticCache captures the cache's
`cumulative_length` by reference, and the first attention layer's cache
update advances it before attention runs, so every decode query also
attended to the next, unwritten slot. The workload copies the offset
(`fix_flex_attention_masks`); with the fix FlexAttention's KL matches SDPA's.
Inductor max-autotune with FlexAttention (4.792 ms/token, against the 4.54 ms
floor) is the fastest PyTorch configuration measured: that is the bar.

## Patch slot `decode_step` (agent planner only)
The RMSNorm (`_rms_norm`: the 42 block norms and the final norm) and
Mamba-step (`NemotronDecodeStep._mamba`) formulations in
`tools/llm/nemotron_decode.py` inside EVOLVE-BLOCK fences. `_mamba` covers
the in_proj split, the depthwise conv over the conv-state window, the
softplus/exp decay, the SSM state update h = exp(dt*A)*h + dt*B*x with
y = C.h + D*x, the gated group RMSNorm, out_proj and the state outputs
(written in place with `index_copy` when `workload.state_update=inplace`).
The attention block is outside the slot.

Myelin fuses these ops, so op order, casts, layouts and reductions decide
the non-GEMM kernels; in the baseline those are the Mamba-step, RMSNorm and
gated RMSNorm kernels (about 0.7 ms per token together). Keep the math (FP32
statistics and SSM update), every input and output with its shape and dtype,
and the in-place `index_copy` writes (TensorRT lowers them to in-place
IKVCacheUpdateLayer writes).

A patched build first runs an FP32 formulation gate: the patched and the
original step decode 8 tokens eagerly in FP32 from the same FP32 prefill, and
the build fails (stage `build`, with the measured errors) unless the logits
agree within 1e-3 and every cache and state output within 1e-4 of its
magnitude. BF16 noise can hide a small formulation error from the accuracy
limits (see Accuracy); in FP32 the same error moves the logits by far more
than 1e-3. Gate calibration on the GPU:
- an equivalent rewrite (the gated norm dividing by sqrt instead of
  multiplying by rsqrt) passed with a logit difference of 1.7e-5 and a state
  difference of 1.1e-6 of magnitude, in under 1 s, and its engines built and
  measured like the unpatched ones;
- a rewrite that never updates the SSM state failed the build in 1.3 s
  (logit difference 11, state difference 0.996), before any engine was built.

## How to work
- One hypothesis per trial; say which signal or measurement motivates it.
- Each TensorRT trial takes ~40-70 s (model load + HF prefill ~10 s, export +
  build ~25-41 s, timing ~8 s; plus ~1 s of FP32 gate for a patched build).
- Noise: two fresh-process baseline runs differed by 0.01%, but this shared
  GPU has slowed whole runs by ~7%, and single fresh-process smoke numbers
  differ meaningfully only above ~2%. Differences below the session's
  calibrated threshold are not real: insist on races.
- Revisit a rejected option only when the incumbent changed in a way that
  could change its effect.
- Exporting works: an exported artifact of this engine (about 10.6 GB, pickle
  protocol 4) loads back with `torch_tensorrt.load` in about 24 s and
  reproduces its trial's accuracy exactly.
- Stop when no remaining option can plausibly beat the noise threshold.

## Hybrid tool `kernels.inductor` (TensorRT with Inductor regions)
`kernels.inductor` (`none` | `lm-head` | `mlp-inductor`; default `none`;
requires `workload.state_update=inplace` and `runtime.cudagraphs=true`)
[`dominant_kernel`, `roofline_gap`] hands chosen operations of the decode
step to Inductor-compiled regions inside the compiled program
(`torch_tensorrt.region.execute_in_torch(backend="inductor")`).
- Capture: only strict export captures region scopes, so these builds use
  `torch_tensorrt.compile(..., ir="dynamo", strict=True)` (through a wrapper
  that names each of the 52 inputs, because the tracer cannot key dynamic
  shapes for a `*states` parameter). Regions refuse CPU offload, so the
  BF16 model stays on the GPU next to the engine build; the weights no
  region reads are freed afterwards.
- Compilation: each region is a separate graph that Inductor compiles on its
  first call, during the build, with the options of max-autotune without
  Inductor's CUDA graphs. Its coordinate-descent tuning is what makes
  Inductor lower each batch-1 matrix multiply to a Triton reduction kernel
  (a GEMV) with the preceding RMSNorm fused into it; in default mode Inductor
  calls cuBLAS for these GEMVs instead.
- `lm-head`: the final RMSNorm and the LM-head GEMV (822 MB of weights) as
  one region after the single engine: 1 engine boundary.
- `mlp-inductor`: each of the 17 MLP blocks (block RMSNorm, up GEMV, ReLU²,
  down GEMV, residual add) as its own region. The step becomes 18 engines
  and 17 regions (34 boundaries); the 21 Mamba and 4 attention layers stay in
  TensorRT, between the regions.
- CUDA graphs: the workload records the whole decode step (engines and
  regions) as one CUDA graph over the static cache and state buffers, so a
  boundary costs a gap between kernels instead of a host-side engine or
  region call. Hence the requirements: CUDA graphs, and in-place state
  updates for static state buffers. The library's own whole-graph wrapper
  is not used: it copies every input into its own buffers on each replay,
  so in-place writes would miss the caller's caches. The head-to-head key
  `bench.whole_graph` applies the same capture to the plain session engine,
  as a control.
- Costs and risks: longer builds (strict export and Inductor's autotuning);
  more GPU memory during the build; a kernel gap per boundary; and each
  engine is optimized on its own, so Myelin may fuse the remaining work
  (for example the Mamba state update and its in-place write) differently
  than in the single engine. Accuracy: the regions run BF16 GEMVs with FP32
  accumulation, as TensorRT does.
