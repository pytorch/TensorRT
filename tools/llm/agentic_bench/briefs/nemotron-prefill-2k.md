# Research brief: Nemotron-3-Nano-4B batch-1 prefill (2048 tokens) with Torch-TensorRT

## Objective
Minimize milliseconds per prefill (time to first token) of a 2048-token
prompt at batch 1 for nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 on one RTX 5090,
without breaking accuracy: mean KL <= 0.002 and top-1 agreement >= 0.85
against the FP32 eager model at 64 evenly spaced positions (31, 63, ...,
2047). Workload: `tools/llm/nemotron_decode.py:create_prefill_case`
(`NemotronPrefillCase`); the unit is one prefill.

## Accuracy
The prompt is 2048 tokens of new, non-repeating prose (`PREFILL_TEXT`, 2181
tokens long; no model has memorized it), so every checked position predicts
new text: the FP32 model's median top-1 probability is 0.43, and 13 of the 64
positions have their top two tokens within 0.1 nats. Correct builds therefore
flip a few top-1 tokens, so top-1 only guards against gross errors. Mean KL
is the decisive check.

Correct builds (TensorRT, eager and Inductor), top-1 / mean KL:
- BF16: 0.89-0.97 / 3.3e-4 to 5.5e-4. The TensorRT BF16 baseline measures
  0.969 / 5.5e-4, BF16 eager 0.922 / 5.4e-4.
- FP16 with FP16 accumulation: 0.94-0.95 / 1.1e-4 to 5.9e-4.
- FP16 with FP32 accumulation lands far closer to the reference than either,
  and still flips one top-1 token.

Broken scans fail the KL limit by a wide margin. Injected SSD scan errors
measure a mean KL of 0.31 (no state passing between 256-token chunks), 1.09
(the last Mamba layer drops its D skip term) and 8.2 (every layer drops it).

The reference records its tokens and positions, and `compare` refuses one
computed for other inputs.

## The model and the compiled program
- 42 layers: 21 Mamba-2 mixers, 17 squared-ReLU MLPs, 4 GQA attention layers
  (40 query / 8 KV heads, head_dim 128, no positional encoding); hidden 3136;
  Mamba: 96 heads x head_dim 80, 8 groups, state 128, conv kernel 4, chunk 256.
- Every candidate compiles `PrefillLogits`: Hugging Face's own full-sequence
  forward (no cache; fla/mamba_ssm are not installed, so the Mamba layers run
  the PyTorch chunked SSD scan: pad, chunk, cumsum, segment-sum masks, exp,
  five einsums in FP32) and the LM head on the 64 checked positions only.
  It exports with torch.export and compiles into ONE TensorRT engine
  (865 layers, 0 PyTorch partitions) in 35-50 s.

## What bounds the answer (compute, not bandwidth)
- Linear layers: 2 x 3.15 G parameters x 2048 tokens = 12.9 TFLOP of GEMMs
  (M = 2048; in_proj 3136x17504, out_proj 7680x3136, MLP 3136x12544 and
  back, q/k/v/o). LM head on 64 rows: 0.05 TFLOP.
- SSD scan in FP32 (per layer, chunk 256): C.B scores 12.9 GFLOP (computed per
  HEAD although B and C are shared by the 12 heads of each group: 12x
  redundant), intra-chunk output 8.1, chunk states 4.0, state-to-output 4.0,
  state passing 0.2: ~29 GFLOP x 21 = 0.61 TFLOP; ~0.36 TFLOP with per-group
  C.B. Chunk 128: 0.40 TFLOP; chunk 64: 0.32 TFLOP (HF formulation).
- Attention: 4 layers x 2048^2 x 40 heads: 0.17 TFLOP causal (0.34 dense).
- Weights are read once per prefill (7.1 GB, ~4.4 ms at 1.61 TB/s):
  irrelevant. The `roofline` field is not reported for this workload.
- Measured on this GPU: ~220-233 TFLOPS for BF16 or FP16 GEMMs with FP32
  accumulation, ~390 TFLOPS for FP16 GEMMs with FP16 accumulation (GeForce
  Blackwell runs FP16-accumulate tensor-core math at twice the rate), 68
  TFLOPS FP32 without TF32, 103 TFLOPS TF32. Floor with FP32 accumulation:
  GEMMs 12.9 TFLOP / ~225 TFLOPS = 57 ms, plus the SSD scan (~5 ms with
  per-group C.B), plus attention (~1 ms): about 63 ms. With FP16
  accumulation the GEMMs need ~33 ms: about 39 ms.

## What profiling showed
Smoke numbers on a shared GPU. TensorRT baseline (BF16, HF scan, chunk 256,
no TF32): 137.9 ms per 2048-token prefill (14.9 k tokens/s), 863 kernels,
host overhead 0.3%, build 35-45 s. Two fresh-process runs of the baseline
differed by 0.08%.

| Kernel (per prefill) | ms | share | launches |
|---|---|---|---|
| BF16 xmma GEMMs (64x96 and 128x64 tiles) | 58.6 | 43% | 105 |
| Myelin `Cums`: HF's stable segment sum (cumsum over the expanded, masked [256, 256] tensor of every head and chunk) | 18.6 | 13.5% | 21 |
| FP32 xmma GEMM on CUDA cores (ffma): the SSD einsums | 13.4 | 9.8% | 63 |
| Myelin `SeleExpTranMul`: exp of the masked segment sums | 8.3 | 6.0% | 21 |
| Myelin transposes/reshapes around the einsums | 5.7+ | 4.2%+ | 21+ |
| everything else (norms, conv, attention, LM head, ...) | ~33 | 24% | ~630 |

The GEMMs already run at the FP32-accumulation tensor-core peak; HF's
chunked SSD scan costs ~50 ms, most of it in work the math does not need:
the O(chunk^2) cumulative sum, per-head C.B scores (12x the per-group work)
and layout shuffles.

## Tools: what each changes and what it costs
Each value was validated on its own: it builds into one engine (35-50 s) and
passes the accuracy limits.

- `engine.precision` (`bf16` | `fp16`; default `bf16`, the checkpoint's):
  weight and activation precision. FP16 has more mantissa bits than BF16 but
  a narrower range; the SSD scan runs in FP32 either way (Hugging Face
  upcasts it). Without FP32 accumulation, TensorRT builds the FP16 GEMMs
  from FP16-accumulate tactics (`f16f16_f16f16`); see the measured
  FP16-accumulation GEMM rate above. Targets the BF16 GEMMs (43% of the
  baseline).
- `engine.use_fp32_acc` (`false` | `true`; default `false`; requires
  `engine.precision=fp16`): FP16 matrix multiplications accumulate in FP32,
  through inserted casts (BF16 GEMMs accumulate in FP32 already). FP16 GEMMs
  with FP32 accumulation measured the same rate as BF16 GEMMs (above). It is
  an accuracy lever (see Accuracy).
- `engine.tf32` (`false` | `true`; default `false`): runs the FP32 matrix
  multiplications, the SSD scan's five einsums (C.B scores, intra-chunk
  output, chunk states, state-to-output, state passing), on TF32 tensor
  cores instead of FP32 CUDA cores (103 against 68 TFLOPS measured).
  Targets the FP32 xmma (ffma) GEMMs (13.4 ms of the baseline). TF32 rounds
  the einsum inputs to a shorter mantissa. Applies to every backend
  (`torch.backends` `allow_tf32` for eager and Inductor).
- `workload.ssd` (`hf` | `segsum` | `grouped`; default `hf`): the
  formulation of the chunked SSD scan. All three are exact in real
  arithmetic (unit-tested against Hugging Face's scan, including padding and
  initial states) and apply to the eager and Inductor backends too.
  - `hf`: Hugging Face's PyTorch fallback. Its stable segment sums take a
    cumulative sum over an expanded, masked [chunk, chunk] tensor per head
    and chunk, and it repeats B and C for every head.
  - `segsum`: segment sums as differences of one cumulative sum per chunk,
    as mamba_ssm's kernels compute them. Targets the Myelin `Cums` kernel
    (18.6 ms of the baseline).
  - `grouped`: `segsum` plus the C.B scores, chunk states and state outputs
    per group of 12 heads instead of per head: 12x fewer score FLOPs (the
    scan needs ~0.36 instead of ~0.61 TFLOP at chunk 256). It also targets
    the FP32 einsums.
- `workload.chunk_size` (`256` | `128` | `64`; default `256`, the
  checkpoint's): the scan's chunk length. Mathematically equivalent
  (unit-tested): smaller chunks shrink the quadratic intra-chunk tensors
  ([heads, chunk, chunk] per chunk; 201 MB per layer in FP32 at 256) and
  their FLOPs (HF formulation: 0.61, 0.40 and 0.32 TFLOP at 256, 128 and
  64), at the cost of more inter-chunk state passing. In the baseline
  profile, the `Cums` and `SeleExpTranMul` kernels work on these tensors.
  Applies to every backend.

`engine.tf32`, `workload.ssd` and `workload.chunk_size` all act on the SSD
scan's FP32 work, so the effect of one can depend on the others.

Not offered, because they measured as no-ops on the baseline:
`runtime.cudagraphs` (one engine call per prefill; host overhead is 0.3%)
and `engine.optimization_level` 5 (the same kernels as level 3). Level 4 is
not offered either.

## PyTorch for reference
Same `PrefillLogits`, same tokens and outputs; compile time inside build:

| Config | ms/prefill | top-1 / mean KL | notes |
|---|---|---|---|
| eager, BF16 | 178.2 | 0.922 / 5.4e-4 | |
| Inductor default | 125.4 | 0.922 / 3.3e-4 | compile 22 s |
| Inductor reduce-overhead | 125.0 | 0.922 / 3.3e-4 | |
| Inductor max-autotune | 120.1 | 0.922 / 3.3e-4 | compile 77 s |
| Inductor max-autotune + coordinate-descent tuning | 120.4 | 0.891 / 3.4e-4 | compile 5 s, from the GEMM autotuning cache of the previous run |
| Inductor max-autotune + FlexAttention | 117.1 | 0.922 / 3.3e-4 | compile 62 s |
| Inductor default + `workload.ssd=grouped` | 93.4 | 0.953 / 3.4e-4 | |
| Inductor max-autotune + `workload.ssd=grouped` | 88.7 | 0.938 / 3.5e-4 | the strongest BF16 PyTorch configuration measured |
| Inductor max-autotune, fp16 + FP16 accumulation | 99.7 | 0.953 / 1.2e-4 | `bench.fp16_accumulation`: cuBLAS FP16-accumulate GEMMs, the analog of TensorRT's plain fp16; compile 110 s |
| Inductor max-autotune, fp16 + FP16 accumulation + ssd grouped | 68.7 | 0.938 / 1.1e-4 | compile 86 s |
| Inductor max-autotune, fp16 + FP16 accumulation + ssd grouped + chunk 128 | 67.1 | 0.953 / 1.2e-4 | the strongest PyTorch configuration measured; compile 80 s |

With Hugging Face's scan, Inductor is 13% ahead of TensorRT's BF16 baseline
(120.1 vs 137.9 ms): Myelin spends 18.6 ms on the masked O(chunk^2)
cumulative sum that Inductor fuses better. Compile times share one Inductor
cache per workload, so later max-autotune variants reuse earlier GEMM
autotuning (the coordinate-descent run compiled in 5 s): they are not cold
compile times.

## How to work
- One hypothesis per trial; say which signal or measurement motivates it.
- The baseline is one GPU-bound engine (host overhead 0.3%): its time is in
  the GEMMs and the SSD scan's kernels, not in launches.
- Each trial takes ~1-1.5 min (model load ~5 s, export ~5 s, build 35-50 s,
  timing ~3 s). Differences below the session's calibrated noise threshold
  are not real.
- Revisit a rejected option only when the incumbent changed in a way that
  could change its effect.
- Stop when no remaining option can plausibly beat the noise threshold.
