# Research brief: Gemma-3-1B batch-1 prefill (2048 tokens) with Torch-TensorRT

## Objective
Minimize milliseconds per prefill (the time-to-first-token computation) of
`unsloth/gemma-3-1b-it` on one RTX 5090. Its weights are byte-identical to
`google/gemma-3-1b-it`. The model has 26 layers, hidden 1152, 4 query heads
and 1 KV head of dimension 256, and a 262k vocabulary. Its 22 local layers use
a 512-token sliding window; layers 5, 11, 17 and 23 attend globally.

The unit is one forward over a fixed 2048-token prompt (CPython's Python
language reference text, BOS first, prompt_sha256 17c046bd5bad1aa6). It
returns logits at 64 evenly spaced positions (31, 63, ..., 2047).
Workload: `tools/llm/agentic_prefill.py:create_case` with
`{"model": "unsloth/gemma-3-1b-it", "length": 2048}`.

## Accuracy
The candidate is checked against an FP32 eager reference (Hugging Face SDPA,
no TF32) at the 64 positions. The limits are the workload defaults:
top-1 agreement >= 0.90, mean KL <= 0.005, and largest KL at any one
position <= 0.1. Do not tighten top-1: BF16 alone flips 1-3 of the 64
near-ties.

Measured values (top-1 / mean KL / largest KL):
- Correct BF16 builds: 0.95-1.0 / 4e-4 to 1.1e-3 / 0.003-0.023 (17 builds:
  TensorRT, eager, Inductor). One custom eager SDPA lowering reached
  1.2e-3 / 0.035.
- Correct FP16 builds: 0.97-1.0 / 3e-5 to 2.5e-4 / 4e-4 to 0.006 (9 builds).
- Broken attention (eager BF16):
  - local layers without their window: 0.39 / 2.73 / 13.1;
  - window doubled: 0.42 / 2.73 / 12.6;
  - global layers limited to 512 tokens: 0.84 / 0.38 / 10.4.
- A window off by one token measures like a correct build
  (0.97-0.98 / 7.5e-4-8.4e-4 / <= 0.011), because BF16 noise hides it end to
  end. The workload's unit tests check each lowering's window exactly; any new
  attention lowering needs the same check.

## What bounds the answer
- Prefill is compute-bound. One prefill needs 3.01 TFLOP:
  - linear layers: 2.86 TFLOP;
  - attention: 0.117 TFLOP with the causal and sliding-window masked pairs
    skipped, or 0.447 TFLOP if every layer computes the full 2048 x 2048
    square;
  - LM head: 0.039 TFLOP (64 rows of the 262k vocabulary).
- Measured dense GEMM rates and the floors they give:
  - BF16 with FP32 accumulation: 234 TFLOPS, floor 12.9 ms.
  - FP16 with FP16 accumulation (cuBLAS): 357 TFLOPS, floor 8.4 ms.

  Each trial's `structure` reports `gflop_per_unit`, `compute_floor_ms` and
  `compute_floor_ms_fp16_acc`.
- Memory does not bound it: reading 2.0 GB of weights takes 1.24 ms at the
  measured 1.61 TB/s. Prefill declares no bytes per unit, so trials have no
  `roofline` field.

## What profiling showed
Baseline (documented compile, BF16): 20.4 ms in one engine of 369 kernels.
It is GPU-bound: host overhead 0.6%, no host syncs, no copies.
- GEMMs: 13.9 ms (68%), at about 200-220 TFLOPS (bf16 tiles with FP32
  accumulation).
- Fused attention `_gemm_mha_v2`: 5.07 ms (25%, 26 calls).
- RMSNorms: 0.39 ms; LM head GEMM: 0.41 ms.
- GELU(gate)*up: 0.37 ms as a separate kernel. At 2k its inputs (56 MB) stay
  in the 96 MB L2.

The attention runs dense over the full 2048 x 2048 square in all 26 layers:
- Under torch.export, Hugging Face materializes its masks. It ANDs the causal
  and window pattern with a packed-sequence term (a cumsum over the position
  ids), and TensorRT's fused-attention mask analysis cannot read that term.
- In isolated one-layer micro-benchmarks at 8k, TensorRT's fused attention
  skipped fully masked tiles when the mask came from position ranges only:
  1.54 ms causal and 0.36 ms window-512, against 2.65 ms dense. Building the
  mask tensor itself costs about 0.25 ms at 8k.

Attention kernels for head_dim 256 on this GPU (from end-to-end profiles):
- PyTorch's flash attention (FA2) beats TensorRT's fused attention in BF16 at
  8k: 0.89 vs 1.36 ms per causal layer, and 0.10 vs 0.24 ms per window-512
  layer.
- At 2k in FP16 there is no gap: all attention takes 1.48 ms with FA2
  (including split-KV combine kernels) against 1.36 ms in TensorRT (with
  position-range masks and causal global layers).

FP16 engines use FP16-accumulation GEMM tactics (`f16f16_f16f16_f16`), at
about 310 TFLOPS at this length.

FP16 range (+-65504):
- The residual stream peaks at 61,040 at the BOS token (layer 22). The peak is
  the same for every prompt, because position 0 attends only to itself.
- Content tokens peak at 48,800 (74% of the FP16 maximum).
- The largest MLP activation is 380. No FP16 activation is non-finite on this
  prompt.

## Tools: what each changes and what it costs
- `workload.masks` (`hf` | `positions`): `positions` builds the causal and
  window masks from position ranges only. It is unit-tested equal to Hugging
  Face's masks and has no effect when no layer uses a mask.
- `workload.global_attention` (`mask` | `causal` | `torch-flash`), for the 4
  global layers:
  - `causal`: SDPA `is_causal` with native GQA (no mask tensor, K/V not
    repeated);
  - `torch-flash`: PyTorch's FA2 in a custom operator outside TensorRT, which
    splits the engine at each such layer.
- `workload.local_attention` (`mask` | `banded-128` | `banded-512` |
  `torch-flash`), for the 22 window layers:
  - `banded-B`: chunked window attention as one batched SDPA (B + 512 keys
    per chunk of B queries). It needs no 2048 x 2048 mask, but some masked
    pairs remain;
  - `torch-flash`: FA2 with its sliding window, outside TensorRT.

  With PyTorch attention in every layer the model becomes 27 engines plus 26
  PyTorch partitions, and the build takes 85-110 s instead of about 20 s.
- `engine.precision` (`bf16` | `fp16`): FP16 GEMMs accumulate in FP16 at
  about 1.5x the BF16 rate. FP16 has 3 more mantissa bits than BF16 but a
  smaller range (see the peaks above). An overflow shows as inf/NaN, which
  gives an infinite KL and a rejection.
- `engine.build_route` (`''` | `'-cask:fp16_acc=off'`, requires fp16): an
  FP16 engine with FP32-accumulation GEMMs, slower and more accurate. Other
  Myelin routes are not offered: the attention-fusion variants built the
  same kernels, and split-K off changed one GEMM's layout, all with
  bit-identical logits and latency within 1%.
- `engine.optimization_level` (3 | 5 | 4),
  `engine.tiling_optimization_level` (none | moderate | fast | full),
  `runtime.cudagraphs` and `runtime.pre_allocated_outputs`: every value was
  validated on one FP16 recipe at 2k and 8k. Each built in 23-30 s, tiling
  `full` included, and gave the same accuracy metrics as the default build,
  with latency within +-0.5%. A single-engine build has under 1% host
  overhead and no syncs, so the runtime options have little to remove there.
  Engines split by PyTorch attention have more launch overhead.

Not in the catalog:
- A fused GELU(gate)*up GEMM epilogue or a Triton attention region.
- Running the MLP in sequence chunks was tried (at 8k) and was slower:
  TensorRT re-fuses the chunks' elementwise work.

## Reference points
Head-to-head smoke numbers, each a fresh process, taken while other jobs
shared the GPU.
- TensorRT baseline (documented compile, BF16): 20.4 ms.
- PyTorch, BF16:
  - eager 29.7 ms; Inductor default 26.6 and max-autotune 24.4. Under
    torch.compile, Hugging Face also materializes every mask, so SDPA runs
    dense.
  - FlexAttention: eager 25.0, Inductor default 19.2, max-autotune 17.9
    (with coordinate-descent tuning 17.8).
  - Inductor max-autotune with this workload's `torch-flash` lowering: 17.1.
    This is the strongest BF16 PyTorch configuration.
- PyTorch, FP16 with cuBLAS FP16 accumulation (as TensorRT's FP16 tactics):
  - eager 24.5; Inductor max-autotune 20.0 (dense SDPA).
  - With FlexAttention: 13.3.
  - With the `torch-flash` lowering: 13.2, or 12.9 with coordinate-descent
    tuning. This is the strongest PyTorch configuration measured.

## How to work
- One hypothesis per trial. Name the signal or kernel it targets and what the
  profile predicts.
- Each trial loads the 2 GB model and builds: about 20-30 s, or 85-110 s with
  PyTorch attention in every layer. Timing then takes a few seconds.
- Differences below the calibrated noise threshold are not real. Revisit a
  rejected option only when the incumbent changed in a way that could change
  its effect.
- Stop when no remaining option can plausibly clear the threshold.
