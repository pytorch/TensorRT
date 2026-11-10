# Research brief: Gemma-3-1B batch-1 prefill (8192 tokens) with Torch-TensorRT

## Objective
Minimize milliseconds per prefill (the time-to-first-token computation) of
`unsloth/gemma-3-1b-it` on one RTX 5090. Its weights are byte-identical to
`google/gemma-3-1b-it`. The model has 26 layers, hidden 1152, 4 query heads
and 1 KV head of dimension 256, and a 262k vocabulary. Its 22 local layers use
a 512-token sliding window; layers 5, 11, 17 and 23 attend globally.

The unit is one forward over a fixed 8192-token prompt (CPython's Python
language reference text, BOS first, no paragraph repeated; prompt_sha256
afe00cb828bb120f). It returns logits at 64 evenly spaced positions
(127, 255, ..., 8191).
Workload: `tools/llm/agentic_prefill.py:create_case` with
`{"model": "unsloth/gemma-3-1b-it", "length": 8192}`.

## Accuracy
The candidate is checked against an FP32 eager reference (Hugging Face SDPA,
no TF32) at the 64 positions. The limits are the workload defaults:
top-1 agreement >= 0.90, mean KL <= 0.005, and largest KL at any one
position <= 0.1.

Measured values (top-1 / mean KL / largest KL):
- Correct BF16 builds: 0.98-1.0 / 3e-4 to 8e-4 / 0.002-0.025 (17 builds:
  TensorRT, eager, Inductor).
- Correct FP16 builds: 1.0 / 3e-5 to 1.6e-4 / 4e-4 to 0.003 (9 builds).
- Broken attention (eager BF16):
  - local layers without their window: 0.17 / 4.44 / 13.9;
  - window doubled: 0.17 / 4.26 / 13.4;
  - global layers limited to 512 tokens: 0.81 / 0.56 / 10.4;
  - global layers limited to 2048 tokens: 0.875 / 0.21 / 6.2.
- A window off by one token measures like a correct build
  (0.98-1.0 / 4.8e-4-5.4e-4 / <= 0.004), because BF16 noise hides it. The
  workload's unit tests check each lowering's window exactly; any new
  attention lowering needs the same check.

## What bounds the answer
- Prefill is compute-bound. One prefill needs 12.39 TFLOP:
  - linear layers: 11.43 TFLOP;
  - attention: 0.92 TFLOP with the causal and sliding-window masked pairs
    skipped, or 7.15 TFLOP if every layer computes the full 8192 x 8192
    square (7.8x more);
  - LM head: 0.04 TFLOP.
- Measured dense GEMM rates and the floors they give:
  - BF16 with FP32 accumulation: 233 TFLOPS, floor 53.1 ms.
  - FP16 with FP16 accumulation (cuBLAS): 354 TFLOPS, floor 35.0 ms.

  Each trial's `structure` reports `gflop_per_unit`, `compute_floor_ms` and
  `compute_floor_ms_fp16_acc`.
- Memory does not bound the GEMMs: reading 2.0 GB of weights takes 1.24 ms at
  1.61 TB/s. It does bound the elementwise kernels (see GELU below).

## What profiling showed
Baseline (documented compile, BF16): 141.5 ms in one engine of 369 kernels.
It is GPU-bound: host overhead 0.6%, no host syncs, no copies.
- Fused attention `_gemm_mha_v2`: 80.7 ms (57%, 26 calls), about 88 TFLOPS
  dense-equivalent.
- GEMMs: 52 ms (about 220 TFLOPS).
- GELU(gate)*up: 4.95 ms as a separate kernel. This is about DRAM bandwidth:
  each layer moves about 340 MB, too much for the 96 MB L2.
- RMSNorms: 1.2 ms.

The attention runs dense over the full 8192 x 8192 square in all 26 layers:
- Under torch.export, Hugging Face materializes its masks. It ANDs the causal
  and window pattern with a packed-sequence term (a cumsum over the position
  ids), and TensorRT's fused-attention mask analysis cannot read that term.
- In isolated one-layer micro-benchmarks, TensorRT's fused attention skipped
  fully masked tiles when the mask came from position ranges only: 1.54 ms
  causal and 0.36 ms window-512, against 2.65 ms dense. Building the
  8192 x 8192 mask tensor itself costs about 0.25 ms.

Attention kernels for head_dim 256 on this GPU (from end-to-end profiles,
TensorRT measured with position-range masks):
- BF16: PyTorch's flash attention (FA2) takes 0.89 ms per causal layer
  against TensorRT's 1.36, and 0.10 vs 0.24 ms per window-512 layer.
- FP16: TensorRT takes 1.01 ms per causal layer and 0.135 ms per window
  layer (7.0 ms in all); FA2 takes 0.89 and 0.10 ms (5.8 ms in all).

FP16 engines use FP16-accumulation GEMM tactics (`f16f16_f16f16_f16`), at
about 350 TFLOPS.

FP16 range (+-65504):
- The residual stream peaks at 61,040 at the BOS token (layer 22). The peak is
  the same for every prompt, because position 0 attends only to itself.
- Content tokens peak at 48,800 (74% of the FP16 maximum, at position 3002).
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
    per chunk of B queries). It needs no 8192 x 8192 mask, but some masked
    pairs remain;
  - `torch-flash`: FA2 with its sliding window, outside TensorRT.

  With PyTorch attention in every layer the model becomes 27 engines plus 26
  PyTorch partitions, and the build takes 85-110 s instead of 20-30 s.
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
  with latency within +-0.5%. A single-engine build has little host overhead
  to remove. Engines split by PyTorch attention have more launch overhead.

Not in the catalog:
- A fused GELU(gate)*up GEMM epilogue (the only way to remove that
  bandwidth-bound kernel) or a Triton attention region.
- Running each MLP in 2 or 4 sequence chunks to keep gate/up in L2 was slower
  (+2% and +7%): TensorRT re-fuses the chunks' GELU work into one
  whole-sequence kernel, and the smaller GEMMs cost more.

## Reference points
Head-to-head smoke numbers, each a fresh process, taken while other jobs
shared the GPU.
- TensorRT baseline (documented compile, BF16): 141.5 ms.
- PyTorch, BF16:
  - eager 185.5 ms; Inductor default 186.2 and max-autotune 178.8. Under
    torch.compile, Hugging Face also materializes every mask, so SDPA runs
    dense.
  - FlexAttention: eager 89.9, Inductor default 73.5, max-autotune 68.5
    (with coordinate-descent tuning 68.8).
  - This workload's `torch-flash` lowering: eager 84.8, Inductor
    max-autotune 65.5. This is the strongest BF16 PyTorch configuration.
- PyTorch, FP16 with cuBLAS FP16 accumulation (as TensorRT's FP16 tactics):
  - eager 169.3; Inductor max-autotune 166.8 (dense SDPA).
  - With FlexAttention: 55.4.
  - With the `torch-flash` lowering: 52.6 (52.8 with coordinate-descent
    tuning). This is the strongest PyTorch configuration measured.

## How to work
- One hypothesis per trial. Name the signal or kernel it targets and what the
  profile predicts.
- Each trial loads the 2 GB model and builds: 20-30 s, or 85-110 s with
  PyTorch attention in every layer. It also times about 30 prefills of
  50-140 ms each.
- Differences below the calibrated noise threshold are not real. Revisit a
  rejected option only when the incumbent changed in a way that could change
  its effect.
- Stop when no remaining option can plausibly clear the threshold.
