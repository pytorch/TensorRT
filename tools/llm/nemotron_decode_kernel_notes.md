# Notes for the Nemotron-3-Nano-4B decode MLP regions

These regions run inside a batch-1, one-token decode step of NVIDIA-Nemotron-3-Nano-4B (BF16 weights by default;
hidden size 3136 = 49 x 64, MLP intermediate size 12544, 17 MLP blocks). `mlp_in_<layer>` is the block RMSNorm (FP32
statistics and weight multiply, one rounding to the model dtype), the up GEMV and ReLU^2; `mlp_out_<layer>` is the
down GEMV and the residual add. Each GEMV streams a 12544 x 3136 weight (about 79 MB in BF16) from DRAM once per
token, so these kernels are memory-bound.

Measured on this GPU (RTX 5090) in Qwen3-0.6B decode-block regions of the same kind (an RMSNorm, GEMVs and an
epilogue at batch 1, inside the TensorRT decode engine), not in this model:
- A GEMV reaches about 1.6-1.7 TB/s of DRAM bandwidth, and TensorRT's own GEMVs reach most of it: the gain comes from
  doing the small work inside the same kernel, not from a faster GEMV.
- Load weight tiles with `eviction_policy="evict_first"`. With the default policy the weights, read once per token,
  evicted L2-resident tensors that the surrounding TensorRT kernels reuse: one block cost about +97 us per token in
  the model although it was as fast as TensorRT's GEMV in isolation.
- Recompute the RMSNorm of the input vector in every program instead of a separate pass; issue the weight loads
  first, so the norm's reduction overlaps them.
- One program per block of output rows, each reading full contiguous rows of the [N, K] weight with wide loads; no
  atomics or split-K across programs.
- With FP16 weights, accumulating in FP32 and rounding each output once halved the KL divergence against the FP32
  reference compared with TensorRT's FP16-accumulating kernels.
