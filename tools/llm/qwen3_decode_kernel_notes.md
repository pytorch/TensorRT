# Notes for the Qwen3-0.6B decode-block regions (measured, RTX 5090)

These regions run inside a batch-1, one-token decode step of Qwen3-0.6B (FP16 weights, hidden size 1024,
intermediate size 3072, 16 query heads and 8 KV heads of 128). Each region is a matrix-vector product (GEMV)
with the small work around it: an RMSNorm before it, a residual add or SiLU(gate) * up after it, or the
per-head q/k RMSNorm + rotary embedding. A GEMV streams its whole weight matrix from DRAM once per token, so
these kernels are memory-bound: the reachable rate is about 1.6-1.7 TB/s, and TensorRT's own GEMVs already
reach most of it. The gain comes from doing the small work inside the same kernel, not from a faster GEMV.

Measured in the full model (the region benchmark alone does not show these):
- Load weight tiles with `eviction_policy="evict_first"`. The weights are read once per token; with the
  default policy they evicted the L2-resident KV cache and small tensors that the surrounding TensorRT kernels
  reuse, and one block cost about +97 us per token although it was as fast as TensorRT's GEMV in isolation.
- Accumulate in FP32 and round each output to FP16 once; this halved the KL divergence against the FP32
  reference compared with TensorRT's FP16-accumulating kernels.
- Recompute the RMSNorm of the input vector in every program instead of a separate pass: issue the weight
  loads first, so the norm's reduction overlaps them.
- One program per block of output rows, each reading full contiguous rows of the [N, K] weight with wide
  loads; no atomics or split-K across programs.
- Write each output once. Keep the outputs in the order the region produces them.
