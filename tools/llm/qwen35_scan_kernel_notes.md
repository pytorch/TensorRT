# Notes for the Qwen3.5-0.8B gdn_scan regions

Each region is the inter-chunk state recurrence of one Gated DeltaNet layer in a batch-1 Qwen3.5-0.8B prefill, in
FP32: for every (batch, head) row, from S_0 = 0, v_n = u_n - w_n S_n and S_{n+1} = S_n * gamma_n + k_n^T v_n over the
chunks n. It returns every chunk's start state S_n [batch, heads, chunks, 128, 128] and v_n [batch, heads, chunks,
chunk, 128]; at 2048 tokens: 16 heads, 32 chunks of 64 tokens, key and value head dims 128. The chunks are
sequential; the rows, and the value columns of a row, are independent of each other.

Measured in an isolated sweep of the workload's hand-written Triton kernel for this region on this GPU (RTX 5090,
2048 tokens): one program per 16 value columns of one row, keeping its 128 x 16 FP32 state on chip across all chunks
and processing a chunk's 64 rows in one step, took about 200 us per layer with IEEE FP32 dots (8 warps) and 137 us
with TF32 dots (4 warps); wider value blocks ran slower.
