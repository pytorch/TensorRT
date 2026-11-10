# Notes for the ModernBERT-base split_rope regions

Each region is one layer's QKV split and rotary embedding in a batch-1 ModernBERT-base forward: from the layer's
Wqkv output `qkv` [1, length, 2304] and its layer type's RoPE `inv_freq` [32], the cos/sin tables of positions
0 .. length - 1 and the rotated query and key, each [1, length, 12, 64], query first. The value stays in TensorRT.

Measured in the whole model (RTX 5090, TensorRT 11.4, FP16, the workload's 512-token input): the region's code rounds
the cos/sin tables to the model dtype (FP16) before the products, as Hugging Face does. A kernel that kept them in
FP32 instead (Inductor's generated kernel for this region) changed the last FP16 bit of part of the query and key
elements. That moved the model's few ill-conditioned output tokens and raised the forward's bulk error (relative L2
over the best 99% of the tokens) to 0.0062-0.0068, against the workload's limit of 0.0075; the same kernel with the
tables rounded as the region's code rounds them left the engine's output bitwise equal to that of the build without
the kernel. Round the tables as the region's code does.
