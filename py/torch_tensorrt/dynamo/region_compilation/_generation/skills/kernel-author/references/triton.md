# Triton backend

Import `triton` and `triton.language as tl`. Define exactly one `@triton.jit`
function. Tensor parameters have no annotations; compile-time parameters use
`tl.constexpr`. Follow the task's ordered tensor and constant parameter ABI.

The built-in backend permits imports and the single kernel definition at module
scope. Do not add helper functions, host assignments, allocations, PyTorch
fallbacks, or launch wrappers. Dimensions and launch grid are static. Supply
literal grid dimensions and supported `num_warps`/`num_stages` in the proposal.

Use masked boundary loads and stores for partial tiles. When vectorizing a
broadcast, derive each input's index from its original shape rather than
assuming every flattened tensor has the output's size. For reductions, choose
identity values for masked lanes that preserve the operation.

Preserve intermediate rounding from the reference. For exact float32 rounding,
the permitted `triton.language.extra.cuda.libdevice as libdevice` import offers
`add_rn` and `mul_rn`. There is no `enable_fp_fusion` option in this interface.
Do not invent backend compiler options or use inline assembly.

Tune tile dimensions and supported warp/stage settings using measured feedback
and the supplied target. Larger blocks can reduce launch count but increase
register pressure or waste work on masked lanes; select them for this region.
