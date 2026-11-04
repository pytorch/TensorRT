# Triton backend

Import `triton` and `triton.language as tl`. Define exactly one `@triton.jit`
function. Its parameters are the input tensors `arg0`, `arg1`, and so on, then
the outputs `out0`, `out1`, and so on, then one parameter per entry of the
proposal's `constants` array, in that order and with the same names. Tensor
parameters have no annotations; each constant is annotated `tl.constexpr`.

The built-in backend permits imports and the single kernel definition at module
scope. Do not add helper functions, host assignments, allocations, PyTorch
fallbacks, or launch wrappers. Kernels may call `tl` and `tl.math` functions,
`libdevice` functions, `triton.cdiv`, `triton.next_power_of_2`, tensor methods
such as `x.to(tl.float32)`, and the builtins `abs`, `bool`, `float`, `int`,
`len`, `max`, `min`, `range`, and `tuple`. Dimensions and launch grid are
static. Supply literal grid dimensions and supported `num_warps` (1, 2, 4, 8,
16, 32, or null) and `num_stages` (positive or null) in the proposal.

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

For a reference that adds two contiguous float32 tensors of 1000 elements, this
source matches the ABI:

```python
import triton
import triton.language as tl

@triton.jit
def add(arg0, arg1, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(arg0 + offsets, mask)
    y = tl.load(arg1 + offsets, mask)
    tl.store(out0 + offsets, x + y, mask)
```

With that text as `source`, the remaining proposal fields are:

```json
{"kernel_name": "add", "grid": [4], "constants": [{"name": "N", "value": 1000}, {"name": "BLOCK", "value": 256}], "num_warps": 4, "num_stages": null}
```

Adapt the computation, dtypes, indexing, and launch settings to the actual task;
this example is not evidence of a measured result.
