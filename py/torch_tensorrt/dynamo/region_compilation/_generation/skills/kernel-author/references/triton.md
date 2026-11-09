# Triton backend

Import the Triton modules with exactly these statements, leaving out any you
do not use:

```python
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
```

`from triton.language.extra.cuda import libdevice`,
`from triton.language.extra import libdevice` and
`from triton import language as tl` are accepted as well. No other module may
be imported.

Define the kernel as the `@triton.jit` function named by `kernel_name`. Its
parameters are the input tensors `arg0`, `arg1`, and so on, then the outputs
`out0`, `out1`, and so on, then one parameter per entry of the proposal's
`constants` array, in that order and with the same names. Tensor parameters
have no annotations; each constant is annotated `tl.constexpr`.

Module scope may also define helper functions, each decorated exactly
`@triton.jit`, with plain positional parameters, optionally annotated
`tl.constexpr`, and no defaults. The kernel and other helpers call them by name
or pass them to operations such as `tl.reduce` and `tl.associative_scan`; their
bodies follow the kernel's rules. Nothing else belongs at module scope: no
assignments or constants, host code, allocations, PyTorch fallbacks, or launch
wrappers. Kernels may call `tl` and `tl.math` functions, `libdevice`
functions, `triton.cdiv`, `triton.next_power_of_2`, their helpers, tensor
methods such as `x.to(tl.float32)`, and the builtins `abs`, `bool`, `float`,
`int`, `len`, `max`, `min`, `range`, and `tuple`. No variable or parameter may
reuse the name of one of these builtins, a module or a helper. A variable named
like another builtin, such as `sum`, is usable only after an assignment earlier
in its block or an enclosing one, not one inside an `if`, loop or comprehension
that has ended. The modules may only name their functions, types and
constants, such as `tl.float32`. Dimensions and launch grid
are static. Supply literal grid dimensions and supported `num_warps` (1, 2, 4,
8, 16, 32, or null) and `num_stages` (positive, or null for Triton's default
of 3) in the proposal.

A TensorRT plugin launch gives each block at most
`target.plugin_max_shared_memory_bytes` bytes of shared memory (49152 on
current GPUs). This is a hard limit: TensorRT does not opt plugin kernels into
more, as Triton's own launcher does, so a kernel above it builds an engine that
fails at every launch. Triton places `tl.dot` operand tiles, layout
conversions, reductions and pipelined loads in shared memory, with pipelined
buffers multiplied by `num_stages`; choose tile shapes and stages that fit from
the first attempt. If a candidate does not fit, the evaluator compiles smaller
variants of it, with one stage and then with halved `tl.arange` tile
constants, and evaluates the first that fits; its report's `fit_log` lists what
it tried, and that variant is the candidate recorded. Halving never changes the
literal grid, so it skips tiles that scale `tl.program_id`.

Use masked boundary loads and stores for partial tiles. When vectorizing a
broadcast, derive each input's index from its original shape rather than
assuming every flattened tensor has the output's size. For reductions, choose
identity values for masked lanes that preserve the operation.

Preserve intermediate rounding from the reference. For exact float32 rounding,
the `libdevice` module imported above offers `add_rn` and `mul_rn`. There is
no `enable_fp_fusion` option in this interface.
Do not invent backend compiler options or use inline assembly.

Tune tile dimensions and supported warp/stage settings using measured feedback
and the supplied target. Larger blocks can reduce launch count but increase
register pressure or waste work on masked lanes; select them for this region.

When the task's `codegen.aligned_io` is true, the executor compiles the plugin
with every tensor pointer known to be 16-byte aligned, as Triton's own launcher
does for aligned tensors, so contiguous accesses can use 128-bit loads and
stores. Otherwise pointers are only aligned to their element size; do not claim
more with `tl.multiple_of` on a pointer. The evaluator's direct launches then
use pointers one element past a 16-byte boundary, so that Triton's launcher
compiles, checks and times the code the plugin runs.

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
