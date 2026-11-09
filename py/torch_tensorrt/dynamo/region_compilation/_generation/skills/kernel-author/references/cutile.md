# NVIDIA cuTile Python backend

Import `cuda.tile as ct`. Define exactly one `@ct.kernel` function. Its
parameters are the input arrays `arg0`, `arg1`, and so on, then the outputs
`out0`, `out1`, and so on, then one parameter per entry of the proposal's
`constants` array, in that order and with the same names. Array parameters have
no annotations; each constant is annotated `ct.Constant[int]`,
`ct.Constant[float]`, or `ct.Constant[bool]` to match its JSON value, so `256`
needs `ct.Constant[int]` and `0.5` needs `ct.Constant[float]`. JSON need not
keep `256.0` apart from `256`, so a `ct.Constant[float]` constant also accepts
a whole number, and the kernel receives it as a float.

The built-in adapter presents tensors as flattened rank-one views. Use
`ct.bid`, `ct.load`, `ct.store`, and tile operations. A tiled load has the form
`ct.load(array, index=(block,), shape=(BLOCK,))`; interpret the index as a tile
index under the installed cuTile API. Derive broadcast and multidimensional
indexing from the original shapes in the task.

Import cuTile as `import cuda.tile as ct`; `from cuda import tile as ct` is
accepted as well, and no other module may be imported. Besides the kernel,
module scope may define helper functions, decorated `@ct.function` or not
decorated, with plain positional parameters, optionally annotated
`ct.Constant[int]`, `ct.Constant[float]` or `ct.Constant[bool]`, and no
defaults; their bodies follow the kernel's rules. Nothing else belongs at
module scope. Kernels may call `ct` functions, their helpers, tile and array
methods such as `t.astype(...)` or `a.slice(...)`, and the builtins `abs`,
`bool`, `float`, `int`, `len`, `max`, `min`, `range`, and `tuple`. No variable
or parameter may reuse the name of one of these builtins, `ct` or a helper. A
variable named like another builtin, such as `sum`, is usable only after an
assignment earlier in its block or an enclosing one, not one inside an `if`,
loop or comprehension that has ended. `ct` may only name its functions, types
and constants, such as `ct.float32` or `ct.PaddingMode.ZERO`. Output
allocation and `cuda.tile.launch` wrappers are executor responsibilities.
Supply a static grid. The proposal has no `num_warps` or `num_stages` fields;
those are Triton options. cuTile Python and CuTe DSL are separate backends; do
not substitute one API for the other.

cuTile's compiler decides how much shared memory a kernel uses, for example to
stage `ct.mma` operands or transpose a tile, and allocates it statically, so a
plugin launch requests none. `target.plugin_max_shared_memory_bytes` (49152 on
current GPUs) limits the shared memory a launch requests, not this allocation.
On compute capability 10.0 and newer, 12.0 included, the compiler targets the
exact GPU, and a plugin can hold up to the GPU's per-block maximum of static
shared memory. On compute capability 8.x it emits generic PTX; if registration
reports that the CUDA driver could not load the kernel, use smaller tiles.

Preserve reference dtype transitions and numerical behavior. Account for
partial tiles using the bounds behavior supported by the selected operations.
Choose tile shapes and work distribution from correctness and timing feedback.
Only use APIs and options supported by the installed toolchain in the task.

The executor compiles the plugin with layout facts that hold for the captured
shapes, such as unit strides and extents divisible by 16 elements. When the
task's `codegen.aligned_io` is true, it also declares every
array's base address 16-byte aligned, so tiles whose extents and offsets are
multiples of 16 bytes can use 128-bit loads and stores. Otherwise the
evaluator's direct launches use base addresses one element past a 16-byte
boundary, as a plugin can get, so cuTile's launcher compiles and times the
code the plugin runs. The plugin never uses
TMA, which needs host-built descriptors that a TensorRT plugin cannot pass, and
the executor's direct launch compiles without TMA as well, so its timings are
the plugin's. Without TMA a loaded tile is held in registers rather than staged
through shared memory, so a large tile can spill and run many times slower.

For a reference that adds two contiguous float32 tensors of 1000 elements, this
source matches the ABI:

```python
import cuda.tile as ct

@ct.kernel
def add(arg0, arg1, out0, BLOCK: ct.Constant[int]):
    x = ct.load(arg0, index=(ct.bid(0),), shape=(BLOCK,))
    y = ct.load(arg1, index=(ct.bid(0),), shape=(BLOCK,))
    ct.store(out0, index=(ct.bid(0),), tile=x + y)
```

With that text as `source`, the remaining proposal fields are:

```json
{"kernel_name": "add", "grid": [4], "constants": [{"name": "BLOCK", "value": 256}]}
```

Adapt the computation, dtypes, indexing, and launch settings to the actual task;
this example is not evidence of a measured result.
