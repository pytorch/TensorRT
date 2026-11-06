# NVIDIA cuTile Python backend

Import `cuda.tile as ct`. Define exactly one `@ct.kernel` function. Its
parameters are the input arrays `arg0`, `arg1`, and so on, then the outputs
`out0`, `out1`, and so on, then one parameter per entry of the proposal's
`constants` array, in that order and with the same names. Array parameters have
no annotations; each constant is annotated `ct.Constant[int]`,
`ct.Constant[float]`, or `ct.Constant[bool]` to match its JSON value, so `256`
needs `ct.Constant[int]` and `256.0` needs `ct.Constant[float]`.

The built-in adapter presents tensors as flattened rank-one views. Use
`ct.bid`, `ct.load`, `ct.store`, and tile operations. A tiled load has the form
`ct.load(array, index=(block,), shape=(BLOCK,))`; interpret the index as a tile
index under the installed cuTile API. Derive broadcast and multidimensional
indexing from the original shapes in the task.

Only the backend imports and one kernel definition belong at module scope.
Kernels may call `ct` functions, tile and array methods such as `t.astype(...)`
or `a.slice(...)`, and the builtins `abs`, `bool`, `float`, `int`, `len`,
`max`, `min`, `range`, and `tuple`. Output allocation and `cuda.tile.launch`
wrappers are executor responsibilities. Supply a static grid. The proposal has
no `num_warps` or `num_stages` fields; those are Triton options. cuTile Python
and CuTe DSL are separate backends; do not substitute one API for the other.

Preserve reference dtype transitions and numerical behavior. Account for
partial tiles using the bounds behavior supported by the selected operations.
Choose tile shapes and work distribution from correctness and timing feedback.
Only use APIs and options supported by the installed toolchain in the task.

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
