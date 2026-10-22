# NVIDIA cuTile Python backend

Import `cuda.tile as ct`. Define exactly one `@ct.kernel` function. Tensor
parameters have no annotations; compile-time constants use `ct.Constant[int]`,
`ct.Constant[float]`, or `ct.Constant[bool]`. Follow the task's parameter order.

The built-in adapter presents tensors as flattened rank-one views. Use
`ct.bid`, `ct.load`, `ct.store`, and tile operations. A tiled load has the form
`ct.load(array, index=(block,), shape=(BLOCK,))`; interpret the index as a tile
index under the installed cuTile API. Derive broadcast and multidimensional
indexing from the original shapes in the task.

Only the backend imports and one kernel definition belong at module scope.
Output allocation and `cuda.tile.launch` wrappers are executor responsibilities.
Supply a static grid. Omit Triton's `num_warps` and `num_stages`; they are not
fields in the cuTile proposal schema. cuTile Python and CuTe DSL are separate backends;
do not substitute one API for the other.

Preserve reference dtype transitions and numerical behavior. Account for
partial tiles using the bounds behavior supported by the selected operations.
Choose tile shapes and work distribution from correctness and timing feedback.
Only use APIs and options supported by the installed toolchain in the task.
