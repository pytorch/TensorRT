# NVIDIA CuTe DSL backend

Return the fields `source`, `kernel_name`, `grid`, `block`, `shared_mem_bytes`,
and `constants`. The executor imports `source`, compiles the kernel with the
installed CuTe DSL (`nvidia-cutlass-dsl`, the `cutlass` package) to PTX for the
target GPU, embeds that PTX in a TensorRT plugin, and launches the same PTX
when it screens the candidate. CuTe DSL is not cuTile Python (`cuda.tile`) and
not CUTLASS C++; do not substitute their APIs.

## Source and tensor ABI

Write `import cutlass` and `import cutlass.cute as cute`, then define exactly one
`@cute.kernel` function. Nothing else belongs at module scope: no other imports,
`from` imports, helper functions, host `@cute.jit` wrappers, `cute.compile`
calls, assignments, or output allocation. The executor builds the launch.

The kernel's parameters are the input pointers `arg0`, `arg1`, and so on, then
the output pointers `out0`, `out1`, and so on, then one parameter per entry of
the proposal's `constants` array, in that order and with the same names. Pointer
parameters have no annotation. Each constant is annotated `cutlass.Constexpr`
and is a compile-time Python value: use constants for shapes, strides, tile
sizes, and flags. No runtime scalar arguments are supplied.

Each pointer is a global-memory `cute.Pointer` whose element type is its
boundary dtype: `cutlass.Float32`, `cutlass.Float16`, or `cutlass.BFloat16`.
Every boundary tensor is contiguous with the captured shape, so its row-major
strides follow from the task's shapes. Index a pointer directly
(`out0[i] = arg0[i]`) or give it a static layout, for example
`cute.make_tensor(arg0, cute.make_layout((M, N), stride=(N, 1)))` with constant
`M` and `N`. Static layouts let CuTe fold index arithmetic at compile time.

The executor tells CuTe how far each pointer is aligned. By default that is the
pointer's element size, which every tensor satisfies. When the task's
`codegen.aligned_io` is true, every input and output pointer is 16-byte aligned
and the executor compiles with that assumption, so CuTe can use 128-bit loads,
stores, and `cp.async` copies where the layouts and index arithmetic also prove
16-byte steps, for example with `cute.assume(offset, divby=8)` for 16-bit
elements. Do not assume more alignment than the task states: a deployed
engine can pass a pointer at a smaller offset, and unless `codegen.aligned_io`
is true the evaluator's direct launches use pointers one element past a
16-byte boundary, where a kernel that forces wider accesses fails with a
misaligned address.

## Supported operations

Inside the kernel you may call:

- `cute` functions such as `make_tensor`, `make_layout`, `local_tile`,
  `zipped_divide`, `make_rmem_tensor`, `autovec_copy`, `copy`, `gemm`,
  `assume`, `make_tiled_mma`, `make_tiled_copy_tv`, `make_copy_atom`,
  `recast_ptr`, `where`, and `full`;
- `cute.arch` functions such as `thread_idx`, `block_idx`, `lane_idx`,
  `warp_idx`, `barrier`, `sync_threads`, `shuffle_sync_bfly`,
  `warp_reduction_sum`, `warp_reduction_max`, `cp_async_commit_group`, and
  `cp_async_wait_group`;
- `cute.math` functions;
- `cute.nvgpu.warp` atoms such as `MmaF16BF16Op`, `MmaTF32Op`, `LdMatrix8x8x16bOp`,
  and `StMatrix8x8x16bOp`;
- `cute.nvgpu.cpasync.CopyG2SOp` and `cute.nvgpu.CopyUniversalOp`;
- `cutlass` numeric types such as `cutlass.Float32(x)`, `cutlass.range`,
  `cutlass.range_constexpr`, `cutlass.const_expr`, and
  `cutlass.utils.SmemAllocator`;
- methods of CuTe tensors, `TensorSSA` values, layouts, and MMA and copy atoms,
  such as `load`, `store`, `fill`, `reduce`, `to`, `get_slice`, `partition_A`,
  and `partition_S`;
- the builtins `abs`, `bool`, `float`, `int`, `len`, `max`, `min`, `range`, and
  `tuple`.

The source check rejects:

- helper functions, lambdas, classes, `with`, `try`, and global state;
- private (`_`) names, `cute.printf`, and `cute.make_ptr`;
- names that reuse one of the builtins listed above, `cutlass`, `cute`, or
  `type` and `locals`, which CuTe DSL itself calls inside kernels;
- other builtins, such as `open`, even when passed rather than called. A
  variable may take such a name, such as `sum`, but is usable only after an
  assignment earlier in its block or an enclosing one, not one inside an `if`,
  loop or comprehension that has ended;
- modules or other packages reached through `cutlass` or `cute`, such as
  `cute.runtime`;
- the pointer `align` method;
- TMA, tensor-memory, cluster, `warpgroup`, and `tcgen05` APIs.

CuTe stages Python control flow on runtime values. A `for` loop over `range` or
`cutlass.range` with runtime bounds, and an `if` on a runtime value, become GPU
control flow. Code inside them cannot `return`, `break`, or `continue`: write
`if i < N:` around the body instead of an early return. Use
`cutlass.range_constexpr` and `if cutlass.const_expr(...)` for loops and branches
that constants decide at compile time.

## Launch contract

Supply a static `grid` and `block` of one to three positive dimensions each.
`block` is the thread-block shape. It holds at most 1024 threads, with a z
dimension of at most 64, and the compiled kernel's `.reqntid` must equal it.

`shared_mem_bytes` is the dynamic shared memory the launch reserves. Allocate
shared memory only through `cutlass.utils.SmemAllocator`, for example
`smem.allocate_tensor(cutlass.Float32, cute.make_layout(BLOCK), byte_alignment=16)`.
Set `shared_mem_bytes` to at least the bytes the allocations use, including
alignment padding: CuTe does not check this for a plugin launch, and an
overrun fails at run time. Use 0 when the kernel allocates no shared memory;
a kernel that allocates some and declares 0 is rejected.

Static and dynamic shared memory together must not exceed 48 KiB (49152 bytes).
A TensorRT plugin launch does not opt into the larger per-block carve-out that
the GPU offers. Thread-block clusters, distributed shared memory, TMA
descriptors, and compiler-added workspace arguments are unavailable, because the
plugin passes only the tensor pointers.

## Target notes: sm_120 (RTX Blackwell)

The task's target metadata names the GPU and its compute capability. On compute
capability 12.0:

- Tensor-core math is warp-level `mma.sync`. Use
  `cute.nvgpu.warp.MmaF16BF16Op(ab_dtype, acc_dtype, (16, 8, 16))` for float16
  or bfloat16 operands with float32 accumulation, or `MmaTF32Op` for float32
  inputs that the reference permits to round to TF32. Each warp runs the MMA
  collectively. A operands are K-major (row-major) and B operands are K-major
  (column-major). `ldmatrix` atoms load operand fragments from shared memory.
- Hopper `wgmma` (`cute.nvgpu.warpgroup`) fails to compile for this target.
  Datacenter-Blackwell `tcgen05` MMA and tensor memory (TMEM) do not exist here.
- TMA (`cp.async.bulk.tensor`) needs a descriptor built on the host, and the
  plugin cannot pass one. Stage tiles in shared memory with `cp.async`
  (`CopyG2SOp`) or with vectorized loads.
- A block may use at most 48 KiB of shared memory (the plugin limit), and an
  SM has 64K 32-bit registers and at most 1536 resident threads. Smaller
  per-block register and shared-memory footprints allow more resident blocks.
  The metadata's multiprocessor count sizes the grid needed to fill the GPU.

## Numerics

Convert loaded half and bfloat16 values with `.to(cutlass.Float32)` before
arithmetic. Restore the reference's dtype at the same points it does, and
convert with `.to(...)` before storing into a pointer of another type. CuTe
emits float32 `add` and `mul` without a rounding modifier, so the PTX assembler
may contract them into FMA. The float64 comparison described in
`correctness.md` normally absorbs this, but reproduce explicit reference
roundings exactly. Reductions should accumulate in float32 unless the reference
accumulates in a lower precision.

## Example

For a reference that adds two contiguous float32 tensors of 1000 elements, this
source matches the ABI:

```python
import cutlass
import cutlass.cute as cute

@cute.kernel
def add(arg0, arg1, out0, N: cutlass.Constexpr, BLOCK: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * BLOCK + tidx
    if i < N:
        out0[i] = arg0[i] + arg1[i]
```

With that text as `source`, the remaining proposal fields are:

```json
{"kernel_name": "add", "grid": [4], "block": [256], "shared_mem_bytes": 0, "constants": [{"name": "N", "value": 1000}, {"name": "BLOCK", "value": 256}]}
```

`block` must equal the launch's thread count, here `BLOCK`. When tuning `block`,
tune the matching constant together with it. Adapt the computation, dtypes,
layouts, and launch settings to the actual task; this example is not evidence
of a measured result.
