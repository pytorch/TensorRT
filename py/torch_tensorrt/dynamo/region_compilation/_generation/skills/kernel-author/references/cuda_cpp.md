# CUDA C++ through CUDA Python

Return the fields `source`, `kernel_name`, `grid`, `block`, and
`shared_mem_bytes`. The executor compiles CUDA C++ with CUDA Python's `cuda.core`
and NVRTC, then registers its compiled GPU code. This is not Numba CUDA Python.

Define exactly one `extern "C" __global__ void` entry point. Its parameters are
tensor pointers named `arg0`, `arg1`, and so on, followed by `out0`, `out1`, and
so on. Use `float`, `half`/`__half`, or `__nv_bfloat16` consistently with the
boundary dtype. Input pointers may be `const`; outputs must be mutable.
`__restrict__` pointer qualifiers are supported. Embed static dimensions and
other constants in the source; no runtime scalar arguments are supplied.

Optional includes are limited to `cuda_fp16.h`, `cuda_bf16.h`, `math.h`, `cmath`,
and `stdint.h`. Do not define helper functions, macros, host launch wrappers,
or inline assembly. Output allocation and launches belong to the executor.

Supply one to three positive dimensions for the static grid and block. The
block product must not exceed 1024 threads and its z dimension must not exceed
64. Dynamic shared memory must be an integer from 0 through 49152 bytes. The
executor separately checks actual device and compiled-kernel limits.

Guard out-of-range threads, derive broadcasting indices from the supplied
shapes, and write every output. Preserve each reference operation's rounding.
For half values, use `__half2float` and `__float2half_rn` as needed between
operations; for bfloat16, use `__bfloat162float` and `__float2bfloat16_rn`.
Float32 `__fadd_rn` and `__fmul_rn` can preserve rounding when fusion would change
the reference result. Tune block geometry and data reuse using evaluator
measurements, while keeping the original computation and output contract.

For a reference that adds two contiguous float32 tensors of exactly 1024
elements, this source matches the pointer ABI:

```cpp
extern "C" __global__ void add(const float* arg0, const float* arg1, float* out0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1024) out0[i] = __fadd_rn(arg0[i], arg1[i]);
}
```

Its proposal uses `kernel_name="add"`, `grid=[4]`, `block=[256]`, and
`shared_mem_bytes=0`. Adapt the computation, dtypes, indexing, and launch shape
to the actual task; this example is not evidence of a measured result.
