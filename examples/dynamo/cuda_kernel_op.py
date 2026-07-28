"""
.. _cuda_kernel_op:

Custom Kernels via ``torch_tensorrt.kernels.cuda_kernel_op``
==================================================================================

``cuda_kernel_op`` is the single entry point for exposing a hand-written
CUDA C++ kernel to both PyTorch eager and the Torch-TensorRT compile path.

You describe the kernel with a :class:`KernelSpec` dataclass (inputs, outputs,
extras, launch geometry) and the framework derives the meta function, the
eager CUDA launch, the TensorRT AOT implementation, and the PyTorch op schema.
When your kernel doesn't fit the declarative DSL — e.g. the output shape is
not a simple ``SameAs`` / ``ReduceDims`` of an input — pass ``meta_fn=`` /
``eager_fn=`` / ``aot_fn=`` keyword arguments and the corresponding
``KernelSpec`` fields become optional.

Calling-convention contract (every kernel must follow this argument order)::

    (input_ptrs..., scalar_inputs..., extras..., output_ptrs...)

This example walks through:

1. **Elementwise — single tensor input** (``sigmoid``)
2. **Elementwise — multiple tensor inputs** (``add``)
3. **Elementwise — with a runtime scalar input** (``scale`` by ``alpha``)
4. **Elementwise — N-D launch geometry** (2-D ``relu``)
5. **Reduction** (``row_sum`` along the last axis)
6. **Override path — shape-changing kernel** (``repeat2`` doubles each element)
7. **Multi-output kernel** (``sin_cos`` returns both ``sin(x)`` and ``cos(x)``)
"""

import cuda.core  # noqa: F401
import tensorrt.plugin as trtp
import torch
from cuda.core import Device as _Device
from cuda.core import LaunchConfig as _LaunchConfig
from cuda.core import Program as _Program
from cuda.core import ProgramOptions as _ProgramOptions
from cuda.core import launch as _cuda_launch

import torch_tensorrt
import torch_tensorrt.kernels as ttk

# ---------------------------------------------------------------------------
# Case 1: Elementwise(flat) — single tensor input
# ---------------------------------------------------------------------------
# The simplest case: one thread per output element, 1-D launch over the
# flattened output. Output shape == input shape (``SameAs(0)``).
#
# Schema cheatsheet:
#   - SameAs("x")       output has the same shape and dtype as input ``x``
#                       (an integer index into the tensor-only input list is
#                       also accepted, but names survive input reordering)
#   - Numel("x")        pass ``x.numel()`` to the kernel as an int extra
#   - Elementwise(flat) 1-D launch over the flattened output; any input rank
#                       works (the kernel just sees a contiguous buffer).

CU_SIGMOID = """
extern "C" __global__ void my_sigmoid(
        const float* __restrict__ x, int n, float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + __expf(-x[i]));
}
"""

ttk.cuda_kernel_op(
    "kern_ex::sigmoid",
    ttk.KernelSpec(
        kernel_source=CU_SIGMOID,
        kernel_name="my_sigmoid",
        inputs=[ttk.InputDecl("x")],
        outputs=[ttk.OutputDecl("y", shape=ttk.SameAs("x"))],
        extras=[ttk.Numel("x")],
        geometry=ttk.Elementwise(block=(256,), layout="flat"),
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Case 2: Elementwise(flat) — multiple tensor inputs
# ---------------------------------------------------------------------------
# Same flat geometry, but the kernel takes two input pointers. Multiple
# inputs work fine as long as they line up with the output (here all the
# same shape). The extras still come from a single named input — ``Numel("a")``
# resolves to ``a.numel()`` and is passed as an int after the input pointers.

CU_ADD = """
extern "C" __global__ void my_add(
        const float* a, const float* b, int n, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""

ttk.cuda_kernel_op(
    "kern_ex::add",
    ttk.KernelSpec(
        kernel_source=CU_ADD,
        kernel_name="my_add",
        inputs=[ttk.InputDecl("a"), ttk.InputDecl("b")],
        outputs=[ttk.OutputDecl("c", shape=ttk.SameAs("a"))],
        extras=[ttk.Numel("a")],
        geometry=ttk.Elementwise(block=(256,), layout="flat"),
    ),
)


# ---------------------------------------------------------------------------
# Case 3: ``ScalarInput`` — pass a runtime float / int / bool by value
# ---------------------------------------------------------------------------
# Use ``ScalarInput("alpha", float)`` when your kernel needs a scalar that can
# vary per call (e.g. a leaky-relu slope, dropout probability, scale factor).
# The scalar lands in the kernel signature *between* the tensor pointers and
# the extras, in source order.
#
# Scalars are represented as TensorRT plugin attributes during compilation and
# forwarded by value to the CUDA kernel.

CU_SCALE = """
extern "C" __global__ void my_scale(
        const float* x, float alpha, int n, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i];
}
"""

ttk.cuda_kernel_op(
    "kern_ex::scale",
    ttk.KernelSpec(
        kernel_source=CU_SCALE,
        kernel_name="my_scale",
        inputs=[ttk.InputDecl("x"), ttk.ScalarInput("alpha", float)],
        # Naming the source input is especially handy here: ``SameAs("x")``
        # stays correct even though a non-tensor ``ScalarInput`` sits between
        # the tensors in declaration order.
        outputs=[ttk.OutputDecl("y", shape=ttk.SameAs("x"))],
        extras=[ttk.Numel("x")],
        geometry=ttk.Elementwise(block=(256,), layout="flat"),
    ),
)


# ---------------------------------------------------------------------------
# Case 4: Elementwise(layout="nd") — N-D launch geometry
# ---------------------------------------------------------------------------
# Use ``layout="nd"`` when the kernel indexes the output via 2-D / 3-D thread
# coordinates rather than a flat index — e.g. for innermost-axis coalescing
# or per-row block parallelism. Mapping convention:
#
#   block[0]   -> innermost (last) output axis  -> grid_x / block_x
#   block[1]   -> next-innermost                -> grid_y / block_y
#   leading axes beyond ``len(block)`` fold into grid_z
#
# Here we declare ``DimSize("x", 0)`` and ``DimSize("x", 1)`` so the kernel
# receives ``H`` and ``W`` directly instead of having to back them out of
# ``numel``.

CU_RELU_2D = """
extern "C" __global__ void my_relu_2d(
        const float* x, int H, int W, float* y) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // innermost
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= H || j >= W) return;
    float v = x[i * W + j];
    y[i * W + j] = v > 0.f ? v : 0.f;
}
"""

ttk.cuda_kernel_op(
    "kern_ex::relu_2d",
    ttk.KernelSpec(
        kernel_source=CU_RELU_2D,
        kernel_name="my_relu_2d",
        inputs=[ttk.InputDecl("x")],
        outputs=[ttk.OutputDecl("y", shape=ttk.SameAs("x"))],
        extras=[ttk.DimSize("x", 0), ttk.DimSize("x", 1)],  # H, W
        geometry=ttk.Elementwise(block=(16, 16), layout="nd"),
    ),
)


# ---------------------------------------------------------------------------
# Case 5: Reduction — one block per output element
# ---------------------------------------------------------------------------
# Use ``Reduction(reduce_dims=...)`` when the output has fewer dims than the
# input because some axes are collapsed. Output shape is described with
# ``ReduceDims(input_idx, dims, keepdim=...)``. The grid is sized to the
# number of non-reduced rows; threads in each block cooperate across the
# reduction axes.
#
# The kernel below assumes ``block_size == 256`` (matching the Reduction
# spec) because it uses a fixed-size shared-memory buffer.

CU_ROW_SUM = """
extern "C" __global__ void my_row_sum(
        const float* x, int D, float* y) {
    int row = blockIdx.x;
    const float* xr = x + row * D;
    float s = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) s += xr[j];
    __shared__ float sbuf[256];
    sbuf[threadIdx.x] = s;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (threadIdx.x < step) sbuf[threadIdx.x] += sbuf[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sbuf[0];
}
"""

ttk.cuda_kernel_op(
    "kern_ex::row_sum",
    ttk.KernelSpec(
        kernel_source=CU_ROW_SUM,
        kernel_name="my_row_sum",
        inputs=[ttk.InputDecl("x")],
        outputs=[ttk.OutputDecl("y", shape=ttk.ReduceDims("x", (-1,)))],
        extras=[ttk.DimSize("x", -1)],
        geometry=ttk.Reduction(reduce_dims=(-1,), block_size=256),
    ),
)


# ---------------------------------------------------------------------------
# Case 6: Override path — shape-changing kernel
# ---------------------------------------------------------------------------
# ``repeat2`` duplicates each element: ``y[2*i] = y[2*i + 1] = x[i]``. The
# output shape ``(2 * n,)`` is not expressible as ``SameAs`` or
# ``ReduceDims``, so the declarative DSL doesn't cover it. Instead, pass
# ``meta_fn`` / ``eager_fn`` / ``aot_fn`` directly — the corresponding
# ``KernelSpec`` fields (``outputs``, ``geometry``) drop out.

CU_REPEAT2 = """
extern "C" __global__ void repeat2_kernel(
        const float* __restrict__ x, const int n, float* __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        y[2 * i] = v;
        y[2 * i + 1] = v;
    }
}
"""

_repeat2_device = _Device()
_repeat2_device.set_current()
_repeat2_opts = _ProgramOptions(
    std="c++17",
    arch=f"sm_{_repeat2_device.arch}",
    include_path=["/usr/local/cuda/include"],
)
_repeat2_kernel = (
    _Program(CU_REPEAT2, code_type="c++", options=_repeat2_opts)
    .compile("ptx", name_expressions=("repeat2_kernel",))
    .get_kernel("repeat2_kernel")
)


class _PTStream:
    def __cuda_stream__(self):
        return (0, torch.cuda.current_stream().cuda_stream)


def _repeat2_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.numel() * 2,), device=x.device, dtype=x.dtype)


def _repeat2_eager(x: torch.Tensor) -> torch.Tensor:
    flat = x.contiguous().view(-1)
    n = int(flat.numel())
    y = torch.empty((n * 2,), device=x.device, dtype=x.dtype)
    block = 256
    grid = max(1, (n + block - 1) // block)
    _cuda_launch(
        _repeat2_device.create_stream(_PTStream()),
        _LaunchConfig(grid=(grid,), block=(block,)),
        _repeat2_kernel,
        flat.data_ptr(),
        n,
        y.data_ptr(),
    )
    return y


def _repeat2_aot(inputs, outputs, tactic):
    n = inputs[0].shape_expr.numel()
    params = trtp.KernelLaunchParams()
    params.grid_x = trtp.cdiv(n, 256)
    params.block_x = 256
    params.shared_mem = 0
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(n)
    return params, extra


ttk.cuda_kernel_op(
    "kern_ex::repeat2",
    ttk.KernelSpec(
        kernel_source=CU_REPEAT2,
        kernel_name="repeat2_kernel",
        inputs=[ttk.InputDecl("x")],
        # outputs / geometry omitted — supplied via overrides below.
    ),
    meta_fn=_repeat2_meta,
    eager_fn=_repeat2_eager,
    aot_fn=_repeat2_aot,
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Case 7: Multi-output kernel — one launch produces two outputs
# ---------------------------------------------------------------------------
# A single kernel can emit any number of outputs by listing them in
# ``outputs=[OutputDecl(...), OutputDecl(...), ...]``. Output pointers are
# appended to the kernel argument list in declaration order, matching the
# calling-convention contract.
#
# The op's torch schema is auto-derived as ``-> (Tensor, Tensor)`` so callers
# unpack with ``s, c = torch.ops.kern_ex.sin_cos(x)``. Each output gets its
# own ``OutputDecl.shape`` — here both are ``SameAs("x")`` because sin/cos
# are elementwise — but you can mix ``SameAs`` and ``ReduceDims`` freely for
# multi-output kernels with different output shapes (e.g. mean + variance
# from a single reduction pass).

CU_SIN_COS = """
extern "C" __global__ void my_sin_cos(
        const float* __restrict__ x, int n,
        float* __restrict__ s, float* __restrict__ c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        s[i] = __sinf(x[i]);
        c[i] = __cosf(x[i]);
    }
}
"""

ttk.cuda_kernel_op(
    "kern_ex::sin_cos",
    ttk.KernelSpec(
        kernel_source=CU_SIN_COS,
        kernel_name="my_sin_cos",
        inputs=[ttk.InputDecl("x")],
        outputs=[
            ttk.OutputDecl("s", shape=ttk.SameAs("x")),
            ttk.OutputDecl("c", shape=ttk.SameAs("x")),
        ],
        extras=[ttk.Numel("x")],
        geometry=ttk.Elementwise(block=(256,), layout="flat"),
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Per-case Modules
# ---------------------------------------------------------------------------


class SigmoidModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.kern_ex.sigmoid(x)


class AddModel(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.ops.kern_ex.add(a, b)


class ScaleModel(torch.nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.kern_ex.scale(x, self.alpha)


class Relu2DModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.kern_ex.relu_2d(x)


class RowSumModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.kern_ex.row_sum(x)


class Repeat2Model(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.kern_ex.repeat2(x)


class SinCosModel(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Multi-output ops return a tuple; the caller unpacks it.
        return torch.ops.kern_ex.sin_cos(x)


# ---------------------------------------------------------------------------
# Drive each case: eager + (where applicable) Torch-TensorRT compile
# ---------------------------------------------------------------------------


def _allclose(out, ref) -> bool:
    """Compare a (possibly multi-output) result tuple against a reference."""
    if isinstance(out, (tuple, list)):
        return all(torch.allclose(o, r, atol=1e-2, rtol=1e-2) for o, r in zip(out, ref))
    return torch.allclose(out, ref, atol=1e-2, rtol=1e-2)


def _run(name, model, inputs, ref, *, compile_trt=True):
    eager_out = model(*inputs)
    print(f"[{name}] eager matches reference:", _allclose(eager_out, ref))
    if not compile_trt:
        print(f"[{name}] TRT compile skipped (see case docstring)")
        return
    trt_model = torch_tensorrt.compile(
        model,
        inputs=list(inputs),
        enabled_precisions={torch.float32},
        min_block_size=1,
    )
    with torch.no_grad():
        for _ in range(5):
            out = trt_model(*inputs)
            assert _allclose(out, ref), f"{name}: mismatch"
    print(f"[{name}] TRT compile + run successful")


if __name__ == "__main__":
    # Case 1: sigmoid (Elementwise flat, single input)
    x = torch.randn(4, 128, device="cuda", dtype=torch.float32)
    _run("sigmoid", SigmoidModel().cuda().eval(), (x,), torch.sigmoid(x))

    # Case 2: add (Elementwise flat, two inputs)
    a = torch.randn(256, device="cuda", dtype=torch.float32)
    b = torch.randn(256, device="cuda", dtype=torch.float32)
    _run("add", AddModel().cuda().eval(), (a, b), a + b)

    # Case 3: scale (ScalarInput)
    alpha = 2.5
    _run(
        "scale",
        ScaleModel(alpha).cuda().eval(),
        (x,),
        alpha * x,
    )

    # Case 4: relu_2d (Elementwise nd, 2-D block)
    x2d = torch.randn(33, 47, device="cuda", dtype=torch.float32)
    _run("relu_2d", Relu2DModel().cuda().eval(), (x2d,), torch.relu(x2d))

    # Case 5: row_sum (Reduction along last axis)
    xr = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    _run("row_sum", RowSumModel().cuda().eval(), (xr,), xr.sum(-1))

    # Case 6: repeat2 (override path — shape-changing kernel)
    xrep = torch.randn(1024, device="cuda", dtype=torch.float32)
    _run(
        "repeat2",
        Repeat2Model().cuda().eval(),
        (xrep,),
        torch.repeat_interleave(xrep, 2, dim=0),
    )

    # Case 7: sin_cos (multi-output — single kernel emits two tensors)
    xsc = torch.randn(7, 31, device="cuda", dtype=torch.float32)
    _run(
        "sin_cos",
        SinCosModel().cuda().eval(),
        (xsc,),
        (torch.sin(xsc), torch.cos(xsc)),
    )
