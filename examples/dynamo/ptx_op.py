"""
.. _ptx_op:

Pre-compiled PTX kernels via ``torch_tensorrt.kernels.ptx_op``
==================================================================================

``cuda_kernel_op`` compiles your CUDA C++ source internally with NVRTC. When
you already have PTX in hand — emitted by Triton's JIT, cached from a prior
NVRTC run on another machine, or hand-written — use ``ptx_op`` instead. It
skips NVRTC entirely and registers the supplied PTX directly as a TensorRT
Quick Deployable Plugin.

You supply:

* **the PTX bytes** (whatever produced them)
* **the kernel entry symbol** inside that PTX
* **``meta_fn`` / ``eager_fn`` / ``aot_fn``** by hand — there's no
  :class:`KernelSpec` DSL on this path
* optionally, an explicit PyTorch ``schema`` string (inferred from
  ``meta_fn`` type hints if omitted)

This example walks through ``gelu`` (approximate-tanh GELU activation):

1. NVRTC-compile the source to PTX *once*, manually, to simulate having
   PTX in hand from any external source.
2. Write ``meta_fn`` / ``eager_fn`` / ``aot_fn``.
3. Register via :func:`ptx_op`.
4. Run eager and Torch-TensorRT compile against a PyTorch reference.
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
# Step 1: obtain PTX
# ---------------------------------------------------------------------------
# In real use, ``GELU_PTX`` could be loaded from a file, fetched from a cache,
# or emitted by a Triton kernel's ``.asm["ptx"]``. Here we NVRTC-compile a
# small CUDA source on the fly so the example is self-contained — the point
# of ptx_op is that this step is *separate* from registration.

CU_GELU = r"""
extern "C" __global__ void my_gelu(
        const float* __restrict__ x, int n, float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float kSqrt2OverPi = 0.7978845608f;   // sqrt(2/pi)
        const float kCubicCoef   = 0.044715f;
        float xi = x[i];
        float inner = kSqrt2OverPi * (xi + kCubicCoef * xi * xi * xi);
        y[i] = 0.5f * xi * (1.f + tanhf(inner));
    }
}
"""

_device = _Device()
_device.set_current()
_opts = _ProgramOptions(
    std="c++17",
    arch=f"sm_{_device.arch}",
    include_path=["/usr/local/cuda/include"],
)
_module = _Program(CU_GELU, code_type="c++", options=_opts).compile(
    "ptx", name_expressions=("my_gelu",)
)
GELU_PTX: bytes = _module.code
_gelu_kernel = _module.get_kernel("my_gelu")


# ---------------------------------------------------------------------------
# Step 2: meta / eager / aot functions
# ---------------------------------------------------------------------------
# ``ptx_op`` does no derivation — you write each of these directly.
#
#   * meta_fn   : shape/dtype inference for FakeTensors (torch.compile path).
#                  The torch op schema is inferred from its type hints if you
#                  don't pass ``schema=`` explicitly.
#   * eager_fn  : the CUDA launch that runs under PyTorch eager.
#   * aot_fn    : returns (KernelLaunchParams, SymExprs) for TRT engine build.


def _gelu_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class _PTStream:
    """Adapter so cuda.core.Stream uses PyTorch's current stream."""

    def __cuda_stream__(self):
        return (0, torch.cuda.current_stream().cuda_stream)


def _gelu_eager(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n = int(x.numel())
    block = 256
    grid = max(1, (n + block - 1) // block)
    _cuda_launch(
        _device.create_stream(_PTStream()),
        _LaunchConfig(grid=(grid,), block=(block,)),
        _gelu_kernel,
        x.data_ptr(),
        n,
        y.data_ptr(),
    )
    return y


def _gelu_aot(inputs, outputs, tactic):
    # ``inputs`` are TensorDescs with symbolic shape_expr (TRT-side algebra).
    n = inputs[0].shape_expr.numel()
    params = trtp.KernelLaunchParams()
    params.grid_x = trtp.cdiv(n, 256)
    params.block_x = 256
    params.shared_mem = 0
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(n)
    return params, extra


# ---------------------------------------------------------------------------
# Step 3: register via ptx_op
# ---------------------------------------------------------------------------
# After this call, ``torch.ops.ptx_ex.gelu`` exists and works in eager,
# torch.compile, and torch_tensorrt.compile — same as ``cuda_kernel_op``.

ttk.ptx_op(
    op_name="ptx_ex::gelu",
    ptx=GELU_PTX,
    kernel_name="my_gelu",
    meta_fn=_gelu_meta,
    eager_fn=_gelu_eager,
    aot_fn=_gelu_aot,
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Step 4: drive eager + Torch-TensorRT compile
# ---------------------------------------------------------------------------


class GeluModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.ptx_ex.gelu(x)


if __name__ == "__main__":
    x = torch.randn(4, 128, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.gelu(x, approximate="tanh")
    model = GeluModel().cuda().eval()

    eager_out = model(x)
    print(
        f"[gelu] eager matches reference: {torch.allclose(eager_out, ref, atol=1e-3)}"
    )

    trt_model = torch_tensorrt.compile(
        model,
        inputs=[x],
        enabled_precisions={torch.float32},
        min_block_size=1,
    )
    with torch.no_grad():
        for _ in range(5):
            out = trt_model(x)
            assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), "gelu: mismatch"
    print("[gelu] TRT compile + run successful")
