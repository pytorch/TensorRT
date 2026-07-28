"""Shared CUDA kernel sources, skip marks, and helpers for kernels tests."""

from __future__ import annotations

import pytest
import torch
import torch_tensorrt

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)
skip_no_qdp = pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    reason="TensorRT QDP plugin not available",
)


def _has_cuda_core() -> bool:
    """True if the cuda-core ``cuda.core`` API (NVRTC/QDP backend) is importable."""
    import importlib.util

    for mod in ("cuda.core", "cuda.core.experimental"):
        try:
            if importlib.util.find_spec(mod) is not None:
                return True
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
    return False


_HAS_CUDA_CORE = _has_cuda_core()

skip_no_cuda_core = pytest.mark.skipif(
    not _HAS_CUDA_CORE,
    reason="cuda-core (cuda.core) not installed; QDP kernel compilation unavailable. "
    "Install via the kernels group (e.g. `just install-test-ext`).",
)


def pytest_collection_modifyitems(config, items):
    """Skip the whole kernels suite when cuda-core is missing.

    The QDP kernel tests compile CUDA via the ``cuda.core`` NVRTC API; without
    cuda-core they all error on import. Skip (not fail) so a plain test run is
    green — `just install-test-ext` pulls cuda-core so they actually run.
    """
    if _HAS_CUDA_CORE:
        return
    skip = pytest.mark.skip(reason="cuda-core (cuda.core) not installed")
    for item in items:
        item.add_marker(skip)


SIGMOID_SRC = """
extern "C" __global__ void ttk_test_sigmoid(
        const float* x, int n, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + __expf(-x[i]));
}
"""

RELU_FLAT_SRC = """
extern "C" __global__ void ttk_kp_relu_flat(
        const float* x, int n, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}
"""

RELU_ND_SRC = """
extern "C" __global__ void ttk_kp_relu_nd(
        const float* x, int H, int W, float* y) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= H || j >= W) return;
    float v = x[i * W + j];
    y[i * W + j] = v > 0.f ? v : 0.f;
}
"""

ROW_SUM_SRC = """
extern "C" __global__ void ttk_kp_row_sum(
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

ADD_SRC = """
extern "C" __global__ void ttk_kp_add(
        const float* a, const float* b, int n, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""

SCALE_SRC = """
extern "C" __global__ void ttk_kp_scale(
        const float* x, float alpha, int n, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i];
}
"""

SIN_COS_SRC = """
extern "C" __global__ void ttk_kp_sin_cos(
        const float* __restrict__ x, int n,
        float* __restrict__ s, float* __restrict__ c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        s[i] = __sinf(x[i]);
        c[i] = __cosf(x[i]);
    }
}
"""


def register_once(register_fn):
    """Invoke `register_fn()`; swallow duplicate-registration errors on re-run."""
    try:
        register_fn()
    except Exception:
        pass


def make_sigmoid_aot(block_size: int = 256):
    """Build a minimal trtp aot_fn for 1-D pointwise kernels."""
    import tensorrt.plugin as trtp

    def _aot(inputs, outputs, tactic):
        n = inputs[0].shape_expr.numel()
        p = trtp.KernelLaunchParams()
        p.grid_x, p.block_x, p.shared_mem = trtp.cdiv(n, block_size), block_size, 0
        extra = trtp.SymIntExprs(1)
        extra[0] = trtp.SymInt32(n)
        return p, extra

    return _aot


def make_eager_sigmoid():
    """Compile SIGMOID_SRC once and return an eager launch fn."""
    try:
        from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
    except ImportError:
        from cuda.core.experimental import (
            Device,
            LaunchConfig,
            Program,
            ProgramOptions,
            launch,
        )

    dev = Device()
    dev.set_current()
    opts = ProgramOptions(
        std="c++17", arch=f"sm_{dev.arch}", include_path=["/usr/local/cuda/include"]
    )
    kernel = (
        Program(SIGMOID_SRC, code_type="c++", options=opts)
        .compile("ptx", name_expressions=("ttk_test_sigmoid",))
        .get_kernel("ttk_test_sigmoid")
    )

    class _Stream:
        def __cuda_stream__(self):
            return (0, torch.cuda.current_stream().cuda_stream)

    def _eager(x: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        n = int(x.numel())
        launch(
            dev.create_stream(_Stream()),
            LaunchConfig(grid=(max(1, (n + 255) // 256),), block=(256,)),
            kernel,
            x.data_ptr(),
            n,
            y.data_ptr(),
        )
        return y

    return _eager
