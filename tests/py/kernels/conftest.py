"""Shared CUDA kernel sources, skip marks, and helpers for kernels tests."""

from __future__ import annotations

from typing import Callable

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


def _has_module(*names: str) -> bool:
    """True if any of ``names`` is importable."""
    import importlib.util

    for name in names:
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
    return False


skip_no_triton = pytest.mark.skipif(
    not _has_module("triton"), reason="triton not installed"
)

# The cuda-core ``cuda.core`` API is the NVRTC/QDP backend.
_HAS_CUDA_CORE = _has_module("cuda.core", "cuda.core.experimental")


def _decode_cuda_driver_version(version: int) -> tuple[int, int]:
    """Decode CUDA's integer driver API version into ``(major, minor)``."""
    if version <= 0:
        raise ValueError(f"invalid CUDA driver version: {version}")
    return version // 1000, (version % 1000) // 10


def _cuda_driver_nvrtc_versions() -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the maximum CUDA version supported by the driver and NVRTC's version."""
    try:
        from cuda.bindings import driver, nvrtc
    except ImportError:
        # cuda-python < 12.8 exposed the same APIs from these modules.
        from cuda import cuda as driver
        from cuda import nvrtc

    driver_status, encoded_driver_version = driver.cuDriverGetVersion()
    if int(driver_status) != 0:
        raise RuntimeError(
            f"cuDriverGetVersion failed with CUDA error {int(driver_status)}"
        )

    nvrtc_status, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()
    if int(nvrtc_status) != 0:
        raise RuntimeError(f"nvrtcVersion failed with CUDA error {int(nvrtc_status)}")

    return _decode_cuda_driver_version(encoded_driver_version), (
        nvrtc_major,
        nvrtc_minor,
    )


def _ptx_compatibility_skip_reason(
    driver_version: tuple[int, int], nvrtc_version: tuple[int, int]
) -> str | None:
    """Explain why NVRTC PTX cannot be loaded, or return ``None`` if compatible."""
    if driver_version >= nvrtc_version:
        return None

    driver = ".".join(map(str, driver_version))
    backend = ".".join(map(str, nvrtc_version))
    return (
        "CUDA kernel tests require a driver capable of loading the PTX emitted "
        f"by NVRTC, but the active driver supports CUDA {driver} and NVRTC is "
        f"CUDA {backend}. Upgrade the CI runner driver to >= {backend}, or use "
        f"a CUDA toolkit/runtime <= {driver}."
    )


def _nvrtc_skip_reason() -> str | None:
    """Return a backend-specific reason for skipping NVRTC-dependent tests."""
    if not _HAS_CUDA_CORE:
        return "cuda-core (cuda.core) not installed"
    if not torch.cuda.is_available():
        # The independent skip_no_cuda marker provides the clearest reason.
        return None
    try:
        driver_version, nvrtc_version = _cuda_driver_nvrtc_versions()
        return _ptx_compatibility_skip_reason(driver_version, nvrtc_version)
    except Exception as exc:
        return f"could not verify CUDA driver/NVRTC compatibility: {exc}"


# Apply this only to tests that actually compile CUDA C++ with NVRTC. Triton
# compilation has its own driver-PTX compatibility handling and must continue
# to run when the NVRTC toolkit happens to be newer than the installed driver.
_NVRTC_SKIP_REASON = _nvrtc_skip_reason()
skip_no_nvrtc = pytest.mark.skipif(
    _NVRTC_SKIP_REASON is not None,
    reason=_NVRTC_SKIP_REASON or "NVRTC unavailable",
)


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


_REGISTERED_OPS: set[str] = set()


def register_once(op_name: str, register_fn: Callable[[], None]) -> None:
    """Register an op once per test process without hiding real failures."""
    if op_name in _REGISTERED_OPS:
        return
    register_fn()
    _REGISTERED_OPS.add(op_name)


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
