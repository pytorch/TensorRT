"""
torch_tensorrt.kernels  (experimental)
=======================================
Register custom CUDA C++ kernels — compiled at runtime with NVRTC via
**cuda-python** — as TensorRT Quick Deployable Plugins (QDP). Tensor-only
declarative kernels use AOT plugin launches when available; kernels with
``ScalarInput`` use TensorRT's QDP JIT path so runtime scalar attributes can
be forwarded by value.

The module exposes a single registration entry point for source kernels:

``cuda_kernel_op`` — fully declarative for the common cases, with optional
    overrides for everything else. Describe the kernel via :class:`KernelSpec`
    (inputs, outputs, extras, launch geometry) and the meta / eager / aot
    functions plus the PyTorch schema are derived for you. For shape-changing
    kernels, multi-output kernels, or anything outside the declarative DSL,
    pass ``meta_fn=`` / ``eager_fn=`` / ``aot_fn=`` / ``schema=`` keyword
    arguments and the corresponding ``KernelSpec`` fields become optional.

``ptx_op`` — register a kernel from pre-compiled PTX bytes. Skips NVRTC
    entirely; you supply ``meta_fn`` / ``eager_fn`` / ``aot_fn`` directly.
    Useful when the PTX comes from an external compiler (Triton, a cached
    NVRTC output, etc.).

Minimal example — declarative ``cuda_kernel_op``::

    import torch, torch_tensorrt
    import torch_tensorrt.kernels as ttk

    cu_code = \"\"\"
    extern "C" __global__ void my_relu(const float* x, int n, float* y) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
    }
    \"\"\"

    ttk.cuda_kernel_op(
        "myns::relu",
        ttk.KernelSpec(
            kernel_source=cu_code,
            kernel_name="my_relu",
            inputs=[ttk.InputDecl("x")],
            outputs=[ttk.OutputDecl("y", shape=ttk.SameAs(0))],
            extras=[ttk.Numel("x")],
            geometry=ttk.Elementwise(block=(256,), layout="flat"),
        ),
        supports_dynamic_shapes=True,
    )

For a shape-changing kernel, leave ``outputs`` / ``geometry`` off the
``KernelSpec`` and pass hand-written ``meta_fn`` / ``eager_fn`` / ``aot_fn``
to ``cuda_kernel_op`` directly — see ``examples/dynamo/cuda_kernel_op.py``.
"""

from torch_tensorrt.kernels._dsl import (
    Custom,
    DimSize,
    Elementwise,
    InputDecl,
    KernelSpec,
    Numel,
    OutputDecl,
    ReduceDims,
    Reduction,
    SameAs,
    ScalarInput,
)
from torch_tensorrt.kernels._ops import cuda_kernel_op, ptx_op

__all__ = [
    "Custom",
    "DimSize",
    "Elementwise",
    "InputDecl",
    "KernelSpec",
    "Numel",
    "OutputDecl",
    "ReduceDims",
    "Reduction",
    "SameAs",
    "ScalarInput",
    "cuda_kernel_op",
    "ptx_op",
]
