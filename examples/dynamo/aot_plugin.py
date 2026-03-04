"""
.. _aot_plugin:
Automatically Generate a TensorRT AOT Plugin
===================================================================

This example builds on :ref:`auto_generate_plugins` by showing the *AOT* (Ahead-of-Time)
plugin path. Instead of calling back into Python at TRT engine runtime (JIT), the
Triton kernel is compiled to PTX at *plugin registration time* and the binary is
embedded in the TRT engine. This eliminates all Python overhead during inference.

**JIT vs AOT — the key difference:**

* **JIT plugin** (``generate_plugin`` default): TRT holds a Python callback. At runtime
  it converts TRT tensor handles to ``torch.Tensor``, calls your op, copies results
  back. Simple, but adds Python overhead per inference call.

* **AOT plugin** (this example): At registration time ``@trtp.aot_impl`` compiles the
  Triton kernel to PTX/CUBIN and returns the binary plus kernel launch parameters.
  TRT embeds that binary in the engine. At runtime TRT launches the kernel directly —
  no Python, no tensor conversion, no copying. Also required for serialized engines
  that will run without a Python environment (e.g. C++ deployment).

**When to use AOT:**

* Performance-critical inference paths.
* Engines that must be serialized and loaded in C++.
* Any case where you need ``use_aot_if_available=True`` and want the guarantee that
  the AOT path is actually taken.
"""

import argparse
from typing import Tuple, Union

import tensorrt as trt
import tensorrt.plugin as trtp
import torch
import torch_tensorrt
import triton
import triton.language as tl

trt_logger = trt.Logger(trt.Logger.VERBOSE)


# %%
# Step 1: Define the Triton Kernel
# -----------------------------------------
#
# Same as the JIT example — the kernel is pure Triton. The difference is how it
# gets compiled: in the JIT path ``add_one_kernel[grid](...)`` is called at runtime;
# in the AOT path it is compiled to PTX inside ``@trtp.aot_impl`` below.


@triton.jit
def add_one_kernel(x_ptr, n_elements, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + 1
    tl.store(y_ptr + offsets, output, mask=mask)


# %%
# Step 2: Register the PyTorch op
# -----------------------------------------
#
# Identical to the JIT example. The meta-kernel (``register_fake``) is still needed:
# TRT uses the shape-descriptor from ``@trtp.register`` (below) for graph-build-time
# shape inference, but Dynamo's tracing and Torch-TensorRT's partitioner still need
# the fake kernel.


@torch.library.custom_op("my::add_one", mutates_args=())  # type: ignore[misc]
def add_one(X: torch.Tensor) -> torch.Tensor:
    assert X.is_cuda
    Y = torch.empty_like(X)
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(X.numel(), meta["BLOCK_SIZE"]),)
    add_one_kernel[grid](X, X.numel(), Y, BLOCK_SIZE=BLOCK_SIZE)
    return Y


@torch.library.register_fake("my::add_one")
def _(X: torch.Tensor) -> torch.Tensor:
    return X


# %%
# Step 3: Register the QDP Shape Descriptor
# -----------------------------------------
#
# ``@trtp.register`` manually registers the plugin *shape descriptor* in TensorRT's
# ``QDP_REGISTRY`` under the key ``"my::add_one"``. This is different from
# ``generate_plugin``, which auto-generates the descriptor from the fake kernel.
#
# The function receives ``trtp.TensorDesc`` objects describing input shapes/dtypes,
# and must return a tuple of ``trtp.TensorDesc`` for outputs.
# ``X.like()`` means "same shape and dtype as X" — shorthand for elementwise ops.
#
# Registering manually here (instead of calling ``generate_plugin``) is required for
# AOT plugins because we need to associate our own ``@trtp.aot_impl`` with the plugin
# entry. ``generate_plugin`` would create its own JIT impl and close the entry.


@trtp.register("my::add_one")
def add_plugin_desc(X: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
    # Output has the same shape and dtype as the input.
    return X.like()


# %%
# Step 4: Register the AOT Implementation
# -----------------------------------------
#
# ``@trtp.aot_impl`` is called **once at registration time** (not at inference time).
# It must compile the kernel to a binary and return everything TRT needs to launch it:
#
# * ``compiled_kernel.metadata.name`` — the kernel function name in the PTX/CUBIN.
# * ``compiled_kernel.asm["ptx"]`` — the PTX source string (or CUBIN bytes).
#   TRT embeds this binary in the serialized engine.
# * ``launch_params`` — grid/block dims and shared memory. These can be symbolic
#   (using ``trtp.SymExprs``) so the same engine works across batch sizes.
# * ``extra_args`` — additional scalar arguments passed at launch. Here ``N`` (number
#   of elements) is a ``SymInt32`` that TRT evaluates from the actual input shape at
#   runtime.
#
# TRT stores the compiled binary in ``QDP_REGISTRY["my::add_one"].aot_impl_func``.
# When ``generate_plugin_converter`` is later called with ``use_aot_if_available=True``
# it detects ``aot_impl_func is not None`` and sets ``aot=True`` on the plugin layer,
# causing TRT to use the binary path instead of a Python callback.


@trtp.aot_impl("my::add_one")
def add_plugin_aot_impl(
    X: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    # Choose the pointer type based on the input dtype.
    type_str = "fp32" if X.dtype == trt.float32 else "fp16"

    block_size = 256
    # Compile the Triton kernel to PTX now, at registration time.
    # ``ASTSource`` describes the kernel's input types and constexprs without
    # running it — Triton compiles it to architecture-specific PTX/CUBIN.
    src = triton.compiler.ASTSource(
        fn=add_one_kernel,
        signature={
            "x_ptr": f"*{type_str}",
            "n_elements": "i32",
            "y_ptr": f"*{type_str}",
        },
        constexprs={
            "BLOCK_SIZE": block_size,
        },
    )
    compiled_kernel = triton.compile(src)

    # Build symbolic launch parameters.
    # ``X.shape_expr.numel()`` is a symbolic expression for the total number of
    # elements — TRT will evaluate it to a concrete integer at engine runtime.
    N = X.shape_expr.numel()
    launch_params = trtp.KernelLaunchParams()
    launch_params.grid_x = trtp.cdiv(N, block_size)  # number of thread blocks
    launch_params.block_x = compiled_kernel.metadata.num_warps * 32  # threads per block
    launch_params.shared_mem = compiled_kernel.metadata.shared  # bytes of shared mem

    # ``extra_args`` are scalar arguments appended to the kernel's argument list at
    # launch. Here ``n_elements`` is passed as a 32-bit symbolic integer so TRT
    # evaluates it from the actual tensor size at runtime.
    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = trtp.SymInt32(N)

    return (
        compiled_kernel.metadata.name,  # kernel function name in PTX
        compiled_kernel.asm["ptx"],  # PTX source — embedded in TRT engine
        launch_params,
        extra_args,
    )


# %%
# Step 5: Generate the Converter
# -----------------------------------------
#
# Unlike the JIT example, we do **not** call ``generate_plugin`` here — the shape
# descriptor and AOT impl are already registered manually above.
# We only need the converter that bridges the Torch op to the TRT network layer.
#
# ``generate_plugin_converter`` finds ``"my::add_one"`` in ``QDP_REGISTRY``, sees
# that ``aot_impl_func is not None``, and creates a converter that calls
# ``ctx.net.add_plugin(plugin, aot=True)``. The ``aot=True`` flag instructs TRT to
# use the pre-compiled PTX rather than a Python JIT callback at runtime.

torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter(
    "my::add_one",
    supports_dynamic_shapes=False,
    requires_output_allocator=False,
    use_aot_if_available=True,
)


# %%
# Step 6: Compile and Run
# -----------------------------------------
#
# Compilation is identical to the JIT example. The difference is what happens at
# inference time: TRT launches the pre-compiled PTX kernel directly on the GPU with
# no Python involvement.


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        res = torch.ops.my.add_one.default(X)
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aot", action="store_true", help="Try to use AOT compilation", default=False
    )
    args = parser.parse_args()

    my_model = MyModel().to("cuda").eval()
    m = torch.full((64, 64), 2, device="cuda", dtype=torch.float)

    assert my_model(X=m)[0][0] == 3.0

    with torch_tensorrt.logging.debug():
        trt_inputs = [m]
        model_trt = torch_tensorrt.compile(
            my_model,
            inputs=trt_inputs,
            min_block_size=1,
        )
        print("Model compiled successfully!")
        print("Running inference with compiled model...")
        with torch.no_grad():
            for i in range(10):
                res = model_trt(m)
                assert torch.allclose(res, my_model(m)), "Results do not match!"

    print("Inference successful!")
