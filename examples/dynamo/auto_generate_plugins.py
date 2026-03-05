"""
.. _auto_generate_plugins:

Automatically Generate a Plugin for a Custom Kernel
===================================================================

This example demonstrates how to register a custom Triton kernel as a TensorRT plugin
using the TensorRT 10.7+ Quick Deployable Plugin (QDP) system, and how Torch-TensorRT
automatically generates the converter that wires the two together.

Without a plugin, a custom op would fall back to PyTorch at runtime, causing a graph
break between two TRT subgraphs. The plugin approach runs the custom kernel *inside*
the TRT engine, avoiding that overhead entirely.

**What "automatically generate" means here:**

``generate_plugin`` uses PyTorch's FakeTensor/symbolic-shape machinery to introspect
your op's schema at registration time. It synthesizes:

* A *shape descriptor* function (``_generic_plugin_desc``) that computes output shapes
  from symbolic input dimensions using ``lambdify`` expressions — this is how TRT knows
  output shapes without running the kernel.
* A *JIT implementation* function (``_generic_plugin_impl``) that, at TRT engine
  runtime, converts TRT tensors back to PyTorch tensors, calls your op directly on the
  CUDA stream TRT provides, and copies results to the output buffers.

Both are registered in TensorRT's ``QDP_REGISTRY`` under ``"torchtrt_ex::elementwise_scale_mul"``.

``generate_plugin_converter`` then creates and registers a
``@dynamo_tensorrt_converter`` for ``torch.ops.torchtrt_ex.elementwise_scale_mul.default``
in Torch-TensorRT's ``DYNAMO_CONVERTERS`` table. When the compiler encounters that op
in the FX graph it calls this converter, which instantiates the QDP plugin and adds a
plugin layer to the TRT ``INetworkDefinition``.

**JIT vs AOT:** The plugin generated here is JIT — at TRT engine runtime, TRT calls
back into Python to execute the Triton kernel via PyTorch. For a pre-compiled binary
that avoids the Python overhead see the :ref:`aot_plugin` example.

See also :ref:`custom_kernel_plugins` for the lower-level
``IPluginV2DynamicExt`` approach that predates TRT 10.7.
"""

# %%
# Step 1: Define the Triton Kernel
# -----------------------------------------
#
# The kernel itself is pure Triton — no TRT-specific code at this stage.
# ``generate_plugin`` will later wrap it in a JIT implementation that TRT
# can call at runtime.

from typing import Tuple

import tensorrt_bindings.plugin as trtp
import torch
import torch_tensorrt
import triton
import triton.language as tl


@triton.jit
def elementwise_scale_mul_kernel(X, Y, Z, a, b, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Compute the range of elements that this thread block will work on
    block_start = pid * BLOCK_SIZE
    # Range of indices this thread will handle
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load elements from the X and Y tensors
    x_vals = tl.load(X + offsets)
    y_vals = tl.load(Y + offsets)
    # Perform the element-wise multiplication
    z_vals = x_vals * y_vals * a + b
    # Store the result in Z
    tl.store(Z + offsets, z_vals)


# %%
# Step 2: Register the Op with PyTorch
# -----------------------------------------
#
# ``@torch.library.custom_op`` registers the kernel as a first-class PyTorch op.
# This is what lets you call it as ``torch.ops.torchtrt_ex.elementwise_scale_mul``
# in model forward passes and have ``torch.export`` trace through it.
#
# ``@torch.library.register_fake`` registers the *meta-kernel* (also called a fake
# kernel or abstract impl). This function runs on ``FakeTensor`` objects — it must
# return a tensor of the correct *shape and dtype* without doing any actual compute.
# Three systems depend on it:
#
# * ``torch.export`` / Dynamo — for tracing shape propagation.
# * ``generate_plugin`` — it runs your meta-kernel symbolically with ``FakeTensorMode``
#   to derive the output-shape expressions it embeds in the QDP shape descriptor.
# * Torch-TensorRT's partitioner — to decide whether the op can be included in a TRT
#   subgraph.


@torch.library.custom_op("torchtrt_ex::elementwise_scale_mul", mutates_args=())  # type: ignore[misc]
def elementwise_scale_mul(
    X: torch.Tensor, Y: torch.Tensor, b: float = 0.2, a: int = 2
) -> torch.Tensor:
    assert X.is_cuda and Y.is_cuda, "Tensors must be on CUDA device."
    assert X.shape == Y.shape, "Tensors must have the same shape."

    Z = torch.empty_like(X)
    BLOCK_SIZE = 1024
    grid = lambda meta: (X.numel() // meta["BLOCK_SIZE"],)
    elementwise_scale_mul_kernel[grid](X, Y, Z, a, b, BLOCK_SIZE=BLOCK_SIZE)
    return Z


@torch.library.register_fake("torchtrt_ex::elementwise_scale_mul")
def _(x: torch.Tensor, y: torch.Tensor, b: float = 0.2, a: int = 2) -> torch.Tensor:
    # Elementwise — output has the same shape and dtype as the first input.
    return x


# %%
# Step 3: Auto-Generate the TensorRT QDP Plugin
# -----------------------------------------
#
# ``generate_plugin`` does the following internally:
#
# 1. Calls your ``register_fake`` function with ``FakeTensor`` objects carrying
#    symbolic ``SymInt`` shapes (via ``ShapeEnv``). This produces symbolic output-shape
#    expressions like ``s0 * s1``.
# 2. Turns those expressions into Python lambda functions with ``lambdify``, and
#    builds a ``_generic_plugin_desc`` that computes TRT ``TensorDesc`` output shapes
#    at graph-construction time.
# 3. Builds a ``_generic_plugin_impl`` that TRT calls at engine *runtime*:
#    it converts each TRT tensor handle to a ``torch.Tensor``, runs
#    ``torch.ops.torchtrt_ex.elementwise_scale_mul`` on the provided CUDA stream,
#    then copies results back to TRT's output buffers.
# 4. Registers both under ``"torchtrt_ex::elementwise_scale_mul"`` in TensorRT's
#    global ``QDP_REGISTRY``.
#
# After this call, ``trtp.op.torchtrt_ex.elementwise_scale_mul`` exists and TRT
# knows how to compute output shapes and execute the kernel.

torch_tensorrt.dynamo.conversion.plugins.generate_plugin(
    "torchtrt_ex::elementwise_scale_mul"
)


# %%
# Step 4: Auto-Generate the Torch-TensorRT Converter
# -------------------------------------------------------------------
#
# ``generate_plugin_converter`` does the following internally:
#
# 1. Looks up ``"torchtrt_ex::elementwise_scale_mul"`` in ``QDP_REGISTRY`` and checks
#    whether an AOT implementation is registered (``desc.aot_impl_func``). Here there
#    is none, so it uses the JIT path.
# 2. Defines a converter function that, when called during TRT graph construction:
#    a. Splits ``args`` into tensor inputs (converted to ``trt.ITensor`` via
#       ``get_trt_tensor``) and non-tensor attributes (scalars, passed as plugin attrs).
#    b. Instantiates the QDP plugin via ``trtp.op.torchtrt_ex.elementwise_scale_mul(...)``.
#    c. Calls ``ctx.net.add_plugin(plugin, aot=False)`` to add a plugin layer to the
#       TRT ``INetworkDefinition``.
# 3. Registers the converter for ``torch.ops.torchtrt_ex.elementwise_scale_mul.default``
#    in Torch-TensorRT's ``DYNAMO_CONVERTERS`` table via the
#    ``@dynamo_tensorrt_converter`` decorator.
#
# From this point, whenever the compiler encounters that op in the FX graph, it will
# call this converter and emit a plugin layer instead of a PyTorch fallback.
#
# ``supports_dynamic_shapes=True`` tells the registry that this converter can handle
# symbolic batch dimensions. ``requires_output_allocator=False`` means TRT knows the
# output size at engine-build time (not data-dependent).

torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter(
    "torchtrt_ex::elementwise_scale_mul",
    supports_dynamic_shapes=True,
    requires_output_allocator=False,
)


# %%
# The two calls above can be combined into one:
#
# .. code-block:: python
#
#     torch_tensorrt.dynamo.conversion.plugins.custom_op(
#         "torchtrt_ex::elementwise_scale_mul",
#         supports_dynamic_shapes=True,
#         requires_output_allocator=False,
#     )


# %%
# Step 5: Compile and Run
# -------------------------------------------------------------------
#
# From here, compilation is identical to any other Torch-TensorRT model.
# ``torch_tensorrt.compile`` will:
#
# * Export the model with ``torch.export``.
# * Partition the FX graph — the custom op node lands in a TRT subgraph because its
#   converter is registered.
# * During TRT graph construction the converter is called, adding a plugin layer.
# * At inference time, TRT calls ``_generic_plugin_impl``, which invokes the Triton
#   kernel on TRT's CUDA stream.


class MyModel(torch.nn.Module):  # type: ignore[misc]
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.add(x, y)
        res = torch.ops.torchtrt_ex.elementwise_scale_mul.default(x, z, b=0.5)
        return res


my_model = MyModel().to("cuda").eval()
m = torch.randint(0, 5, (64, 64), device="cuda", dtype=torch.float)
n = torch.randint(0, 5, (64, 64), device="cuda", dtype=torch.float)

with torch_tensorrt.logging.errors():
    model_trt = torch_tensorrt.compile(my_model, inputs=[m, n], min_block_size=1)
    with torch.no_grad():
        for i in range(300):
            res = model_trt(m, n)
            assert torch.allclose(res, my_model(m, n))

print("Ran with custom plugin!")
