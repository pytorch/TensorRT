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
import triton
import triton.language as tl

import torch_tensorrt

trt_logger = trt.Logger(trt.Logger.VERBOSE)


# %%
# Step 1: Define the Triton Kernel
# -----------------------------------------
#
# Same as the JIT example — the kernel is pure Triton. The difference is how it
# gets compiled: in the JIT path ``add_one_kernel[grid](...)`` is called at runtime;
# in the AOT path it is compiled to PTX inside ``@trtp.aot_impl`` below.


@triton.jit
def add_one_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Arg order matters for the AOT path: TRT launches the embedded PTX with
    # arguments in (input_ptrs, output_ptrs, extra_args) order — inputs first,
    # then outputs, then anything from ``extra_args`` in ``@trtp.aot_impl``.
    # If this kernel declared ``(x_ptr, n_elements, y_ptr, ...)`` then TRT
    # would feed the output pointer into ``n_elements`` and ``n_elements``
    # into ``y_ptr`` at launch, which is a wild pointer dereference (engine
    # builds fine, ``enqueueV3`` returns -1 and the process segfaults).
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + 1
    tl.store(y_ptr + offsets, output, mask=mask)


@triton.jit
def add_one_inplace_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Distinct kernel for the aliased-I/O variant. The plugin descriptor
    # declares its output as ``X.aliased()`` — at runtime TRT passes a
    # *single* pointer for the aliased I/O pair (the shared buffer), not two.
    # If we re-used ``add_one_kernel`` here, TRT would supply: pointer,
    # n_elements, padding... and the kernel's ``y_ptr`` slot would absorb
    # ``n_elements`` while ``n_elements`` would read the padding zero — the
    # mask would be all-false and the kernel would do nothing.
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(x_ptr + offsets, x + 1, mask=mask)


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
    add_one_kernel[grid](X, Y, X.numel(), BLOCK_SIZE=BLOCK_SIZE)
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
            "y_ptr": f"*{type_str}",
            "n_elements": "i32",
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
    # launch. ``n_elements`` is passed as a 32-bit symbolic integer so TRT
    # evaluates it from the actual tensor size at runtime.
    #
    # Triton >= 3.x always emits two trailing ``.param .u64 .ptr`` slots in
    # the compiled PTX for ``global_scratch`` and ``profile_scratch`` — even
    # when their sizes (``compiled_kernel.metadata.global_scratch_size`` /
    # ``profile_scratch_size``) are 0. Triton's own runtime allocates
    # zero-sized scratch buffers and passes those pointers at launch; TRT's
    # AOT plugin path doesn't know about them and would otherwise leave the
    # two trailing slots filled with stale register state — symptom:
    # ``Failed to enqueue status -1`` and a segfault on the first call.
    # We pad ``extra_args`` with four ``SymInt32(0)`` (two per u64 slot) so
    # the kernel sees null pointers for both scratch params; since their
    # sizes are 0 the kernel never dereferences them.
    extra_args = trtp.SymIntExprs(1 + 4)
    extra_args[0] = trtp.SymInt32(N)
    for _i in range(1, 5):
        extra_args[_i] = trtp.SymInt32(0)

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
# In-place variant: aliased plugin I/O
# -----------------------------------------
#
# This second registration shows the same kernel exposed as an *in-place* plugin —
# the engine mutates the input buffer directly instead of allocating a separate
# output. Useful for KV-cache updates and any pattern where only a small slice of
# a large state changes per call.
#
# Three things change vs. ``my::add_one`` above:
#
# 1. ``mutates_args=("X",)`` on the torch op. This is the load-bearing signal —
#    it tells the QDP descriptor in ``_generate_plugin.py`` that input ``X`` is
#    a candidate for aliasing, and it also tells PyTorch's autograd and
#    functionalization machinery that the op has side effects on ``X``.
#
# 2. The registered fake returns ``X`` by identity (``return X``). Combined with
#    the schema's ``mutates_args``, this is what makes the descriptor emit
#    ``X.aliased()`` (see ``_generate_plugin._generic_plugin_desc``) instead of
#    building a fresh output ``TensorDesc``.
#
# 3. The descriptor itself uses ``X.aliased()``. ``aliased()`` returns a
#    ``TensorDesc`` that shares its data buffer with ``X`` — TRT will route the
#    same pointer to both the input and output binding at runtime.
#
# The eager torch impl has to mutate ``X`` itself (so the semantics match what
# the engine will do) and return ``X.clone()``. ``torch.library`` forbids
# returning an input by identity from a custom op, hence the clone.
#
# Note on the AOT kernel: we re-use ``add_one_kernel`` unchanged. Its signature
# takes two pointers (``x_ptr``, ``y_ptr``). With aliased I/O declared, TRT
# passes the same buffer for both — the kernel reads from ``x_ptr`` and writes
# to ``y_ptr``, which is the same memory, so the effect is in-place.


@torch.library.custom_op("my::add_one_inplace", mutates_args=("X",))  # type: ignore[misc]
def add_one_inplace(X: torch.Tensor) -> torch.Tensor:
    assert X.is_cuda
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(X.numel(), meta["BLOCK_SIZE"]),)
    add_one_inplace_kernel[grid](X, X.numel(), BLOCK_SIZE=BLOCK_SIZE)
    # Must not return X by identity — torch.library's no-alias constraint
    # rejects that. The TRT path doesn't observe this clone (aliasing is
    # declared at the descriptor level), it's purely for the eager impl.
    return X.clone()


@torch.library.register_fake("my::add_one_inplace")
def _(X: torch.Tensor) -> torch.Tensor:
    # Identity return is the secondary signal the descriptor uses to detect
    # aliasing. Combined with ``mutates_args=("X",)`` above, this is what
    # makes ``_generic_plugin_desc`` emit ``X.aliased()`` for the output.
    return X


@trtp.register("my::add_one_inplace")
def add_one_inplace_desc(X: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
    # ``aliased()`` is the QDP API that declares output-shares-storage-with-input.
    # Engine build will fail with "PreviewFeature::kALIASED_PLUGIN_IO_10_03 not
    # enabled" unless the build config enables that preview feature; the
    # converter wires this on for you when it sees a non-empty aliased_map.
    return X.aliased()


@trtp.aot_impl("my::add_one_inplace")
def add_one_inplace_aot_impl(
    X: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    type_str = "fp32" if X.dtype == trt.float32 else "fp16"

    block_size = 256
    # Single-pointer signature — see the comment on ``add_one_inplace_kernel``.
    src = triton.compiler.ASTSource(
        fn=add_one_inplace_kernel,
        signature={
            "x_ptr": f"*{type_str}",
            "n_elements": "i32",
        },
        constexprs={
            "BLOCK_SIZE": block_size,
        },
    )
    compiled_kernel = triton.compile(src)

    N = X.shape_expr.numel()
    launch_params = trtp.KernelLaunchParams()
    launch_params.grid_x = trtp.cdiv(N, block_size)
    launch_params.block_x = compiled_kernel.metadata.num_warps * 32
    launch_params.shared_mem = compiled_kernel.metadata.shared

    # See the matching note on the non-in-place ``add_plugin_aot_impl``:
    # Triton 3.x emits two trailing ``.param .u64 .ptr`` slots for the
    # global/profile scratch buffers, and TRT's AOT path needs them zeroed
    # explicitly via ``extra_args`` so the kernel doesn't read stale state.
    extra_args = trtp.SymIntExprs(1 + 4)
    extra_args[0] = trtp.SymInt32(N)
    for _i in range(1, 5):
        extra_args[_i] = trtp.SymInt32(0)

    return (
        compiled_kernel.metadata.name,
        compiled_kernel.asm["ptx"],
        launch_params,
        extra_args,
    )


torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter(
    "my::add_one_inplace",
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


class MyInplaceModel(torch.nn.Module):
    """Drives the in-place plugin. The op mutates ``X`` in place; the returned
    tensor carries the post-mutation value (a clone, only to satisfy
    torch.library's no-alias rule)."""

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.ops.my.add_one_inplace.default(X)


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

    # %%
    # In-place plugin demo
    # ---------------------
    #
    # The standard "compile once, run many times" comparison pattern doesn't
    # work for an in-place op because each call mutates the input — running
    # eager and TRT on the same buffer double-applies the mutation. We work
    # off a base tensor and clone for each call instead.
    #
    # Three things to verify, beyond "it ran":
    # 1. The compiled module contains a TRT engine (not a PyTorch fallback —
    #    a regression here would silently pass on value because the eager
    #    kernel mutates the input the same way).
    # 2. The engine's return value matches the expected post-mutation tensor.
    # 3. The user's input buffer was mutated in place by the engine — the
    #    actual reason to use aliased plugin I/O in the first place.

    print("\nIn-place plugin demo:")
    inplace_model = MyInplaceModel().to("cuda").eval()
    base = torch.full((64, 64), 2, device="cuda", dtype=torch.float)
    expected_post = base + 1

    model_trt_inplace = torch_tensorrt.compile(
        inplace_model,
        inputs=[base.clone()],
        min_block_size=1,
        immutable_weights=True,
    )

    from torch_tensorrt.dynamo.runtime import (
        PythonTorchTensorRTModule,
        TorchTensorRTModule,
    )

    engine_submodules = [
        m
        for _, m in model_trt_inplace.named_modules()
        if isinstance(m, (PythonTorchTensorRTModule, TorchTensorRTModule))
    ]
    assert engine_submodules, (
        "Expected a TRT engine submodule for the in-place plugin path, but the"
        " compiled module is pure PyTorch — check that the un-functionalize"
        " lowering pass restored the mutating op before partitioning. Graph:\n"
        f"{model_trt_inplace.graph}"
    )
    print(f"  TRT engine submodule(s) present: {len(engine_submodules)}")

    with torch.no_grad():
        trt_input = base.clone()
        trt_out = model_trt_inplace(trt_input)
        assert torch.allclose(trt_out, expected_post), "TRT output mismatch"
        assert torch.allclose(trt_input, expected_post), (
            "Engine did not mutate the input buffer — aliased plugin I/O is"
            " not active. Check that PreviewFeature.ALIASED_PLUGIN_IO_10_03"
            " was enabled and that the descriptor emitted X.aliased()."
        )

    print("  Output matches expected post-mutation value.")
    print("  Input buffer was mutated in place by the TRT engine.")
    print("In-place inference successful!")
