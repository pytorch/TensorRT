.. _execute_in_torch:

Region annotations: PyTorch, custom operators, and kernels
==========================================================

``torch_tensorrt.region.execute_in_torch()`` marks a group of tensor
operations that should stay together in PyTorch while eligible surrounding operations
compile to TensorRT. This experimental API is useful when you want to control
placement for a particular part of a model, even if its operations have TensorRT
converters. CUDA tensors remain on CUDA inside the region.

The optional ``backend`` selects how the region executes: ``"eager"`` runs its
captured PyTorch operations normally, and ``"inductor"`` uses PyTorch Inductor.
The default is ``"eager"``. Both choices keep the region outside TensorRT.
Two replacement APIs reuse implementations you supply:

* ``torch_tensorrt.region.fused_op(op)`` calls an existing custom operator in
  PyTorch, outside TensorRT.
* ``torch_tensorrt.region.fused_kernel(...)`` runs a custom kernel inside
  TensorRT, registering its plugin automatically.

This tutorial requires a CUDA device and compatible PyTorch and Torch-TensorRT
installations. Capture currently uses private PyTorch APIs;
missing capture capabilities produce an error when an annotation is used.

Three complete examples are available:

* :ref:`execute_in_torch_example` runs the same region with eager and Inductor.
* :ref:`fused_op_example` replaces the region with a PyTorch custom operator.
* :ref:`fused_kernel_example` replaces the region with a supplied Triton kernel.

Run them from the repository root:

.. code-block:: bash

    python examples/dynamo/execute_in_torch.py
    python examples/dynamo/fused_op.py
    python examples/dynamo/fused_kernel.py

The eager/Inductor example requires Triton support for CUDA Inductor. The
custom-operator example requires neither Triton nor QDP. The plugin example
requires Triton >=3.5 and TensorRT QDP support; see
:ref:`the kernel API prerequisites <torch_tensorrt_kernels_py>`.

Mark a region
-----------------

This model computes ``relu(x + 1) * 2``. Only the ReLU operation is marked to run
in PyTorch. Addition and multiplication remain eligible for TensorRT.

.. code-block:: python

    import torch
    import torch_tensorrt


    class Model(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            with torch_tensorrt.region.execute_in_torch(name="activation"):
                x = torch.relu(x)
            return x * 2


    model = Model().eval().cuda()
    x = torch.randn(4, 16, device="cuda")

The optional ``name`` labels diagnostics. Two scopes with the same name are
separate occurrences; leaving out ``name`` is also supported. The annotation is
a no-op during ordinary eager execution.

Compile and check the result
--------------------------------

Pass the model directly to ``torch_tensorrt.compile`` with ``ir="dynamo"``,
``strict=True``, and ``offload_module_to_cpu=False``. Strict capture records the
region, and disabling CPU weight offload keeps any state it uses on its device.

.. code-block:: python

    compiled = torch_tensorrt.compile(
        model,
        inputs=[x],
        ir="dynamo",
        strict=True,
        offload_module_to_cpu=False,
        min_block_size=1,
    )

    with torch.no_grad():
        expected = model(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected)

``min_block_size=1`` allows the single operations on either side of this small
example to become TensorRT engines. Placement outside the scope still depends
on converter support and the minimum block size in your application. Both the
fast and global partitioners preserve the region.

The resulting execution order is::

    TensorRT addition -> PyTorch ReLU -> TensorRT multiplication

.. important::

    Use the direct compile entry above; only it captures regions. Other entry
    points trace annotated code as ordinary operations: ``torch.compile`` with
    any backend, including ``backend="tensorrt"`` and
    ``torch_tensorrt.compile(..., ir="torch_compile")``;
    ``MutableTorchTensorRTModule``; and compiling a program exported separately
    with ``torch.export`` or ``torch_tensorrt.dynamo.trace``. There, an
    ``execute_in_torch`` body can be converted to TensorRT, and the operator or
    kernel given to ``fused_op`` or ``fused_kernel`` is not used. Torch-TensorRT
    logs a warning the first time one of these traces ignores an annotation.

Compile a region with Inductor
---------------------------------

Inside ``forward``, set ``backend="inductor"`` on the scope and use the same
compilation settings as above:

.. code-block:: python

    with torch_tensorrt.region.execute_in_torch(backend="inductor"):
        x = torch.relu(x)

The surrounding operations remain eligible for TensorRT. The first call to
the compiled model compiles the region with Inductor; later calls reuse it.
See :ref:`execute_in_torch_example` for a complete model and execution checks.

Inductor compiles the whole region with static shapes. It may fuse compatible
operations, but a region does not guarantee one GPU kernel. Compilation errors
are reported to the caller instead of silently falling back to eager execution.

Inductor compilation is lazy: completing ``torch_tensorrt.compile`` does not mean
the region has already been compiled, and a ``copy.deepcopy`` of the compiled
model compiles the region again on the copy's first call. Keep the same inference
context for warmup and measured calls, and exclude the first call from
steady-state timings. Changes to input layout or execution context can cause
recompilation even when shapes are unchanged. Dry runs retain the reference child
and do not install an Inductor callable.

Only ``"eager"`` and ``"inductor"`` are accepted. ``backend`` is the compiler
choice, not the ``mode`` argument of ``torch.compile``: values such as
``"reduce-overhead"`` and ``"max-autotune"`` are not valid here. The region API
does not expose compiler tuning options. Its Inductor configuration disables
``triton.cudagraphs``. For whole-model CUDA graphs, use
``torch_tensorrt.runtime.enable_cudagraphs(compiled)``. For either eager
or Inductor regions, the runtime warms up the model before capture in the
caller's execution context. Under ``torch.autocast``, warmup and capture bypass
autocast's weight cache, as with ``cache_enabled=False``: the graph records the
casts, so each replay casts the current weights.

CUDA graph capture requires strided tensor inputs on one CUDA device. It copies
the bytes each input reaches, from its first element to its last, and keeps its
strides; inputs whose bytes overlap, such as the results of ``chunk``, share one
copy. The CUDA graphs tutorial lists
:ref:`what triggers capture again <cuda_graphs_recording>`. As with
``torch.compile``, a region that reads outside its input views, for example with
``as_strided`` and an explicit ``storage_offset``, can fail or read different
values under CUDA graphs. The copies keep the memory alignment that the inputs
had at capture, so kernels that pick a vectorized path by alignment, such as
reductions, can make replays with differently aligned inputs, such as other
slices of a larger tensor, differ from eager execution in the last bits.
Warmup preserves the default PyTorch CPU and CUDA RNG states, including on recapture.
Capture cannot copy unpinned CPU tensors to the GPU. Eager regions copy CPU
scalars passed to operations such as ``torch.where``, and TensorRT engines copy
CPU scalars that regions return; create such constants on the input's device.

As with the default backend, this context manager does nothing during ordinary
eager execution. It enables Inductor only through Torch-TensorRT compilation.
The ``name`` keyword remains optional and can be combined with either backend.

Replace a region with a PyTorch custom operator
-----------------------------------------------

Use ``fused_op`` when an existing custom operator implements the region and
should execute through PyTorch. Register the operator and its fake implementation
before compiling. The fake implementation describes the output without running
the real implementation during capture.

.. code-block:: python

    @torch.library.custom_op("region_tutorial::bias_relu_scale", mutates_args=())
    def bias_relu_scale(
        x: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return torch.relu(x + bias) * scale


    @bias_relu_scale.register_fake
    def bias_relu_scale_fake(x, bias, scale):
        return torch.empty_like(x)


    class CustomOpModel(torch.nn.Module):
        def forward(self, x, bias, scale):
            x = x + 1.0
            with torch_tensorrt.region.fused_op(
                torch.ops.region_tutorial.bias_relu_scale.default,
                inputs=(x, bias, scale),
                name="bias_relu_scale",
            ):
                x = torch.relu(x + bias) * scale
            return x - 0.5

Compile an eval-mode instance with three equal-shaped CUDA inputs using the
direct compilation settings above. The execution order is::

    TensorRT addition -> PyTorch custom operator -> TensorRT subtraction

The body defines the reference calculation and runs during ordinary eager
execution. Compilation replaces the entire body with the supplied operator;
it does not generate a new implementation or a TensorRT plugin. See
:ref:`fused_op_example` for a complete run that checks outputs, placement, and
actual invocation of the custom operator.

Passing the decorated ``bias_relu_scale`` function, or
``torch.ops.region_tutorial.bias_relu_scale`` without ``.default``, selects
the same operator.

The operator schema must accept positional Tensor arguments and return tensors,
with no mutation or aliasing. ``inputs`` binds the operator's arguments: list
every tensor the region reads, including parameters and buffers, in the
operator's argument order. Here the first argument is the result of ``x + 1``,
followed by ``bias`` and ``scale``. Compilation cannot infer this order from the
body: in ``up * F.silu(gate)``, ``silu`` reads ``gate`` before the
multiplication reads ``up``. ``inputs`` may be omitted only when the region
reads at most one tensor, and compilation rejects a list that omits, repeats, or
adds a tensor. A tensor literal such as ``torch.tensor(2.0)`` is lifted out of
the region as an input, so create it before the region and list it. Outputs
follow their production order in the region. The registered fake implementation
must match the captured output metadata. These boundaries currently have static
shapes. The caller must verify that the actual operator implements the complete
reference calculation.

``fused_op`` also permits random operations in the reference body, for custom
operators that own their RNG behavior. Numerical tests must account for the
operator's random-state contract. ``execute_in_torch`` and ``fused_kernel``
continue to reject random regions.

Register the custom operator's schema and implementation before loading a saved
model in a new process. Saving to ExecuTorch (see `Save the compiled model`_)
keeps the custom operator in the ``.pte`` as its ``out`` overload, so define
one; without it, ExecuTorch fails with ``Missing out variants``:

.. code-block:: python

    torch.library.define(
        "region_tutorial::bias_relu_scale.out",
        "(Tensor x, Tensor bias, Tensor scale, *, Tensor(a!) out) -> Tensor(a!)",
    )


    @torch.library.impl(
        "region_tutorial::bias_relu_scale.out", "CompositeExplicitAutograd"
    )
    def bias_relu_scale_out(x, bias, scale, *, out):
        return out.copy_(torch.relu(x + bias) * scale)

The application's ExecuTorch runtime must register a kernel for that overload.

Replace a region with a custom-kernel plugin
----------------------------------------------

Use ``fused_kernel`` when you already have a kernel that implements the entire
annotated computation. For the Triton kernel and launch callbacks defined in
:ref:`fused_kernel_example`, the annotation is:

.. code-block:: python

    with torch_tensorrt.region.fused_kernel(
        kernel=lambda: bias_relu_scale_kernel,
        inputs=(x, bias, scale),
        constexprs={"BLOCK": BLOCK},
        grid=launch_grid,
        extra_args_fn=kernel_extras,
    ):
        x = x + bias
        x = torch.relu(x)
        x = x * scale

``kernel`` must be a zero-argument function returning an existing ``@triton.jit``
kernel or a ``torch_tensorrt.kernels.KernelSpec`` for CUDA C++; passing the
kernel itself is rejected, even in eager execution. The lambda lets the
compiler obtain the kernel outside the traced tensor program, so a lambda
written in ``forward`` must refer to a global kernel, not to ``self`` or a
local variable. To use a kernel chosen in ``__init__``, create the factory
there, for example ``self.kernel_factory = lambda: kernel``, and pass
``kernel=self.kernel_factory``. ``grid`` and ``extra_args_fn`` follow the same
rules: all three must be functions or ``functools.partial`` objects, not
methods. ``name`` is optional.

Supply the kernel and its launch configuration. Torch-TensorRT uses the
annotated operations to determine the kernel's inputs and outputs, then
registers the operator and TensorRT plugin automatically. You do not write
registration functions or an FX rewrite. The selected operations must become
part of a TensorRT engine; compilation reports an error if that is impossible.
A dry run logs that error as a warning and names the region in its report.
Calling the original model eagerly still executes the PyTorch body.

Generated operator and plugin registrations remain for the lifetime of the
Python process. Recompiling a model, including a compilation that fails after
registration, can accumulate registrations; registration reuse and cleanup
are not implemented.

For Triton, runtime kernel parameters must be ordered as input pointers,
optional ``i32`` scalar extras, then output pointers. Input pointers follow
``inputs``, which lists tensors as it does for ``fused_op``, including
parameters and buffers. Outputs are the tensors used after the scope, in the
order they were produced inside it. This can differ from the model's final
return order: if ``a`` is computed before ``b`` and both leave the region, the
output pointers are for ``a``, then ``b``, even if the model returns ``(b, a)``.
Pointer dtypes are inferred from capture. Supply ``constexprs`` and the
required ``grid(inputs, outputs)`` callback, plus
``extra_args_fn(inputs, outputs)`` when scalar launch arguments are needed.
Those callbacks receive TensorRT tensor descriptors. A CUDA ``KernelSpec``
provides its own launch configuration.

Plugin boundaries currently require static, contiguous CUDA tensors with the
captured shapes and dtypes. Inputs and outputs must satisfy the region aliasing
rules below. The kernel author is responsible for numerical equivalence to the
entire body: automatic registration does not generate the kernel or prove its
correctness. Compare the compiled result against the original model on
representative inputs, as the runnable example does.

Use ``min_block_size=1`` for a small example. ``require_full_compilation=True``
is allowed when the model has plugin regions and no ``execute_in_torch`` or
``fused_op`` regions. The example compiles its surrounding add and subtract plus
the custom kernel into one TensorRT engine.

Select an invocation of a shared module
-------------------------------------------

A scope applies only to the operations captured inside it. For example, if
``self.block`` is called three times, the following marks only its second call:

.. code-block:: python

    class SharedBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU())

        def forward(self, x):
            x = self.block(x)
            with torch_tensorrt.region.execute_in_torch():
                x = self.block(x)
            return self.block(x)

An ``execute_in_torch`` region can consume parameters or buffers and return
multiple independent tensors. Noncontiguous inputs are supported. Tensor
constants created with ``torch.tensor`` inside a functional region are
supported too. Dtype casts such as ``.float()`` or ``.to(torch.float16)`` are
supported inside a region and on its outputs, for example to compute a
normalization in fp32 within an fp16 model. The body must still be exportable
tensor code; the annotation does not restore Python hooks or side effects
removed by export.

Region inputs may share storage, such as the results of ``chunk`` or
``unbind``, a tensor and its transpose, or the weights of a CUDA ``nn.LSTM``,
which cuDNN flattens into one buffer. ``execute_in_torch`` regions read these
inputs without copying them. ``fused_op`` and ``fused_kernel`` regions receive a
copy of each input that shares storage with an earlier one: parameters and
buffers are copied once during compilation, and other inputs may be copied on
every call.

Save the compiled model
---------------------------

The legacy exported-program path supports ``execute_in_torch`` regions with
``backend="eager"``, ``fused_op`` regions, and ``fused_kernel`` regions already
built into TensorRT engines. Save the already chosen placement as follows:

.. code-block:: python

    torch_tensorrt.save(
        compiled,
        "model.ep",
        output_format="exported_program",
        retrace=False,
        use_legacy_exporter=True,
    )
    restored = torch_tensorrt.load("model.ep").module()

    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x))

Serialization works on a copy of the graph, leaving the compiled module intact.
Reloading executes the selected placement; it does not recreate an annotation
for a new optimization pass.

To deploy with ExecuTorch, save the same graph as a ``.pte``. This format does
not retrace by default:

.. code-block:: python

    torch_tensorrt.save(compiled, "model.pte", output_format="executorch")

Each TensorRT engine becomes a TensorRT delegate. An ``execute_in_torch`` region
runs on ExecuTorch's CPU kernels, like any operator TensorRT does not take,
unless a ``CudaPartitioner`` claims it (see :ref:`executorch_deployment`). A
``fused_op`` region calls the kernel the application registers for the
operator's ``out`` overload. TorchScript, AOTInductor, ``retrace=True``, and
``use_legacy_exporter=False`` are not supported for regions.

Built plugin engines include the compiled kernel, so loading them does not
require registering the Python kernel or plugin again. Plugin dry-run graphs
cannot be saved: they contain no built engine, and the exporter rejects them.

Models containing ``backend="inductor"`` regions cannot currently be saved or
exported through this path. The exporter serializes the reference FX operations,
which cannot preserve the installed Inductor callable. Such saves raise an
explicit error. Compile the original model again with ``backend="eager"`` if
you need the supported save/load workflow; loading a saved model does not enable
Inductor automatically.

Supported scope and limitations
-----------------------------------

The initial API supports eval-mode functional inference with static shapes,
Tensor boundary values, read-only state, and one device per region. As in
PyTorch, 0-dim CPU tensors such as ``torch.tensor(2.0)`` or a scalar buffer can
be used with that device's tensors, except at ``fused_kernel`` boundaries,
which require CUDA tensors.

* Models containing live regions and complex tensors are rejected, including
  complex computations outside the region. Complex lowering does not yet
  preserve the region boundary correctly under either complex lowering mode.
* Nested scopes, graph breaks, and mutation are rejected. Operations that
  draw random numbers, such as ``torch.rand_like`` or dropout with
  ``training=True``, are supported only in ``fused_op`` regions. Eval-mode
  dropout and attention with ``dropout_p=0`` draw none and are allowed in
  every region. Reassigning a Python variable, such as ``x = torch.relu(x)``,
  is allowed.
* A region cannot contain, or be inside, a construct that export captures as
  a higher-order operator: ``torch.cond`` and other control-flow operators,
  ``flex_attention``, ``torch.autocast``, and grad-mode changes such as
  ``torch.no_grad()``. Use explicit casts such as ``.float()`` in place of
  ``torch.autocast``. A grad-mode block is captured only when it changes the
  grad mode in effect during compilation, so compile under ``torch.no_grad()``
  when a region contains or is inside a ``torch.no_grad()`` block.
* Compilation still traces region bodies through Torch-TensorRT's
  decompositions, so an operator whose decomposition raises an error fails
  compilation even inside a region.
* Mutation of model inputs or registered buffers anywhere in a model with live
  regions is rejected, including outside the annotation. Mutation of local
  temporaries outside a region remains supported.
* Region outputs must not share storage with inputs or with each other. Clone
  escaping views or ``chunk`` results inside the scope to give the outputs
  separate storage. An operation that returns its input itself, such as
  eval-mode dropout or a cast to the input's own dtype, makes that input an
  output of a ``fused_op`` or ``fused_kernel`` region when code after the
  region reads the input or the result; clone the input inside the scope
  before that operation.
* Disable CPU weight offload, ``enable_autocast``, resource partitioning,
  refit, and weight streaming. Cross-compilation and engine-only output are not
  supported. Conflicting dry runs return a reference graph and report conflicts
  without building engines or offloading state.
  ``require_full_compilation=True`` is incompatible with ``execute_in_torch``
  and ``fused_op`` regions because they must remain outside TensorRT.
  ``hardware_compatible=True`` is incompatible with ``fused_kernel`` regions
  because their kernel targets the compiling GPU's architecture.
* Empty or dead pure scopes are no-ops. A scope whose outputs are used only for
  their shape or dtype, as by ``torch.full_like``, can become dead during
  compilation, but it still counts as a live region for the settings above,
  static shapes, and complex tensors. A dead ``fused_op`` scope that draws
  random numbers is rejected. Dynamic shapes and training are outside the
  initial scope.

``execute_in_torch`` and ``fused_op`` select supplied implementations.
``fused_kernel`` uses a supplied kernel factory and launch configuration, or
generates one during compilation with ``generator=KernelGenerator(...)``. See
the :ref:`agent-generated kernel example <agentic_region_kernel_generation_example>`
and the :ref:`region annotation API <torch_tensorrt_region_py>` for search,
resume, and provider configuration.

Inspect placement and measure performance
---------------------------------------------

To check placement, inspect ``compiled.graph`` and its submodules: each
``torch_tensorrt.runtime.TorchTensorRTModule`` is one TensorRT engine, and other
calls run in PyTorch. An engine's ``get_layer_info()`` lists its layers,
including ``fused_kernel`` plugins. For compiler development, the internal
manifest returned by
``torch_tensorrt.dynamo.region_compilation.get_region_records(compiled)``
records each region's placement; it can change without notice.

For ``execute_in_torch``, capture extracts each region into a child FX graph
and makes it an opaque PyTorch call for partitioning. The child keeps the
operations export captured; Torch-TensorRT's TensorRT-oriented decompositions
do not rewrite it. Audits check that graph rewrites preserve the child and
keep it outside TensorRT partitions. ``fused_op`` uses the same boundary but
replaces the child body with a custom-op call. No capture marker executes at
runtime.

Each PyTorch region limits TensorRT fusion and can introduce extra
engine calls around a small block. Plugin regions keep the supplied kernel
inside TensorRT, but do not guarantee a speedup. Measure full-model latency and
engine counts against the same model without annotations. Fewer or larger
scopes may be faster; the annotation itself does not promise a performance
improvement.
