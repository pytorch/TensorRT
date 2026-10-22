.. _execute_in_torch:

Region annotations: PyTorch and custom kernels
==============================================

``torch_tensorrt.region_annotations.execute_in_torch()`` marks a group of tensor
operations that should stay together in PyTorch while eligible surrounding operations
compile to TensorRT. This experimental API is useful when you want to control
placement for a particular part of a model, even if its operations have TensorRT
converters. CUDA tensors remain on CUDA inside the region.

The optional ``backend`` selects how the region executes: ``"eager"`` runs its
captured PyTorch operations normally, and ``"inductor"`` uses PyTorch Inductor.
The default is ``"eager"``. Both choices keep the region outside TensorRT.
``torch_tensorrt.region_annotations.fuse_in_plugin()`` instead replaces a region
with a supplied custom kernel inside TensorRT, registering its plugin automatically.

This tutorial requires a CUDA device and a Torch-TensorRT installation targeting
PyTorch 2.16 development builds. Capture currently uses private PyTorch APIs;
missing capture capabilities produce an error when an annotation is used.

Two complete examples are available:

* :ref:`execute_in_torch_example` runs the same region with eager and Inductor.
* :ref:`fuse_in_plugin_example` replaces the region with a supplied Triton kernel.

Run them from the repository root:

.. code-block:: bash

    python examples/dynamo/execute_in_torch.py
    python examples/dynamo/fuse_in_plugin.py

Both runnable examples require Triton support on the CUDA device: the first
uses CUDA Inductor, and the second supplies a Triton kernel. The plugin example
additionally requires Triton >=3.5 and TensorRT QDP support; see
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
            with torch_tensorrt.region_annotations.execute_in_torch(name="activation"):
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

    Use the direct compile entry above. Annotations are no-ops in standalone
    ``torch.export`` and other compiler sessions. Exporting the model separately
    loses the annotation before Torch-TensorRT can capture it.

Compile a region with Inductor
---------------------------------

Use ``backend="inductor"`` to ask PyTorch to compile the selected child graph.
For example, this region contains addition, ReLU, and multiplication:

.. code-block:: python

    class InductorModel(torch.nn.Module):
        def forward(self, x, bias, scale):
            x = x + 1.0
            with torch_tensorrt.region_annotations.execute_in_torch(backend="inductor"):
                x = x + bias
                x = torch.relu(x)
                x = x * scale
            return x - 0.5


    inductor_model = InductorModel().eval().cuda()
    inductor_inputs = tuple(torch.randn(4, 128, device="cuda") for _ in range(3))
    compiled_inductor = torch_tensorrt.compile(
        inductor_model,
        inputs=list(inductor_inputs),
        ir="dynamo",
        strict=True,
        offload_module_to_cpu=False,
        min_block_size=1,
    )

    with torch.inference_mode():
        expected = inductor_model(*inductor_inputs)
        # The first actual call triggers Inductor compilation of the region.
        torch.testing.assert_close(compiled_inductor(*inductor_inputs), expected)
        # Subsequent calls can reuse the compiled region.
        torch.testing.assert_close(compiled_inductor(*inductor_inputs), expected)

The intended execution order is::

    TensorRT addition -> Inductor region (add, ReLU, multiply) -> TensorRT subtraction

Inductor compiles the whole region with static shapes. It may fuse compatible
operations, but a region does not guarantee one GPU kernel. Compilation errors
are reported to the caller instead of silently falling back to eager execution.

Inductor compilation is lazy: completing ``torch_tensorrt.compile`` does not mean
the region has already been compiled. Keep the same inference context for warmup
and measured calls, and exclude the first call from steady-state timings.
Changes to input layout or execution context can cause recompilation even when
shapes are unchanged. Dry runs retain the reference child and do not install an
Inductor callable.

Only ``"eager"`` and ``"inductor"`` are accepted. ``backend`` is the compiler
choice, not the ``mode`` argument of ``torch.compile``: values such as
``"reduce-overhead"`` and ``"max-autotune"`` are not valid here. The region API
does not expose compiler tuning options. Its Inductor configuration disables
``triton.cudagraphs``. For whole-model CUDA graphs, use
``torch_tensorrt.runtime.enable_cudagraphs(compiled_inductor)``. For either eager
or Inductor regions, the runtime warms up the model before capture in the
caller's execution context. Changing that context or CUDA stream triggers
capture again.

As with the default backend, this context manager does nothing during a direct
eager call such as ``inductor_model(*inductor_inputs)``. It enables Inductor only
when the model goes through the direct Torch-TensorRT compilation above.
The ``name`` keyword remains optional and can be combined with either backend.

See :ref:`execute_in_torch_example` to run both backends on the same model.

Replace a region with a custom-kernel plugin
----------------------------------------------

Use ``fuse_in_plugin`` when you already have a kernel that implements the entire
annotated computation. For the Triton kernel and launch callbacks defined in
:ref:`fuse_in_plugin_example`, the annotation is:

.. code-block:: python

    with torch_tensorrt.region_annotations.fuse_in_plugin(
        kernel=lambda: bias_relu_scale_kernel,
        constexprs={"BLOCK": BLOCK},
        grid=launch_grid,
        extra_args_fn=kernel_extras,
    ):
        x = x + bias
        x = torch.relu(x)
        x = x * scale

``kernel`` must be a zero-argument factory returning an existing ``@triton.jit``
kernel or a ``torch_tensorrt.kernels.KernelSpec`` for CUDA C++. The lambda lets
the compiler obtain the kernel outside the traced tensor program. An inline
factory should refer to a global kernel; a factory created before capture may
close over a local kernel. ``name`` is optional.

Supply the kernel and its launch configuration. Torch-TensorRT uses the
annotated operations to determine the kernel's inputs and outputs, then
registers the operator and TensorRT plugin automatically. You do not write
registration functions or an FX rewrite. The selected operations must become
part of a TensorRT engine; compilation reports an error if that is impossible.
Calling the original model eagerly still executes the PyTorch body.

Generated operator and plugin registrations remain for the lifetime of the
Python process. Recompiling a model, including a compilation that fails after
registration, can accumulate registrations; registration reuse and cleanup
are not implemented.

For Triton, runtime kernel parameters must be ordered as input pointers,
optional ``i32`` scalar extras, then output pointers. Inputs follow their
first-use order in the captured region, including parameter and buffer inputs.
Outputs are the tensors used after the scope, in the order they were produced
inside it. This can differ from the model's final return order: if ``a`` is
computed before ``b`` and both leave the region, the output pointers are for
``a``, then ``b``, even if the model returns ``(b, a)``.
Pointer dtypes are inferred from capture. Supply ``constexprs``, a
``grid(inputs, outputs)`` callback, and
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
is allowed when the model has plugin regions and no ``execute_in_torch``
regions. The example compiles its surrounding add and subtract plus the custom
kernel into one TensorRT engine.

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
            with torch_tensorrt.region_annotations.execute_in_torch():
                x = self.block(x)
            return self.block(x)

An ``execute_in_torch`` region can consume parameters or buffers and return
multiple independent tensors. Noncontiguous inputs are supported. Tensor
constants created with ``torch.tensor`` inside a functional region are
supported too. The body must still be exportable tensor code; the annotation
does not restore Python hooks or side effects removed by export.

Save the compiled model
---------------------------

The legacy exported-program path supports ``execute_in_torch`` regions with
``backend="eager"`` and ``fuse_in_plugin`` regions already built into TensorRT
engines. Save the already chosen placement as follows:

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
for a new optimization pass. Other save modes are not supported for regions.

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
Tensor boundary values, read-only state, and one device per region.

* Models containing live regions and complex tensors are rejected, including
  complex computations outside the region. Complex lowering does not yet
  preserve the region boundary correctly under either complex lowering mode.
* Nested scopes, graph breaks, control flow inside a region, mutation, and
  random operations are rejected. Reassigning a Python variable, such as
  ``x = torch.relu(x)``, is allowed.
* Mutation of model inputs or registered buffers anywhere in a model with live
  regions is rejected, including outside the annotation. Mutation of local
  temporaries outside a region remains supported.
* Region outputs must not share storage with inputs or with each other. Clone
  escaping views or ``chunk`` results inside the scope to give the outputs
  separate storage.
* Disable CPU weight offload, autocast, resource partitioning,
  refit, and weight streaming. Cross-compilation and engine-only output are not
  supported. Conflicting dry runs return a reference graph and report conflicts
  without building engines or offloading state.
  ``require_full_compilation=True`` is incompatible with ``execute_in_torch``
  regions because they must remain outside TensorRT.
* Empty or dead pure scopes are no-ops. Dynamic shapes and training are outside
  the initial scope.

``execute_in_torch`` accepts the optional ``name`` and ``backend`` keywords.
``fuse_in_plugin`` accepts either a kernel factory and launch configuration or
``generator=KernelGenerator(...)`` for compilation-time kernel generation.
``SearchPolicy`` controls candidate batches, launch tuning, search budgets,
resume, and final validation alternatives. See the
:ref:`agent-generated kernel example <agentic_region_kernel_generation_example>`
and :ref:`region annotation API <torch_tensorrt_region_annotations_py>` for
configuration and provider requirements.

Inspect placement and measure performance
---------------------------------------------

For compiler development, inspect ``compiled.graph`` and the internal manifest
returned by ``torch_tensorrt.dynamo.region_compilation.get_region_records(compiled)``.
For ``execute_in_torch``, capture extracts each region into a child FX graph,
keeps it through decomposition, and makes it an opaque PyTorch call for
partitioning. Audits check that graph rewrites preserve the child and keep it
outside TensorRT partitions. No capture marker executes at runtime.

Each ``execute_in_torch`` region limits TensorRT fusion and can introduce extra
engine calls around a small block. Plugin regions keep the supplied kernel
inside TensorRT, but do not guarantee a speedup. Measure full-model latency and
engine counts against the same model without annotations. Fewer or larger
scopes may be faster; the annotation itself does not promise a performance
improvement.
