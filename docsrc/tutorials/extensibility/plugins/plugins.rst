.. _plugins:

Plugin System
=============

Torch-TensorRT's plugin system lets you run custom kernels *inside* a TensorRT engine,
avoiding graph breaks and their associated overhead. There are five main approaches
depending on your kernel language and performance requirements:

.. list-table::
   :widths: 20 25 15 40
   :header-rows: 1

   * - Approach
     - Kernel language
     - Execution
     - Example
   * - QDP auto-generate (JIT)
     - Triton
     - JIT callback into Python at runtime
     - :ref:`auto_generate_plugins`
   * - QDP auto-generate (AOT)
     - Triton
     - Pre-compiled PTX embedded in engine
     - :ref:`aot_plugin`
   * - QDP kernels API (AOT/JIT)
     - CUDA C++ / Triton / PTX
     - Embedded PTX, or JIT for CUDA scalar attributes
     - :ref:`cuda_kernel_op`, :ref:`ptx_op`, :ref:`triton_op`
   * - QDP auto-generate (AOT)
     - CUDA C++ via NVRTC
     - Pre-compiled PTX embedded in engine
     - :ref:`nvrtc_aot_plugin`
   * - QDP declarative (AOT)
     - cuTile
     - Pre-compiled PTX embedded in engine
     - :ref:`cutile_op`
   * - Manual (legacy)
     - Triton / any
     - JIT callback into Python at runtime
     - :ref:`custom_kernel_plugins`

The **QDP (Quick Deployable Plugin)** path (TensorRT ≥ 10.7) is the recommended
approach. It uses ``torch.library`` to register your custom op and
``_generate_plugin_converter`` to automatically create the Torch-TensorRT converter.
The **manual** path requires writing both the TRT plugin and the converter by hand,
and is retained for compatibility with older workflows.

The flow for the QDP path is:

1. Register a custom op with ``torch.library`` and implement it as a TRT QDP plugin.
2. Call ``_generate_plugin_converter`` to automatically create a Torch-TensorRT
   converter that bridges the two.
3. Use the custom op in a PyTorch model and compile normally with
   ``torch_tensorrt.dynamo.compile``.

----

Prerequisites
-------------

* TensorRT ≥ 10.7 (``tensorrt.plugin`` module must be importable).
* A registered QDP plugin in ``tensorrt.plugin``'s ``QDP_REGISTRY``.
* A corresponding ``torch.ops`` custom op.

----

Registering a Plugin Converter
--------------------------------

``_generate_plugin_converter`` creates and registers a converter for your custom op
automatically:

.. code-block:: python

    from torch_tensorrt.dynamo.conversion.plugins import _generate_plugin_converter

    _generate_plugin_converter(
        namespace="mylib",
        op_name="my_custom_op",
        overload=None,               # None → "default" overload
        supports_dynamic_shapes=True,
        use_aot_if_available=True,   # prefer AOT plugin if registered
    )

This registers a converter for ``torch.ops.mylib.my_custom_op.default`` in
``DYNAMO_CONVERTERS``. The generated converter:

1. Looks up the QDP plugin object via ``trtp.op.<namespace>.<op_name>``.
2. Converts all tensor inputs to ``trt.ITensor`` using ``get_trt_tensor``.
3. Passes non-tensor arguments (scalars, booleans, etc.) as plugin attributes,
   preserving the order from the op's Torch schema.
4. Adds the plugin layer to ``ctx.net`` and returns its output ITensors.

Parameters
^^^^^^^^^^^

``namespace`` / ``op_name``
    The Torch Library namespace and operator name. The plugin must be registered in
    TRT's registry as ``{namespace}::{op_name}``.

``overload``
    The overload string (e.g., ``"Tensor"``) or ``None`` for the ``default`` overload.

``capability_validator``
    Optional ``(Node, CompilationSettings) -> bool`` function. Same semantics as
    the standard ``@dynamo_tensorrt_converter`` decorator.

``priority``
    ``ConverterPriority.STANDARD`` or ``HIGH``. Use ``HIGH`` to override an existing
    converter.

``supports_dynamic_shapes``
    Set ``True`` if the QDP plugin supports symbolic input dimensions.

``requires_output_allocator``
    Set ``True`` if the plugin produces data-dependent output shapes.

``use_aot_if_available``
    If ``True`` (default), use the plugin's ahead-of-time (AOT) compiled
    implementation when one is registered (``desc.aot_impl_func is not None``).
    Falls back to JIT plugin if the AOT impl is absent.

----

Mutating Custom Operators
------------------------

Declare every modified tensor in ``mutates_args``. Operators that return
``None`` can use the usual fake implementation:

.. code-block:: python

    import torch
    import torch_tensorrt

    @torch.library.custom_op("example::state_add", mutates_args=("state",))
    def state_add(state: torch.Tensor, update: torch.Tensor) -> None:
        state.add_(update)

    @state_add.register_fake
    def _(state, update):
        return None

    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "example::state_add", supports_dynamic_shapes=True
    )

    class Model(torch.nn.Module):
        def forward(self, state, update):
            state_add(state, update)
            return state * 2

The generated plugin has one synthetic aliased output per mutated tensor, in
tensor-argument order. Torch-TensorRT preserves these outputs even when the
model returns ``None``, and binds outputs that update engine inputs to those
inputs' storage. Hidden mutation outputs are excluded from the model's return
value. Static shapes, dynamic profiles, and multiple mutated base tensors are
supported.

For tensor-returning mutating operators, the fake implementation must return
each mutated input by object identity. The real implementation must return
non-aliasing tensors, as required by PyTorch custom operators. For example,
the fake kernel can return ``state`` while the real kernel returns
``state.clone()``. Missing alias signals produce a descriptive error.

Mutations through views and tensor-list arguments remain in PyTorch. Consuming
an aliased output from a plugin with multiple outputs also remains in PyTorch
because TensorRT may insert copies around the plugin. These restrictions do
not prevent returning multiple mutated base tensors from the model.

For a manually registered QDP plugin corresponding to a ``None``-returning
operator, return ``input_desc.aliased()`` for each mutated tensor from the
descriptor function, in tensor-argument order. Then register the converter with
``generate_plugin_converter("example::state_add", supports_dynamic_shapes=True)``.
An AOT implementation must write the corresponding buffers directly. Automatic
``custom_op`` registration generates a JIT implementation; standalone deployment
without Python requires an AOT implementation whose kernels are embedded in the
engine.

Standalone engine callers must retain the input/output alias mapping and bind
both names of each pair to the same device pointer. A compiled
``TorchTensorRTModule`` exposes the mapping as ``aliased_io``; its serialized
state also preserves the mapping and the user output count. Hidden bindings
have names beginning with ``__torch_tensorrt_mutation_``. TensorRT's
``get_aliased_input_tensor`` does not necessarily report plugin aliases, so
discovering them solely through that API is insufficient.

----

For complete end-to-end examples see:

* :ref:`auto_generate_plugins` — Triton kernel, QDP JIT plugin
* :ref:`aot_plugin` — Triton kernel, QDP AOT plugin (pre-compiled PTX, no Python overhead at runtime)
* :ref:`triton_op` — the same Triton AOT plugin registered in one ``triton_op`` call
* :ref:`nvrtc_aot_plugin` — CUDA C++ kernel compiled with NVRTC, QDP AOT plugin
* :ref:`cutile_op` — cuTile kernel, QDP AOT plugin registered in one ``cutile_op`` call
* :ref:`custom_kernel_plugins` — manual plugin + converter registration (legacy approach)

----

Debugging Plugin Converters
-----------------------------

If the converter is not being selected (op falls back to PyTorch):

1. Verify the plugin is in the QDP registry:

   .. code-block:: python

       import tensorrt.plugin as trtp
       from tensorrt.plugin._lib import QDP_REGISTRY
       print("mylib::scaled_add" in QDP_REGISTRY)

2. Verify the converter was registered:

   .. code-block:: python

       from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS
       print(torch.ops.mylib.scaled_add.default in DYNAMO_CONVERTERS)

3. Check the capability validator (if you supplied one) against the actual node in a
   dryrun report (see :ref:`dryrun`).
