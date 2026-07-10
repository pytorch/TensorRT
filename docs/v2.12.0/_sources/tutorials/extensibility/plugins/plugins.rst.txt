.. _plugins:

Plugin System
=============

Torch-TensorRT's plugin system lets you run custom kernels *inside* a TensorRT engine,
avoiding graph breaks and their associated overhead. There are three main approaches
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
   * - QDP auto-generate (AOT)
     - CUDA C++ via NVRTC
     - Pre-compiled PTX embedded in engine
     - :ref:`nvrtc_aot_plugin`
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

For complete end-to-end examples see:

* :ref:`auto_generate_plugins` — Triton kernel, QDP JIT plugin
* :ref:`aot_plugin` — Triton kernel, QDP AOT plugin (pre-compiled PTX, no Python overhead at runtime)
* :ref:`nvrtc_aot_plugin` — CUDA C++ kernel compiled with NVRTC, QDP AOT plugin
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
