.. _converter_registry:

Converter Registry Internals
==============================

This page covers the internals of ``DYNAMO_CONVERTERS`` — the global registry that
maps every ``torch.ops`` target to its TensorRT converter — and explains the supporting
types and lookup algorithm. For a guide on *writing* converters see
:ref:`dynamo_converters`.

----

Key Types
---------

ConverterSupport
^^^^^^^^^^^^^^^^

Every converter registered via ``@dynamo_tensorrt_converter`` is stored as a
``ConverterSupport`` frozen dataclass:

.. code-block:: python

    @dataclass(frozen=True)
    class ConverterSupport:
        converter_implementation: ConverterImplSignature
        capability_validator: Callable[[Node, CompilationSettings], bool] = lambda node, settings: True
        supports_dynamic_shapes: bool = False
        requires_output_allocator: bool = False

``converter_implementation``
    The actual converter function. See :ref:`dynamo_converters` for the expected
    signature.

``capability_validator``
    Called at *partition* time (before conversion) with the live ``torch.fx.Node`` and
    active ``CompilationSettings``. Must return ``True`` if this converter can handle
    this specific node. The default always returns ``True`` (unconditional support). See
    :ref:`partitioning` for how validators interact with the partitioner.

``supports_dynamic_shapes``
    If ``False`` (default), the registry will not select this converter when the node
    has symbolic (dynamic) input dimensions — unless
    ``assume_dynamic_shape_support=True`` in settings.

``requires_output_allocator``
    Marks converters whose TRT implementation produces data-dependent output shapes
    (e.g. ``nonzero``, ``unique``). The runtime will use TensorRT's output allocator
    for these ops rather than pre-allocating fixed buffers.

ConverterPriority
^^^^^^^^^^^^^^^^^

.. code-block:: python

    class ConverterPriority(Enum):
        STANDARD = auto()
        HIGH = auto()

``HIGH`` priority converters are inserted at the **front** of the candidate list for
their target. When the registry iterates candidates for a node it picks the first one
whose ``capability_validator`` returns ``True`` — so ``HIGH`` converters are checked
before ``STANDARD`` ones. Use ``HIGH`` to override a built-in converter:

.. code-block:: python

    @dynamo_tensorrt_converter(
        torch.ops.aten.gelu.default,
        priority=ConverterPriority.HIGH,
        capability_validator=lambda node, settings: node.kwargs.get("approximate") == "tanh",
    )
    def my_fast_gelu(ctx, target, args, kwargs, name):
        ...

CallingConvention
^^^^^^^^^^^^^^^^^

.. code-block:: python

    class CallingConvention(Enum):
        LEGACY = auto()   # Old-style FX converters: (net, target, args, kwargs, name)
        CTX    = auto()   # Dynamo converters:       (ctx, target, args, kwargs, name)

All newly written converters use ``CTX``. ``LEGACY`` is retained only for backward
compatibility with old FX converter dictionaries.

----

``@dynamo_tensorrt_converter`` Decorator
------------------------------------------

.. code-block:: python

    @dynamo_tensorrt_converter(
        key,
        *,
        enabled=True,
        capability_validator=None,
        priority=ConverterPriority.STANDARD,
        supports_dynamic_shapes=False,
        requires_output_allocator=False,
    )

``key`` (``Target``)
    The ``torch.ops`` overload to register for. Must be an ``OpOverload`` (e.g.
    ``torch.ops.aten.relu.default``), not an ``OpOverloadPacket`` (e.g.
    ``torch.ops.aten.relu``), unless the packet has only one or two overloads
    (``"default"`` + ``"out"``).

``enabled`` (``bool``, default ``True``)
    If ``False``, the decorator is a no-op — the function is returned unchanged and
    **not** registered. Useful for gating experimental converters behind a flag:

    .. code-block:: python

        @dynamo_tensorrt_converter(torch.ops.mylib.op.default, enabled=EXPERIMENTAL)
        def my_converter(...): ...

``capability_validator`` (``Callable[[Node, CompilationSettings], bool]``, default ``None``)
    Validated at partition time. ``None`` is equivalent to ``lambda node, settings: True``.
    Must be a **pure function** that does not modify the node or graph.

``priority`` (``ConverterPriority``, default ``STANDARD``)
    Determines insertion position in the candidate list for this target. ``HIGH``
    converters are tried first.

``supports_dynamic_shapes`` (``bool``, default ``False``)
    Set to ``True`` only after verifying the converter handles symbolic dimensions
    correctly (e.g., using ``ctx.net.add_shape`` rather than hardcoded sizes).

``requires_output_allocator`` (``bool``, default ``False``)
    Set to ``True`` for converters implementing data-dependent-shape ops.

----

The ``DYNAMO_CONVERTERS`` Registry Object
------------------------------------------

``DYNAMO_CONVERTERS`` is the singleton ``ConverterRegistry`` instance the interpreter
queries for every ``call_function`` node:

.. code-block:: python

    from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS

The registry wraps a list of converter dictionaries (currently one — ``DYNAMO_ATEN_CONVERTERS``)
and provides these access patterns:

``DYNAMO_CONVERTERS[node]`` — validated lookup
    Pass a ``torch.fx.Node``. Returns ``(converter_impl, CallingConvention, flags_dict)``
    where ``flags_dict`` has keys ``"supports_dynamic_shapes"`` and
    ``"requires_output_allocator"``. Raises ``KeyError`` if no validated converter is
    found. This is the path the interpreter uses.

``DYNAMO_CONVERTERS.get(node, default=None)``
    Same as ``__getitem__`` but returns ``default`` instead of raising on a miss.

``target in DYNAMO_CONVERTERS``
    Checks for a *validated* entry. Pass a ``Node`` for validated check or a ``Target``
    for unvalidated existence check.

Registry Inspection
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS

    # Print a full table of all registered targets and their source registries
    print(DYNAMO_CONVERTERS.display_all_available_converters())

    # Get the table as a dict: {qualified_op_name: {registry_name: count}}
    support_info = DYNAMO_CONVERTERS.get_converter_support_info()

    # Check whether a specific op has a validated converter for a node:
    from torch_tensorrt.dynamo.partitioning import get_graph_converter_support
    n_supported, n_total = get_graph_converter_support(gm, torch_executed_ops=set())
    print(f"{n_supported}/{n_total} ops have TRT converters")

    # List all unique registered targets:
    all_targets = DYNAMO_CONVERTERS.unique_targets()

    # Get all converters (including lower-priority ones) for a target:
    impls, registry_info = DYNAMO_CONVERTERS.get_all_converters_with_target(
        torch.ops.aten.relu.default, return_registry_info=True
    )

Lookup Algorithm
^^^^^^^^^^^^^^^^^

When the interpreter calls ``DYNAMO_CONVERTERS[node]``:

1. Check if ``node.target`` is in ``disallowed_targets`` (i.e., ``torch_executed_ops``).
   If yes, raise ``KeyError`` — node falls back to PyTorch.
2. Iterate registries in order (currently only ``DYNAMO_ATEN_CONVERTERS``).
3. For each registry containing the target, iterate its ``ConverterSupport`` list in
   order (HIGH-priority entries first).
4. For each candidate, evaluate:

   * ``capability_validator(node, compilation_settings)`` → must be ``True``
   * If node has symbolic dims: ``assume_dynamic_shape_support`` must be ``True`` OR
     ``candidate.supports_dynamic_shapes`` must be ``True``

5. Return the first passing candidate together with its ``CallingConvention`` and flags.
6. If no candidate passes, raise ``KeyError`` — node falls back to PyTorch.

----

ConversionContext
-----------------

Every converter receives a ``ConversionContext`` as its first argument (``ctx``):

.. code-block:: python

    @dataclass
    class ConversionContext:
        net: trt.INetworkDefinition      # The TRT network being built
        compilation_settings: CompilationSettings
        requires_output_allocator: bool  # Set True if any converter in the graph needs it
        weight_refit_map: dict[str, torch.Tensor]   # name → weight for refit
        cpu_weights_reference_holder: list[torch.Tensor]

``ctx.net``
    The ``trt.INetworkDefinition``. Converters call methods like
    ``ctx.net.add_elementwise()``, ``ctx.net.add_activation()``, etc. to add TRT layers.

``ctx.compilation_settings``
    Full ``CompilationSettings`` object. Use this to read user preferences inside a converter.

``ctx.record_weight(name, weight)``
    Call this from a converter whenever you add a constant tensor to the TRT network.
    It populates ``weight_refit_map`` (used by :func:`~torch_tensorrt.dynamo.refit_module_weights`)
    so the weight can be updated without recompilation. The name must match the TRT
    layer's weight name as it will appear in the engine.

    .. code-block:: python

        weight_tensor = get_trt_tensor(ctx, weight, f"{name}_weight")
        ctx.record_weight(f"{name}_weight", weight)   # register for refit

``ctx.clear_cpu_weights_reference_holder()``
    Called automatically after engine serialization. Releases the CPU-side references
    to weight tensors that were held alive during the build phase.
