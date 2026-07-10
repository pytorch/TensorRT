.. _conversion:

Conversion Phase
==================

The conversion phase translates each TensorRT-targeted subgraph (an ``torch.fx.Graph``
or, in the legacy path, a TorchScript block) into a TensorRT ``INetworkDefinition``
which is then compiled into a serializable ``ICudaEngine``.

Dynamo Conversion (Primary Path)
----------------------------------

The Dynamo conversion phase uses a **FX interpreter** — a class that walks the nodes of
a ``torch.fx.Graph`` in topological order, resolves each node's inputs, and calls the
appropriate converter to append TensorRT layers to the ``INetworkDefinition``.

The interpreter maintains a ``ConversionContext`` which holds:

* The in-progress ``trt.INetworkDefinition``
* Compilation settings (precision, device, etc.)
* A map from FX node outputs to ``trt.ITensor`` objects (for tensor values)
* A map from FX node outputs to static ``torch.Tensor`` / scalar values (for frozen
  weights and compile-time constants)
* A **refit map** — a lookup from original PyTorch parameter names to their TensorRT
  layer counterparts, used later for efficient weight refit

Node Inputs
^^^^^^^^^^^

Each node's inputs can be in one of three states:

* **ITensor** — output of a previously converted node. Retrieved from the tensor map
  and passed directly to the converter as a ``trt.ITensor``.
* **Static tensor** — a frozen model weight (``get_attr`` node). Passed as a
  ``torch.Tensor`` or ``np.ndarray``.
* **Scalar / compile-time constant** — an integer, float, or similar. Evaluated at
  conversion time and passed as a Python value.

Shape Propagation
^^^^^^^^^^^^^^^^^^

TensorRT requires shape ranges for every input to the network. Torch-TensorRT derives
these from the user-provided input spec (min/opt/max shapes) combined with the
`SymPy symbolic shape expressions <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html#symbolic-shapes>`_
propagated by TorchDynamo through the graph. This means intermediate tensors entering
a TRT subgraph from a PyTorch subgraph get their shape ranges computed analytically —
no re-execution of the model is needed.

Engine Caching
^^^^^^^^^^^^^^^

Before building, the subgraph content is hashed and checked against a persistent engine
cache. A cache hit allows reuse of the previously compiled engine (refit-only if weights
changed, skipping the expensive TRT optimization pass). As of TensorRT 10.14, cached
engines are stored without weights (weightless engines) to reduce memory footprint when
multiple weight variants share the same architecture.

Converters
^^^^^^^^^^^

Each converter is a Python function decorated with
``@torch_tensorrt.dynamo.conversion.dynamo_tensorrt_converter``. It receives:

* ``ctx`` — the ``ConversionContext`` (primarily ``ctx.net``, the ``INetworkDefinition``)
* ``target`` — the ``torch.ops`` overload being converted
* ``args`` / ``kwargs`` — the resolved inputs (mix of ``trt.ITensor``, ``torch.Tensor``, scalars)
* ``name`` — a string name for the output layers

The converter returns one or more ``trt.ITensor`` objects which are registered back into
the interpreter's tensor map for use by subsequent nodes.

Converter capabilities are declared at registration time:

* **``capability_validator``** — a lambda run at *partition* time (before conversion)
  against each node. If it returns ``False`` the node is routed to PyTorch instead.
  This is how dynamic-shape unsupported ops are gracefully handled.
* **``supports_dynamic_shapes``** — whether the converter handles symbolic dimensions.
* **``requires_output_allocator``** — set to ``True`` for data-dependent-shape (DDS)
  ops that need TensorRT's output allocator at runtime.
* **``priority``** — allows user converters to override built-in ones.

The ``torch_tensorrt.dynamo.conversion.impl`` subpackage contains building-block
implementations (elementwise ops, activations, normalizations, etc.) that can be
composed to build converters without writing raw TensorRT API calls.

For a full guide, see :ref:`dynamo_converters`.

Example
^^^^^^^^

.. code-block:: python

    @torch_tensorrt.dynamo.conversion.dynamo_tensorrt_converter(
        torch.ops.aten.gelu.default,
        capability_validator=lambda node, settings: (
            node.kwargs.get("approximate") == "tanh"
        ),
        supports_dynamic_shapes=True,
    )
    def aten_ops_gelu(ctx, target, args, kwargs, name):
        # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = args[0]
        mul  = lambda a, b: impl.elementwise.mul(ctx, target, name=name, source_ir=SourceIR.ATEN, lhs_val=a, rhs_val=b)
        add  = lambda a, b: impl.elementwise.add(ctx, target, name=name, source_ir=SourceIR.ATEN, lhs_val=a, rhs_val=b)
        tanh = lambda a:    impl.activation.tanh(ctx, target, name=name, source_ir=SourceIR.ATEN, input_val=a)

        x7  = mul(x, 0.5)
        x8  = mul(x, 0.79788456)
        x9  = mul(x, 0.044715)
        x10 = mul(x9, x)
        x11 = add(x10, 1.0)
        x12 = mul(x8, x11)
        x13 = tanh(x12)
        x14 = add(x13, 1.0)
        return mul(x7, x14)

----

TorchScript Conversion (Legacy ``ts`` Path)
---------------------------------------------

.. note::

   The following describes the legacy TorchScript conversion path. For new
   development use the Dynamo path above.

Once the TorchScript graph has been simplified by the lowering phase, a conversion
context is created to manage construction of a TensorRT ``INetworkDefinition`` from the
graph blocks. The conversion context records converted nodes, block inputs/outputs, and
build-time weights.

The block converter iterates over nodes and assembles inputs in one of four states:

* **Block parameter** — stored as an ``IValue`` in ``evaluated_value_map``; passed
  directly to the converter.
* **Output of a previously converted node** — ``ITensor`` retrieved from
  ``value_tensor_map``.
* **Static value node** (e.g. ``prim::Constant``, ``prim::ListConstruct``) — evaluated
  at conversion time via the evaluator registry; result stored in ``evaluated_value_map``.
* **Unconverted node** — Torch-TensorRT errors out.

Node Evaluators
^^^^^^^^^^^^^^^^

Some nodes produce static data (parameters for operators) that can be resolved at
compile time. Any node kind can have a compile-time evaluator as long as it produces a
static ``IValue``. Common types are ``prim::Constant`` and ``prim::ListConstruct``.

Node Converters
^^^^^^^^^^^^^^^^

Node converters map TorchScript nodes to TensorRT layers, associating TRT outputs back
to TorchScript graph values in the conversion context so downstream nodes can consume
them. For more information see :ref:`writing_converters`.
