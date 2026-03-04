Compiler Phases
----------------

.. toctree::
    :caption: Compiler Phases
    :maxdepth: 1
    :hidden:

    lowering
    partitioning
    conversion
    runtime

Lowering
^^^^^^^^^^^
:ref:`lowering`

The lowering phase maps the captured FX graph from its original opset down to the
subset that TensorRT can work with. There are two mechanisms:

* **Decompositions** — the primary mechanism (~80% of lowering). A higher-level ATen
  op is replaced by an equivalent subgraph of
  `Core ATen ops <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_ir.html>`_
  using a plain PyTorch function registered with ``@register_torch_trt_decomposition``.
  For example, ``aten.log_softmax`` decomposes to ``aten.softmax`` + ``aten.log``.

* **Subgraph matching and replacement** — for more complex structural rewrites such as
  inserting attention optimizations (KV-caching), decomposing complex-number arithmetic,
  or removing vestigial subgraphs that TensorRT cannot handle.

Passes are run in order via the pass manager on the ``torch.fx.Graph`` and produce a
new (lowered) ``torch.fx.Graph``.

Partitioning
^^^^^^^^^^^^^
:ref:`partitioning`

The partitioner splits the lowered FX graph into subgraphs that will run in TensorRT and
subgraphs that will remain in PyTorch. The split is based on converter availability and
explicit user overrides (``torch_executed_ops``, ``torch_executed_modules``,
``min_block_size``).

Two strategies are available:

* **Adjacency partitioning** (default) — fast best-effort traversal that merges adjacent
  TensorRT-capable ops. Good enough for the vast majority of models.
* **Global partitioning** — guarantees the minimum number of subgraphs, important for
  models where context-switch overhead between PyTorch and TensorRT is significant.
  Takes longer at compile time.

An experimental **hierarchical / multi-backend partitioner** also exists for assigning
subgraphs to different backends (TensorRT, MLIR-TRT, TorchInductor, etc.) by priority
order.

Conversion
^^^^^^^^^^^
:ref:`conversion`

For each TensorRT subgraph produced by the partitioner, an FX interpreter traverses the
graph node-by-node and builds a TensorRT ``INetworkDefinition``. Each node is handled
by a registered **converter** — a function that receives the current
``INetworkDefinition`` plus the node's arguments (either ``trt.ITensor`` objects from
earlier nodes, frozen ``torch.Tensor`` weights, or scalar constants) and appends the
appropriate TensorRT layers.

Intermediate tensor shape ranges are derived from user-provided input shape specs
combined with the
`symbolic shape (SymPy) expressions <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html#symbolic-shapes>`_
propagated by TorchDynamo — no re-execution of the model is needed.

Before building, the subgraph is checked against an engine cache keyed on the graph
hash. A cache hit skips building (weight-only refit is performed instead if weights
changed). As of TensorRT 10.14, cached engines are stored weightless to reduce memory.

A **refit map** — a lookup table from original PyTorch parameter names to their
corresponding TensorRT layer indices — is also constructed during conversion to
support efficient weight refit later (e.g. for LoRA adapters via
``MutableTorchTensorRTModule``).

Runtime and Module Wrapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:ref:`runtime`

Once all TensorRT subgraphs have been compiled to engines, the compiler wraps them
together with the remaining PyTorch subgraphs into a single callable module
(``TorchTRTModule``).

Two runtime backends are available:

* **C++ runtime** (default) — more performant and serializable. TensorRT engines are
  stored as ``torch.classes.tensorrt.Engine`` custom objects and executed via the
  ``torch.ops.tensorrt.execute_engine`` custom op. Supports CUDAGraphs and
  multi-device safety.

* **Python runtime** — a pure-Python execution path using TensorRT's Python API.
  Useful for environments where a C++ build is not available, and easier to
  instrument for debugging.
