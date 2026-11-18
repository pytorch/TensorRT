.. _lowering:

Lowering Phase
===============

The lowering phase maps the captured FX graph from its original opset down
to a form that the conversion stage can translate 1-to-1 into TensorRT operations. The
goal is to reduce the converter library to a small set of simple, well-defined
translations rather than requiring each converter to handle complex multi-op patterns.

Dynamo Lowering (Primary Path)
--------------------------------

The Dynamo lowering phase operates on ``torch.fx.Graph`` objects and has two mechanisms:

Decompositions
^^^^^^^^^^^^^^^

The primary mechanism, covering roughly 80% of lowering work. A higher-level ATen
operator is replaced inline by an equivalent subgraph of
`Core ATen operators <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_ir.html>`_
using a plain PyTorch function. This is easy to write because the body uses normal
PyTorch code and is automatically shape-propagating via
`FakeTensor <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_fake_tensor.html>`_.

.. code-block:: python

    @register_torch_trt_decomposition(aten.linear, registry=TORCH_TRT_DECOMPOSITIONS)
    def linear_decomposition(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        return torch.add(torch.ops.aten.mm(x, w.T), b)

Decompositions are registered per-op and are applied automatically during the lowering
pass over the graph. You can also adjust which PyTorch-builtin decompositions are
enabled or disabled via ``torch_tensorrt.dynamo.lowering.torch_enabled_decompositions``
and ``torch_tensorrt.dynamo.lowering.torch_disabled_decompositions``.

Subgraph Matching and Replacement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used for structural rewrites that cannot be expressed as a simple per-op substitution.
Examples include:

* Inserting KV-cache operators for attention optimizations in LLM models
* Decomposing complex-number arithmetic into real/imaginary components (since TensorRT
  has no complex dtype support)
* Removing vestigial subgraphs (e.g. training-only operations that survive export)

These passes use
`torch.fx.subgraph_rewriter <https://docs.pytorch.org/docs/stable/fx.html#graph-manipulation>`_
or manual graph surgery and are registered into the pass manager alongside decompositions.

For a guide on writing new Dynamo lowering passes, see :ref:`writing_dynamo_aten_lowering_passes`.

.. note::

   After each lowering pass the graph must remain valid and callable. Use
   ``gm.graph.lint()`` and ``gm.recompile()`` after any structural changes.

----
