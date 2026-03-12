.. _subgraph_builder:

SubgraphBuilder — Cursor-Based FX Node Insertion
=================================================

Writing lowering passes that replace one node with several new nodes requires
careful management of insertion order: each new node must be inserted
*after the previous one* so that the topological ordering of the graph is
preserved.  Doing this by hand with repeated ``graph.inserting_after(cursor)``
context managers is verbose and error-prone.

``SubgraphBuilder`` is a lightweight context-manager helper in
``torch_tensorrt.dynamo.lowering._SubgraphBuilder`` that automates this
cursor-tracking pattern.

Basic Usage
-----------

Construct a ``SubgraphBuilder`` with the target graph and the *anchor* node —
the node immediately before where you want to start inserting.  Then use it
as a callable inside a ``with`` block to add nodes one at a time:

.. code-block:: python

    from torch_tensorrt.dynamo.lowering._SubgraphBuilder import SubgraphBuilder
    import torch.ops.aten as aten

    # Inside a lowering pass, given a node `mul_node` to replace:
    with SubgraphBuilder(gm.graph, mul_node) as b:
        # Each call inserts a node after the current cursor and advances it.
        re_a = b(aten.select.int, a, -1, 0)   # a[..., 0]  (real part of a)
        im_a = b(aten.select.int, a, -1, 1)   # a[..., 1]  (imag part of a)
        re_b = b(aten.select.int, b_node, -1, 0)
        im_b = b(aten.select.int, b_node, -1, 1)
        real = b(aten.sub.Tensor, b(aten.mul.Tensor, re_a, re_b),
                                   b(aten.mul.Tensor, im_a, im_b))  # ac - bd
        imag = b(aten.add.Tensor, b(aten.mul.Tensor, re_a, im_b),
                                   b(aten.mul.Tensor, im_a, re_b))  # ad + bc
        result = b(aten.stack, [real, imag], -1)

    mul_node.replace_all_uses_with(result)
    gm.graph.erase_node(mul_node)

On ``__exit__``, the builder automatically calls ``graph.lint()`` to validate
the modified graph.  If your code raises an exception inside the block, the
lint is skipped so you see the original error rather than a secondary graph
validation failure.

How It Works
------------

The builder maintains a *cursor* — initially the anchor node passed to
``__init__``.  Every time you call it:

1. A new ``call_function`` node is inserted via ``graph.inserting_after(cursor)``.
2. The cursor advances to the newly inserted node.
3. The new node is appended to an internal ``_inserted`` list for debug logging.

This ensures that successive calls produce a correctly ordered chain:

.. code-block:: text

    anchor → node_0 → node_1 → node_2 → ...

without any manual bookkeeping.

Debug Logging
-------------

When the ``torch_tensorrt`` logger is set to ``DEBUG``, the builder emits a
compact summary of all inserted nodes after a successful block, for example::

    rewrite  %mul_17[(4, 32, 2),torch.float32]  ->
      %select_72[(4, 32),torch.float32] = select_int(%inp_0, -1, 0)
      %select_73[(4, 32),torch.float32] = select_int(%inp_0, -1, 1)
      %mul_18[(4, 32),torch.float32]    = mul_Tensor(%select_72, %select_73)
      ...

This makes it easy to trace exactly which nodes were produced by a particular
rewrite rule.

API Reference
-------------

.. autoclass:: torch_tensorrt.dynamo.lowering._SubgraphBuilder.SubgraphBuilder
   :members:
   :undoc-members:

When to Use SubgraphBuilder
---------------------------

Use ``SubgraphBuilder`` whenever a lowering pass needs to **expand one node into
a sequence of several nodes** in a single linear chain.  Typical use cases:

* Replacing a complex-arithmetic op with real-arithmetic equivalents
  (e.g. the ``complex_mul_replacement`` in :ref:`complex_number_support_design`).
* Decomposing a high-level op (e.g. ``layer_norm``) into its ATen primitives
  when a custom replacement strategy is needed beyond the standard decomposition
  table.
* Inserting diagnostic nodes (shape probes, debug prints) around a target op.

If you only need to insert a *single* node, a plain
``graph.inserting_after(node)`` is simpler.  If you need to insert into multiple
disconnected locations in the same pass, create a separate ``SubgraphBuilder``
for each anchor.
