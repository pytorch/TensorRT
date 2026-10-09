.. _complex_number_support_design:

Complex Number Support
=======================

.. note::

   This page documents the design for complex number support in Torch-TensorRT.
   Original design discussion:
   `RFC #3456 <https://github.com/pytorch/TensorRT/discussions/3456>`_.

Goal
----

TensorRT does not natively support complex-dtype tensors (``torch.complex64``,
``torch.complex128``). Complex numbers appear in models that use rotary position
embeddings (RoPE), for example in Llama 3, where frequency vectors are computed
in polar form (``torch.polar``) and applied via complex multiplication.

The goal is to allow such models to be compiled end-to-end by Torch-TensorRT
through a graph-rewrite lowering pass that eliminates all complex-dtype nodes
before the graph reaches TensorRT.

The primary motivation was enabling end-to-end compilation of Llama 3 in
distributed (multi-GPU) settings where the ``torch.compile`` + distributed-tensor
workflow hoists ``freqs_cis`` (a complex64 tensor) to a graph input.

Rotary Embedding Pattern
-------------------------

The canonical complex-number subgraph in RoPE looks like:

.. code-block:: python

    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

After export+lowering the critical sub-pattern is:

.. code-block:: text

    placeholder (complex freq) ──► reshape ──► mul (complex) ──► view_as_real
    placeholder (real xq)      ──► view_as_complex ──┘

Implementation Overview
------------------------

The rewrite is a lowering pass in
``py/torch_tensorrt/dynamo/lowering/passes/complex_graph_rewrite.py``.
It operates in three conceptual stages:

.. image:: https://github.com/user-attachments/assets/fafd7353-a7cf-42f1-9849-14a0dbad8de1
   :alt: Complex number support overview diagram

Stage 1 — Detection
^^^^^^^^^^^^^^^^^^^^

The pass anchors on ``view_as_real`` nodes and walks backward through the graph
to identify all nodes participating in complex arithmetic. A node is included in
the complex subgraph if its output dtype is complex or if it is ``view_as_complex``.

The resulting ``ComplexOpSubGraphInfo`` records:

* ``anchor_nodes`` — the ``view_as_real`` nodes that terminate the complex subgraph.
* ``subgraph_nodes`` — all nodes between the inputs and the anchors.
* ``input_nodes`` — nodes feeding into the subgraph from outside.

Stage 2 — Input Node Replacement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each complex input node is replaced with a real-dtype equivalent:

* **``get_attr`` buffers** (constant complex tensors): a new ``_unpacked_complex``
  buffer is registered on the graph module using ``torch.stack([real, imag], dim=-1)``,
  which has dtype ``float32`` and one additional trailing dimension of size 2.
* **``placeholder`` inputs** (runtime complex tensors): the placeholder's metadata
  (``meta["val"]``) is updated to reflect the new ``float32`` shape with the
  appended ``2`` dimension. SymInt dynamic dimensions are preserved.

.. image:: https://github.com/user-attachments/assets/6c2ebe8d-04fd-45a1-9549-da43d0986109
   :alt: Graph rewrite — source pattern (blue arrows = modifications)

Stage 3 — Subgraph Rewrite
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once inputs are real, the complex ops within the subgraph are rewritten:

* **``view_as_complex``** — erased (the input is already real with trailing dim 2).
* **``view_as_real``** — erased (the output is already real).
* **``aten.mul.Tensor`` on complex tensors** — replaced with the manual
  complex-multiplication identity:

  .. math::

      (a + bi)(c + di) = (ac - bd) + (ad + bc)i

  Implemented as:

  .. code-block:: python

      # a, b = real/imag parts of left operand (shape [..., 2])
      # c, d = real/imag parts of right operand (shape [..., 2])
      real = a * c - b * d
      imag = a * d + b * c
      result = torch.stack([real, imag], dim=-1)

* **``permute``** on complex tensors — the dims list is extended by appending the
  original last dimension index so the trailing ``2`` dimension (real/imag) is
  permuted correctly.
* **``reshape``/``slice``** — trailing-dimension arguments are updated to account
  for the new ``...×2`` layout.

.. image:: https://github.com/user-attachments/assets/3b686362-742b-4435-aa78-5ab8db7c55b2
   :alt: Modified target graph after complex rewrite

Runtime Changes
----------------

At runtime the TRT engine receives a real-valued tensor with shape
``(*orig_complex_shape, 2)`` instead of the original complex tensor. The three
runtime modules handle the conversion:

* ``prepare_inputs`` (``dynamo/utils.py``) — builds the ``Input`` spec with the
  ``view_as_real`` shape/dtype but retains the original complex tensor in
  ``inp.torch_tensor`` for tracing.
* ``_PythonTorchTensorRTModule.forward`` — applies ``torch.view_as_real(i).contiguous()``
  for each complex input before feeding it to the engine.
* ``_TorchTensorRTModule.forward`` — same ``view_as_real`` conversion.

Key Implementation Invariants
-------------------------------

* **``node.meta["is_complex_layout"]``** — every node that represents a complex
  quantity (either originally complex-dtype, or a real ``(..., 2)`` tensor produced
  by the rewriter) is annotated with ``node.meta["is_complex_layout"] = True``.
  This annotation is set during the detection phase (before any rewrites begin) and
  propagated by every rewrite handler as it emits new nodes.  It survives dtype
  changes: after ``replace_input_node`` converts a ``placeholder`` from complex to
  ``float32``, the dtype-based check ``is_complex_dtype()`` would return ``False``,
  but the metadata flag remains.  ``_is_complex_layout_node(n)`` is simply
  ``n.meta.get("is_complex_layout", False)`` — no shape heuristics or recursion.
* **FakeTensorMode reuse** — ``propagate_metadata`` must use the ``FakeTensorMode``
  from existing placeholder fake tensors (not a fresh mode) to avoid mode-mismatch
  errors under ``torch.compile`` and to preserve SymInt for dynamic shapes.
* **Dotted buffer names** — ``register_buffer`` rejects names containing ``.``.
  Nested submodule parameter names (e.g. ``layers.0.weight``) must have ``.``
  replaced with ``__`` before registration.

The Decomposition System — How It Is Built
-------------------------------------------

The rewriter is split across two classes and wired together by a lightweight
dispatch mechanism.  This section walks through each piece and explains the
design decisions.

ComplexOpDetector — Subgraph Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ComplexOpDetector`` walks the graph to find the set of nodes that participate
in complex arithmetic.

``node_include_in_subgraph``
""""""""""""""""""""""""""""

A node is included in a complex subgraph if:

1. Its output dtype is ``complex64`` or ``complex128`` (``is_complex_dtype``), **or**
2. Any of its inputs are complex (``has_complex_input``).

The second condition is necessary to catch real-output ops — ``abs``, ``angle``,
``real``, ``imag`` — whose inputs are complex.  These must be rewritten alongside
the rest of the subgraph even though their outputs are real.

``subgraph_from_anchor``
""""""""""""""""""""""""

For ``view_as_real``-bounded subgraphs, detection starts at a ``view_as_real``
*anchor* node and performs a backward BFS:

.. code-block:: text

    view_as_real  ←  mul (complex)  ←  reshape  ←  placeholder (complex)
         ↑ anchor         ↑ subgraph              ↑ subgraph     ↑ input

At each step, if an upstream node satisfies ``node_include_in_subgraph`` it is
added to the subgraph; otherwise it becomes an *input node* (the boundary).  The
result is a ``ComplexSubGraphInfo`` containing anchor nodes, subgraph nodes, and
input nodes.

After collection the subgraph is **sorted in topological order** (by position in
the graph's node list).  This is critical: without it a ``mul`` node could be
processed before its ``sin`` or ``cos`` operands, causing the rewriter to see the
original complex node instead of the already-rewritten real node.

``find_complex_op_subgraphs`` and subgraph merging
"""""""""""""""""""""""""""""""""""""""""""""""""""

When a model has multiple ``view_as_real`` anchors that share upstream nodes
(e.g. ``xq_out`` and ``xk_out`` in a RoPE layer both descend from the same
``freqs_cis`` placeholder), their subgraphs would otherwise be detected
separately.  ``find_complex_op_subgraphs`` merges overlapping subgraphs by
set intersection so each node is rewritten exactly once.

``find_all_complex_subgraphs`` — unbounded complex ops
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

Some models produce a complex tensor as a graph *output* without passing it
through ``view_as_real``.  ``find_all_complex_subgraphs`` is a forward scan that
collects every ``call_function`` node with a complex output, regardless of
anchoring.  The resulting subgraph is processed the same way as an
anchor-bounded one.

ComplexGraphRewriter — Dispatch-Based Rewriting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ComplexGraphRewriter`` is decorated with ``@_register_unpackers``, which at
class-definition time scans every method for the ``@_complex_unpacker(op, ...)``
decorator and builds a ``cls._DISPATCH`` dictionary mapping aten ops to rewrite
methods.

.. code-block:: python

    @_complex_unpacker(torch.ops.aten.mul.Tensor)
    def _rewrite_mul(self, node: Node, b: SubgraphBuilder, ...):
        ...

The entry point ``rewrite_subgraph_nodes`` iterates over the (topologically
ordered) subgraph nodes and for each node:

1. Looks up ``node.target`` in ``_DISPATCH``.
2. If found, calls the corresponding rewrite method.
3. If not found but the op is in ``_ELEMENTWISE_SAFE``, skips it (the op applies
   independently to every scalar, so the ``(..., 2)`` real layout is already
   correct).
4. Otherwise logs a warning and leaves the node unchanged.

``_ELEMENTWISE_SAFE``
"""""""""""""""""""""

The ``_ELEMENTWISE_SAFE`` set contains ops that apply to every element of the
tensor independently — ``add.Tensor``, ``sub.Tensor``, ``neg``, ``mul.Scalar``,
``clone``, ``where``, etc.  On the ``(..., 2)`` real layout these are already
correct: adding two complex tensors element-wise is the same as adding their
real and imaginary parts independently.

Notably **excluded** from this set:

* ``permute.default`` — must append the trailing real/imag dim index.
* ``add.Scalar`` / ``sub.Scalar`` — a scalar added to a complex number only
  shifts the real part; on the ``(..., 2)`` layout both parts would be shifted.
* ``reshape`` / ``view`` — shape arguments need updating for the extra ``2`` dim.

Complex Multiply Decomposition
"""""""""""""""""""""""""""""""

The most important rewrite is ``mul.Tensor`` between two complex operands.
The rewriter calls ``complex_mul_replacement``:

.. code-block:: python

    # inputs a, b have shape (..., 2) — last dim is [real, imag]
    re_a = select(a, -1, 0);  im_a = select(a, -1, 1)
    re_b = select(b, -1, 0);  im_b = select(b, -1, 1)
    real_out = re_a * re_b - im_a * im_b   # ac - bd
    imag_out = re_a * im_b + im_a * re_b   # ad + bc
    result   = stack([real_out, imag_out], dim=-1)

Each step is inserted via a ``SubgraphBuilder`` anchored at the ``mul`` node,
so all six new nodes appear immediately after it in topological order.  The
original ``mul`` node is then replaced and erased.

See :ref:`subgraph_builder` for more on how ``SubgraphBuilder`` manages
cursor-based insertion.

The ``is_complex_layout`` Metadata Invariant
"""""""""""""""""""""""""""""""""""""""""""""

Input replacement (Stage 2) converts complex ``placeholder`` nodes to
``float32``.  After that, ``is_complex_dtype(node)`` returns ``False`` for those
nodes even though they logically represent complex quantities.

To avoid missed rewrites, every node that represents a complex quantity is
annotated with ``node.meta["is_complex_layout"] = True`` during the detection
phase (lines in ``rewrite_subgraph_nodes`` before any rewrites begin).  The
annotation is then propagated forward by every rewrite handler:

* ``replace_input_node`` stamps it on the new placeholder and ``get_attr`` nodes.
* ``_inline_cat_re_im`` stamps it on every ``[re_u, im_u]`` concatenation node,
  covering all math handlers (``exp``, ``log``, ``sin``, ``mul``, etc.) at once.
* Each shape-manipulation handler (``reshape``, ``permute``, ``unsqueeze``,
  ``cat``, ``stack``, etc.) stamps it on its output node explicitly.

``_is_complex_layout_node(n)`` is therefore a direct metadata lookup — no shape
heuristics (``val.shape[-1] == 2``), no recursive ``_SHAPE_TRANSPARENT_OPS``
propagation.  This also eliminates false-positives on real parameters that
coincidentally have a trailing dimension of size 2.

FakeTensorMode Reuse for Dynamic Shapes
"""""""""""""""""""""""""""""""""""""""""

When inserting a new ``placeholder`` for a complex input, the pass must populate
``meta["val"]`` with a ``FakeTensor`` of the new real shape.  Using a fresh
``FakeTensorMode()`` would create a *new* ``ShapeEnv``, which is incompatible
with the one that ``torch.export`` used to encode dynamic shape constraints
(SymInt ranges).

The fix is to extract the ``FakeTensorMode`` from the *original* placeholder's
``meta["val"].fake_mode`` and reuse it.  The new fake tensor is then constructed
by appending a concrete ``2`` to the symbolic shape list:

.. code-block:: python

    orig_fake = input_node.meta["val"]
    sym_shape = list(orig_fake.shape) + [2]
    with orig_fake.fake_mode:
        fake_tensor = torch.empty(sym_shape, dtype=new_dtype, device=device)

This preserves all SymInt identity across the graph and keeps
dynamic-shape exports working correctly.

Entry Point: ``complex_graph_detection``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The public entry point called by the lowering pipeline is
``complex_graph_detection(gm, settings)``.  It:

1. Instantiates ``ComplexOpDetector`` and ``ComplexGraphRewriter``.
2. Calls ``find_complex_op_subgraphs`` anchored on ``view_as_real`` to find
   bounded complex subgraphs.
3. Calls ``find_all_complex_subgraphs`` for any remaining complex nodes that
   are not ``view_as_real``-bounded.
4. For each subgraph:

   a. Calls ``replace_input_node`` on every boundary input node (Stage 2).
   b. Calls ``rewrite_subgraph_nodes`` on the ordered subgraph (Stage 3).
   c. Calls ``clean_up_graph_after_modifications`` to remove dead nodes.

5. Returns the modified ``GraphModule``.

Adding New Op Rewrites
^^^^^^^^^^^^^^^^^^^^^^^

To teach the rewriter about a new complex op, add a method to
``ComplexGraphRewriter`` tagged with ``@_complex_unpacker``:

.. code-block:: python

    @_complex_unpacker(torch.ops.aten.my_new_op.default)
    def _rewrite_my_new_op(self, node: Node) -> bool:
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            out = b(my_real_impl, re, im)
            # If the output is still a complex-layout [..., 2] tensor, annotate it.
            # (Not needed if using _inline_cat_re_im, which sets the flag automatically.)
            out.meta["is_complex_layout"] = True
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
        return True

``@_register_unpackers`` (applied to the class) picks up the new entry
automatically at import time — no other registration is required.

If the new op is elementwise-safe on the ``(..., 2)`` layout (i.e. it acts
independently on every scalar), add it to ``_ELEMENTWISE_SAFE`` instead.

Related
-------

* :ref:`lowering` — the complex rewrite is a lowering pass.
* :ref:`subgraph_builder` — the ``SubgraphBuilder`` helper used in every rewrite method.
* :ref:`lowering_passes_catalog` — pass ordering and management.
