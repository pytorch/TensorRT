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

* **``originally_complex`` set** — the set of nodes that were complex-dtype
  *before* any rewrites. After ``replace_input_node``, complex placeholders become
  ``float32`` so ``is_complex_dtype()`` returns ``False``. The ``originally_complex``
  set is used to decide which ``mul.Tensor`` nodes need the complex mul rewrite.
* **FakeTensorMode reuse** — ``propagate_metadata`` must use the ``FakeTensorMode``
  from existing placeholder fake tensors (not a fresh mode) to avoid mode-mismatch
  errors under ``torch.compile`` and to preserve SymInt for dynamic shapes.
* **Dotted buffer names** — ``register_buffer`` rejects names containing ``.``.
  Nested submodule parameter names (e.g. ``layers.0.weight``) must have ``.``
  replaced with ``__`` before registration.

Related
-------

* :ref:`lowering` — the complex rewrite is a lowering pass.
* :ref:`lowering_passes_catalog` — pass ordering and management.
