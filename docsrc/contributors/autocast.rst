.. _autocast_design:

Autocast and Precision Management
===================================

.. note::

   This page documents the design for rule-based autocast in Torch-TensorRT.
   Original design discussion:
   `RFC #3869 <https://github.com/pytorch/TensorRT/discussions/3869>`_.

Background
-----------

TensorRT historically supported *weak typing* — the builder was allowed to select
the lowest-precision kernel for each layer (e.g. downcast fp32 inputs to fp16
automatically). This behavior was deprecated in TensorRT 10.x. Torch-TensorRT
replaces it with a PyTorch-native approach: a lowering pass that inserts explicit
cast nodes before layers that should run in lower precision, following rules
similar to
`NVIDIA ModelOpt Autocast <https://nvidia.github.io/TensorRT-Model-Optimizer/guides/8_autocast.html>`_.

Three Precision Modes
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Mode
     - ``enable_autocast``
   * - User-defined per-layer precision (strong typing, always on)
     - ``False`` (default)
   * - Autocast (rule-based mixed precision)
     - ``True``

The autocast mode is the focus of this page.

User API
---------

.. code-block:: python

    import torch_tensorrt

    trt_mod = torch_tensorrt.compile(
        exported_program.module(),
        arg_inputs=inputs,
        min_block_size=1,
        enable_autocast=True,
        low_precision_type=torch.float16,   # target low-precision dtype
        nodes_to_exclude={"^conv2d$"},       # regex patterns → keep fp32
        targets_to_exclude={},
        data_max=512,                        # threshold for data-sensitive ops
        max_depth_of_reduction=None,
    )

Autocast-aware PyTorch code (``torch.autocast`` context managers) is respected:
ops inside a ``torch.autocast(fp32)`` context remain fp32 even when the global
low-precision type is fp16.

Internal Implementation
------------------------

Stage 1 — Rule-Based Node Classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each node in the FX graph is assigned a precision (high or low) according to a
predefined ruleset. A node stays in fp32 (high precision) if **any** of the
following rules fire:

* It is inside a ``torch.autocast`` context (``wrap_with_autocast`` node).
* Its name matches a ``nodes_to_exclude`` regex.
* Its target matches ``targets_to_exclude``.
* It performs a reduction over a large domain (``max_depth_of_reduction`` heuristic).
* The maximum data value flowing through it exceeds ``data_max`` (guards against
  overflow in fp16).
* It reads a floating point weight or a number such as the ``1e6`` of ``x / 1e6``,
  with a finite value that becomes ``inf`` in the low precision type, such as a
  BatchNorm running variance of ``1e6`` in fp16. ``NaN`` and ``inf`` constants
  stay as they are, so they do not count.
  A mask tensor whose largest finite magnitude is the limit of its type, such
  as ``torch.finfo(torch.float32).min``, does not count when an op only
  places it in its output, as ``torch.where``, ``masked_fill``,
  ``index_put`` without ``accumulate=True``, ``fill_`` and ``copy_`` do, or
  when ``scaled_dot_product_attention`` adds it to its scores as
  ``attn_mask``: such a value stands for infinity and becomes ``-inf``, as it
  does without this rule. So a row where every value is masked holds only
  ``-inf``, also as without this rule. For a row of only ``-inf``, eager
  ``scaled_dot_product_attention`` gives 0 on CPU and CUDA, and a softmax
  gives ``NaN``. For a row of only ``torch.finfo(torch.float32).min``, as in
  the fp32 model, it averages the values of all keys on CPU and gives 0 on
  CUDA by default, and a softmax gives a uniform row. The other finite values
  of that tensor still count, so a buffer that holds both the mask value and
  ``1e6`` keeps the op in fp32. A placed tensor with a smaller largest
  magnitude, such as a weight of ``1e6``, still counts. A value at the limit
  always stands for infinity, so a placed weight that holds
  ``torch.finfo(torch.float32).max`` as a real value becomes ``inf``. A number
  given to ``masked_fill`` still counts: in fp16, ``masked_fill(mask, -1e9)``
  does not compile otherwise. An additive mask, as in ``scores + mask``, also
  counts, so the add runs in fp32 and the scores are cast twice; pass the mask
  as ``attn_mask`` to avoid that. A view or an in-place op on a tensor that
  this rule keeps runs in fp32 with it, since a cast would make an in-place op
  write into a copy. So a constant that an in-place op reads, such as the
  ``1e6`` of ``h.add_(1e6)``, keeps the op that made ``h`` in fp32 when
  another op also reads ``h``, such as a view of it taken before. When the
  only readers of ``h`` are in-place ops on it, as for
  ``scores.masked_fill_(mask, -1e9)`` after
  ``scores = q @ k.transpose(-2, -1)``, the matmul stays in fp16, and
  ``masked_fill_`` runs in fp32 on a copy that the ops after it read. A
  weight, buffer or graph input is not an op, so the in-place ops on it run in
  fp32 when one of them reads such a constant, as ``buffer.add_(w)`` does for
  a weight ``w`` of ``1e6``. So do the in-place ops on another tensor that
  this rule cannot keep, such as the output of ``torch.full``, of an op given
  a dtype such as ``x.float()``, or of an op that another rule keeps in fp32.
  On one of another floating type, they compute in fp32 and write the result
  in that type, rounded as in eager for float16 and bfloat16, and its views
  see the write. A float64 buffer written this way needs
  ``truncate_double=True``, as without autocast. An op that writes another
  tensor it takes, such as the ``out=`` of ``torch.add``, writes into a cast
  copy of it, which is rounded and written back, as in eager. A list writer
  such as ``_foreach_add_`` writes each tensor in its own type, as in eager,
  so the constants it reads count for each tensor it writes, and it gets each
  source in the type of its tensor. A ``torch.no_grad()`` block runs in one
  type for all the tensors it takes, so this rule does not keep an op that
  such a block may write into while it also takes other tensors.
  Not covered yet: an integer weight, such as an int64 buffer of ``100000``; a
  number or a computed value used inside a ``torch.no_grad()`` block, which
  export keeps as one node; a view of an op that another rule keeps in fp32; a
  list writer such as ``_foreach_mul_`` that writes into a float16 tensor and
  reads such a constant, which it gets in float16; a list of numbers, such as
  the ``[1e6, 1.0]`` of ``_foreach_add_(tensors, [1e6, 1.0])``, or a number
  the graph reads with ``item()``; an ``out=`` tensor of a type other than
  float32 that is also an input of the op, as in ``torch.add(b, 1e6, out=b)``
  for a bfloat16 ``b``, whose result is not rounded and which a view taken
  before does not see; and the users of an op kept for a constant that it only
  places or gathers, such as ``torch.where(c, x, w)``,
  ``torch.cat([x, w])``, ``index_select``, ``gather`` or ``embedding`` with a
  weight ``w`` of ``1e6``: they read its large values in the low precision type.
* It is an ``operator.getitem`` node (output unpacking).

All other nodes are assigned the ``low_precision_type``.

For the example CNN in the RFC, the classifier output is:

.. code-block:: text

    Low Precision:  relu, max_pool2d, conv2d_1, relu_1, max_pool2d_1, flatten
    High Precision: conv2d  (matches nodes_to_exclude="^conv2d$")

Stage 2 — Graph Modification (Pre-Lowering Pass)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pass is implemented in
``py/torch_tensorrt/dynamo/lowering/passes/rule_based_autocast.py``.
It inserts ``aten._to_copy`` (cast) nodes before each low-precision op to convert
its inputs to the target dtype. High-precision nodes are left unmodified.

Example transformed graph fragment:

.. code-block:: text

    %convolution   : fp32  (conv2d excluded from autocast)
    %_to_copy      : cast convolution → fp16
    %relu          : fp16
    %max_pool2d    : fp16
    %_to_copy_1    : cast back → fp32  (before fc1 which is in autocast(fp32) context)
    %linear        : fp32
    %add           : fp32

The resulting graph has explicit dtype annotations on every edge, satisfying
TensorRT's strong-typing requirement without relying on deprecated weak-typing.

Interaction with ``torch.autocast``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a user wraps ops in ``torch.autocast``:

.. code-block:: python

    with torch.autocast(x.device.type, enabled=True, dtype=torch.float32):
        x = self.fc1(x)
        x = torch.add(x, x)

the exported graph contains a ``torch.ops.higher_order.wrap_with_autocast`` node.
The classifier detects this and marks all ops inside the context as high-precision
(fp32), regardless of the global ``low_precision_type``.

Related
-------

* :ref:`lowering` — autocast is a pre-lowering pass.
* :ref:`lowering_passes_catalog` — pass ordering and management.
* `Example: autocast_example.py <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/autocast_example.py>`_
