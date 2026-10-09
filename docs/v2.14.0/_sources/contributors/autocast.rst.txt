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
