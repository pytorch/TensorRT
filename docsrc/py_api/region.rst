.. _torch_tensorrt_region_py:

torch_tensorrt.region
=====================

.. automodule:: torch_tensorrt.region

These experimental APIs let you place a group of model operations yourself:
keep it in PyTorch, or replace it with a custom operator or a kernel. Mark the
group with a scope in ``forward``, or select whole submodules with ``regions=``
rules when you compile.

``torch.export`` records the scopes, with ``strict=True`` and with the default
``strict=False``. ``torch_tensorrt.compile(..., ir="dynamo")`` honors them, and
``torch_tensorrt.dynamo.compile`` honors them in a program you exported
yourself. So does ``torch.compile`` with ``backend="torch_tensorrt"``. The same
calls accept rules. Other ``torch.compile`` backends and ``torch.jit.trace``
ignore the scopes with a warning, and ordinary eager execution runs the body of
a scope as ordinary code. Regions need static shapes and an eval-mode model.

See :ref:`execute_in_torch` for the tutorial, which lists
:ref:`the compile paths that honor regions <region_compile_paths>`,
:ref:`what the body of a region may contain <region_body>`, the supported
settings, and the save/load limits. Runnable examples cover
:ref:`eager and Inductor regions <execute_in_torch_example>`,
:ref:`PyTorch custom operators <fused_op_example>`,
:ref:`custom-kernel plugins <fused_kernel_example>`, and
:ref:`rules <region_rules_example>`.

Region annotations
------------------

Write these scopes in ``forward``.

.. autofunction:: execute_in_torch

.. autofunction:: fused_op

.. autofunction:: fused_kernel

Region rules
------------

Pass these to a compile call as ``regions={selector: rule}``. A selector is a
module path, a module class, or a module object; see :ref:`region_rules`. With
``torch.compile``, pass them as ``options={"regions": {...}}``.

.. autoclass:: ExecuteInTorch

.. autoclass:: FusedOp

.. autoclass:: FusedKernel

Tracing without regions
-----------------------

Use this block to export a model with scopes for a tool that does not know the
region markers; see :ref:`region_ignore`.

.. autofunction:: ignore_regions

Errors
------

``RegionCaptureError`` is a subclass of ``RegionError``. ``torch.compile``
wraps both in its own exception types; see :ref:`region_torch_compile`.

.. autoexception:: RegionError

.. autoexception:: RegionCaptureError
