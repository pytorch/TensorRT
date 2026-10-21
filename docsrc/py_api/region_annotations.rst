.. _torch_tensorrt_region_annotations_py:

torch_tensorrt.region_annotations
================================

.. currentmodule:: torch_tensorrt.region_annotations

These experimental context managers choose how a selected group of model
operations executes. Use direct ``torch_tensorrt.compile(..., ir="dynamo",
strict=True, offload_module_to_cpu=False)`` capture with static shapes and an
eval-mode model. During ordinary eager execution, the annotated body runs
normally.

See :ref:`execute_in_torch` for the tutorial, supported settings, and save/load
limits. Runnable examples cover :ref:`eager and Inductor regions
<execute_in_torch_example>` and :ref:`automatic custom-kernel plugins
<fuse_in_plugin_example>`.

Region annotations
------------------

.. autofunction:: execute_in_torch

.. autofunction:: fuse_in_plugin

Errors
------

.. autoexception:: RegionError

.. autoexception:: RegionCaptureError
