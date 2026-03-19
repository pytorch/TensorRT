.. _torch_tensorrt_py:

torch_tensorrt.runtime
==============================

.. automodule:: torch_tensorrt.runtime
   :members:
   :undoc-members:
   :show-inheritance:

Functions
------------

.. autofunction:: set_multi_device_safe_mode

.. autofunction:: enable_cudagraphs

.. autofunction:: get_cudagraphs_mode

.. autofunction:: get_whole_cudagraphs_mode

.. autofunction:: set_cudagraphs_mode

.. autofunction:: enable_pre_allocated_outputs

.. autofunction:: weight_streaming

.. autofunction:: enable_output_allocator

Runtime backend selection
-------------------------

.. autofunction:: torch_tensorrt.runtime.get_runtime_backend

.. autofunction:: torch_tensorrt.runtime.set_runtime_backend

Classes
---------

.. autoclass:: TorchTensorRTModule
   :members:
   :special-members: __init__
   :show-inheritance:

   Single runtime module for TensorRT engines. Dispatches to the C++ or Python execution
   implementation based on :func:`~torch_tensorrt.runtime.get_runtime_backend` /
   :func:`~torch_tensorrt.runtime.set_runtime_backend`. See :ref:`python_runtime`.

.. autoclass:: PythonTorchTensorRTModule
   :members:
   :special-members: __init__
   :show-inheritance:

   Subclass of ``TorchTensorRTModule`` that **pins** the Python engine path. Prefer
   ``TorchTensorRTModule`` plus compile flags unless you need this guarantee. See :ref:`python_runtime`.
