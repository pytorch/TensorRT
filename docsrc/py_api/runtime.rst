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

Runtime backend
---------------

Execution uses the C++ Torch-TensorRT extension when it is installed; otherwise the
Python ``TRTEngine`` path is used. There is no separate process-wide backend switch
in ``torch_tensorrt.runtime``.

Classes
---------

.. autoclass:: TorchTensorRTModule
   :members:
   :special-members: __init__
   :show-inheritance:

   Single runtime module for TensorRT engines. Dispatches to the C++ or Python execution
   implementation depending on whether the C++ extension is available. See :ref:`python_runtime`.
