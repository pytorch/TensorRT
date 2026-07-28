.. _torch_tensorrt_py:

torch_tensorrt
===============

.. automodule torch_tensorrt
   :undoc-members:



.. automodule:: torch_tensorrt
   :members:
   :undoc-members:
   :show-inheritance:

Functions
------------

.. autofunction:: set_device

.. autofunction:: compile

.. autofunction:: convert_method_to_trt_engine

.. autofunction:: cross_compile_for_windows

.. autofunction:: load_cross_compiled_exported_program

.. autofunction:: get_build_info

.. autofunction:: dump_build_info

.. autofunction:: save

.. autofunction:: load

Classes
---------
.. autoclass:: MutableTorchTensorRTModule
   :members:
   :special-members: __init__

.. autoclass:: Input
   :members:
   :special-members: __init__

.. autoclass:: Device
   :members:
   :special-members: __init__

Enums
-------

.. autoclass:: dtype
   :members:
   :member-order:

.. autoclass:: DeviceType
   :members:
   :member-order:

.. autoclass:: EngineCapability
   :members:
   :member-order:

.. autoclass:: memory_format
   :members:
   :member-order:

Submodules
----------

.. toctree::
   :maxdepth: 1

   logging
   ptq
   ts
   fx
   dynamo
   runtime
