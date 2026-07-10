.. _torch_tensorrt_py:

torch_tensorrt
===============

.. automodule:: torch_tensorrt
   :members:
   :undoc-members:
   :show-inheritance:

Functions
------------

.. autofunction:: compile

.. autofunction:: convert_method_to_trt_engine

.. autofunction:: cross_compile_for_windows

.. autofunction:: load_cross_compiled_exported_program

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

:doc:`torch_tensorrt.logging <logging>` |
:doc:`torch_tensorrt.dynamo <dynamo>` |
:doc:`torch_tensorrt.runtime <runtime>` |
:doc:`torch_tensorrt.ts <ts>` |
:doc:`torch_tensorrt.ptq <ptq>` |
:doc:`torch_tensorrt.fx <fx>`
