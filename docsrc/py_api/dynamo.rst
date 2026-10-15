.. _torch_tensorrt_dynamo_py:

torch_tensorrt.dynamo
===================

.. currentmodule:: torch_tensorrt.dynamo

.. automodule:: torch_tensorrt.dynamo
   :members:
   :undoc-members:
   :show-inheritance:

Functions
------------

.. autofunction:: compile

.. autofunction:: trace

.. autofunction:: export

.. autofunction:: convert_exported_program_to_serialized_trt_engine

.. autofunction:: cross_compile_for_windows

.. autofunction:: save_cross_compiled_exported_program

.. autofunction:: load_cross_compiled_exported_program

.. autofunction:: refit_module_weights

Classes
--------

.. autoclass:: CompilationSettings

Tools
-----

Weight mapping: compile from a safetensors checkpoint's weights in place (see the resource management guide).

.. currentmodule:: torch_tensorrt.dynamo.tools

.. autofunction:: map_weights

.. autofunction:: save_weights_for_mapping

.. autofunction:: release_mapped_storage

.. autoclass:: MappedWeights
