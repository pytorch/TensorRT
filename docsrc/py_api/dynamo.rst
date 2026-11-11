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

.. autoclass:: low_memory

.. note::

   ``low_memory`` is **experimental**. In 2.16 it takes the loaded model and writes its weights to a file
   itself; in 2.15 it took a function that builds the model and a weights file.

.. autoclass:: CompilationSettings

Tools
-----

Weight mapping: compile from a safetensors checkpoint's weights in place (see the resource management guide).

.. note::

   These tools are **experimental** in 2.15: their API will change in 2.16.

.. currentmodule:: torch_tensorrt.dynamo.tools

.. autofunction:: map_weights

.. autofunction:: save_weights_for_mapping

.. autofunction:: checkpoint_source

.. autoclass:: MappedWeights
