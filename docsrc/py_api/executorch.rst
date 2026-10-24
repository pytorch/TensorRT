.. _torch_tensorrt_executorch_py:

torch_tensorrt.executorch
=========================

.. currentmodule:: torch_tensorrt.executorch

.. automodule:: torch_tensorrt.executorch

.. note::

   This module needs the ``executorch`` extra. It runs on Linux, and on Windows x86-64 in builds
   newer than 2.15. See :ref:`executorch_deployment`.

Overview
--------

The ``executorch`` module lowers exported programs into an ExecuTorch Edge
program with TensorRT engines delegated to the TensorRT backend. Use it when you
want to inspect or customize the Edge program before serialization, or when a
single ``.pte`` has to carry more than one method.

.. code-block:: python

    import torch_tensorrt.executorch

    edge = torch_tensorrt.executorch.export({"forward": exported_program})
    program = edge.to_executorch()

    with open("model.pte", "wb") as f:
        program.write_to_file(f)

For the simpler case where no inspection is needed, ``torch_tensorrt.save(...,
output_format="executorch")`` writes the file in one call.

Functions
------------

.. autofunction:: export
.. autofunction:: get_edge_compile_config
.. autofunction:: zero_copy_backend_config
.. autofunction:: check_zero_copy_kv

Classes
--------

.. autoclass:: TensorRTPartitioner
.. autoclass:: TensorRTBackend
