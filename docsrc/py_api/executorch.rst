.. _torch_tensorrt_executorch_py:

torch_tensorrt.executorch
=========================

.. currentmodule:: torch_tensorrt.executorch

.. automodule:: torch_tensorrt.executorch

.. note::

   This module requires ExecuTorch and is only supported on Linux.

Overview
--------

The ``executorch`` module lowers exported programs into an ExecuTorch Edge
program with TensorRT engines delegated to the TensorRT backend. Use it when you
want to inspect or customize the Edge program before serialization, or when a
single ``.pte`` has to carry more than one method.

.. code-block:: python

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

Classes
--------

.. autoclass:: TensorRTPartitioner
.. autoclass:: TensorRTBackend
