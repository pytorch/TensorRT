.. _dynamo_export:

Compiling Exported Programs with Torch-TensorRT
=============================================
.. currentmodule:: torch_tensorrt.dynamo

.. automodule:: torch_tensorrt.dynamo
   :members:
   :undoc-members:
   :show-inheritance:

Pytorch 2.1 introduced ``torch.export`` APIs which
can export graphs from Pytorch programs into ``ExportedProgram`` objects. Torch-TensorRT dynamo
frontend compiles these ``ExportedProgram`` objects and optimizes them using TensorRT. Here's a simple
usage of the dynamo frontend

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224), dtype=torch.float32).cuda()]
    exp_program = torch.export.export(model, tuple(inputs))
    trt_gm = torch_tensorrt.dynamo.compile(exp_program, inputs) # Output is a torch.fx.GraphModule
    trt_gm(*inputs)

.. note::  ``torch_tensorrt.dynamo.compile`` is the main API for users to interact with Torch-TensorRT dynamo frontend. The input type of the model should be ``ExportedProgram`` (ideally the output of ``torch.export.export`` or ``torch_tensorrt.dynamo.trace`` (discussed in the section below)) and output type is a ``torch.fx.GraphModule`` object.

Customizable Settings
----------------------

There are lot of options for users to customize their settings for optimizing with TensorRT.
Some of the frequently used options are as follows:

* ``inputs`` - For static shapes, this can be a list of torch tensors or `torch_tensorrt.Input` objects. For dynamic shapes, this should be a list of ``torch_tensorrt.Input`` objects.
* ``enabled_precisions`` - Set of precisions that TensorRT builder can use during optimization.
* ``truncate_long_and_double`` - Truncates long and double values to int and floats respectively.
* ``torch_executed_ops`` - Operators which are forced to be executed by Torch.
* ``min_block_size`` - Minimum number of consecutive operators required to be executed as a TensorRT segment.

The complete list of options can be found `here <https://github.com/pytorch/TensorRT/blob/123a486d6644a5bbeeec33e2f32257349acc0b8f/py/torch_tensorrt/dynamo/compile.py#L51-L77>`_

.. note:: We do not support INT precision currently in Dynamo. Support for this currently exists in our Torchscript IR. We plan to implement similar support for dynamo in our next release.

Under the hood
--------------

Under the hood, ``torch_tensorrt.dynamo.compile`` performs the following on the graph.

* Lowering - Applies lowering passes to add/remove operators for optimal conversion.
* Partitioning - Partitions the graph into Pytorch and TensorRT segments based on the ``min_block_size`` and ``torch_executed_ops`` field.
* Conversion - Pytorch ops get converted into TensorRT ops in this phase.
* Optimization - Post conversion, we build the TensorRT engine and embed this inside the pytorch graph.

Tracing
-------

``torch_tensorrt.dynamo.trace`` can be used to trace a Pytorch graphs and produce ``ExportedProgram``.
This internally performs some decompositions of operators for downstream optimization.
The ``ExportedProgram`` can then be used with ``torch_tensorrt.dynamo.compile`` API.
If you have dynamic input shapes in your model, you can use this ``torch_tensorrt.dynamo.trace`` to export
the model with dynamic shapes. Alternatively, you can use ``torch.export`` `with constraints <https://pytorch.org/docs/stable/export.html#expressing-dynamism>`_ directly as well.

.. code-block:: python

    import torch
    import torch_tensorrt

    inputs = [torch_tensorrt.Input(min_shape=(1, 3, 224, 224),
                                  opt_shape=(4, 3, 224, 224),
                                  max_shape=(8, 3, 224, 224),
                                  dtype=torch.float32)]
    model = MyModel().eval()
    exp_program = torch_tensorrt.dynamo.trace(model, inputs)
