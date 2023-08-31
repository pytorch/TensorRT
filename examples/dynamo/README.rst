.. _torch_compile:

Dynamo / ``torch.compile``
----------------------------

Torch-TensorRT provides a backend for the new ``torch.compile`` API released in PyTorch 2.0. In the following examples we describe
a number of ways you can leverage this backend to accelerate inference.

* :ref:`torch_compile_resnet`: Compiling a ResNet model using the Torch Compile Frontend for ``torch_tensorrt.compile``
* :ref:`torch_compile_transformer`: Compiling a Transformer model using ``torch.compile``
* :ref:`torch_compile_advanced_usage`: Advanced usage including making a custom backend to use directly with the ``torch.compile`` API
:ref:`dynamo_aten_lowering_passes`: Custom modifications of a graph of ATen operators via lowering passes
