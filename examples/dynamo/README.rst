.. _torch_compile:

Dynamo / ``torch.compile``
----------------------------

Torch-TensorRT provides a backend for the new ``torch.compile`` API released in PyTorch 2.0. In the following examples we describe
a number of ways you can leverage this backend to accelerate inference.

* :ref:`torch_compile_resnet`: Compiling a ResNet model using the Torch Compile Frontend for ``torch_tensorrt.compile``
* :ref:`torch_compile_transformer`: Compiling a Transformer model using ``torch.compile``
* :ref:`torch_compile_advanced_usage`: Advanced usage including making a custom backend to use directly with the ``torch.compile`` API
* :ref:`torch_compile_stable_diffusion`: Compiling a Stable Diffusion model using ``torch.compile``
* :ref:`torch_export_cudagraphs`: Using the Cudagraphs integration with `ir="dynamo"`
* :ref:`custom_kernel_plugins`: Creating a plugin to use a custom kernel inside TensorRT engines
* :ref:`refit_engine_example`: Refitting a compiled TensorRT Graph Module with updated weights
* :ref:`mutable_torchtrt_module_example`: Compile, use, and modify TensorRT Graph Module with MutableTorchTensorRTModule
* :ref:`vgg16_fp8_ptq`: Compiling a VGG16 model with FP8 and PTQ using ``torch.compile``
* :ref:`engine_caching_example`: Utilizing engine caching to speed up compilation times
* :ref:`engine_caching_bert_example`: Demonstrating engine caching on BERT
