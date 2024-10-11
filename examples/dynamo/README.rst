.. _torch_compile:

Torch-TensorRT Examples
====================================

Please refer to the following examples which demonstrate the usage of different features of Torch-TensorRT. We also provide
examples of Torch-TensorRT compilation of select computer vision and language models.

Dependencies
------------------------------------

Please install the following external dependencies (assuming you already have correct `torch`, `torch_tensorrt` and `tensorrt` libraries installed (`dependencies <https://github.com/pytorch/TensorRT?tab=readme-ov-file#dependencies>`_))

.. code-block:: python

    pip install -r requirements.txt


Compiler Features
------------------------------------
* :ref:`torch_compile_advanced_usage`: Advanced usage including making a custom backend to use directly with the ``torch.compile`` API
* :ref:`torch_export_cudagraphs`: Using the Cudagraphs integration with `ir="dynamo"`
* :ref:`converter_overloading`: How to write custom converters and overload existing ones
* :ref:`custom_kernel_plugins`: Creating a plugin to use a custom kernel inside TensorRT engines
* :ref:`refit_engine_example`: Refitting a compiled TensorRT Graph Module with updated weights
* :ref:`mutable_torchtrt_module_example`: Compile, use, and modify TensorRT Graph Module with MutableTorchTensorRTModule
* :ref:`vgg16_fp8_ptq`: Compiling a VGG16 model with FP8 and PTQ using ``torch.compile``
* :ref:`engine_caching_example`: Utilizing engine caching to speed up compilation times
* :ref:`engine_caching_bert_example`: Demonstrating engine caching on BERT

Model Zoo
------------------------------------
* :ref:`torch_compile_resnet`: Compiling a ResNet model using the Torch Compile Frontend for ``torch_tensorrt.compile``
* :ref:`torch_compile_transformer`: Compiling a Transformer model using ``torch.compile``
* :ref:`torch_compile_stable_diffusion`: Compiling a Stable Diffusion model using ``torch.compile``
* :ref:`_torch_export_gpt2`: Compiling a GPT2 model using AOT workflow (`ir=dynamo`)
* :ref:`_torch_export_llama2`: Compiling a Llama2 model using AOT workflow (`ir=dynamo`)