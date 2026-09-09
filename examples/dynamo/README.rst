.. _torch_tensorrt_examples:

Dynamo Examples
===============

Here we provide examples of Torch-TensorRT compilation of popular computer vision and language models.

Dependencies
------------------------------------

Please install the following external dependencies (assuming you already have correct `torch`, `torch_tensorrt` and `tensorrt` libraries installed (`dependencies <https://github.com/pytorch/TensorRT?tab=readme-ov-file#dependencies>`_))

.. code-block:: python

    pip install -r requirements.txt


Model Zoo
------------------------------------
* :ref:`torch_compile_resnet`: Compiling a ResNet model using the Torch Compile Frontend for ``torch_tensorrt.compile``
* :ref:`torch_compile_transformer`: Compiling a Transformer model using ``torch.compile``
* :ref:`torch_compile_stable_diffusion`: Compiling a Stable Diffusion model using ``torch.compile``
* :ref:`_torch_compile_gpt2`: Compiling a GPT2 model using ``torch.compile``
* :ref:`_torch_export_gpt2`: Compiling a GPT2 model using AOT workflow (`ir=dynamo`)
* :ref:`_torch_export_llama2`: Compiling a Llama2 model using AOT workflow (`ir=dynamo`)
* :ref:`_torch_export_sam2`: Compiling SAM2 model using AOT workflow (`ir=dynamo`)
* :ref:`_torch_export_flux_dev`: Compiling FLUX.1-dev model using AOT workflow (`ir=dynamo`)
* :ref:`quantize_linear_fp8_woq`: TorchAO FP8 weight-only quantization of a Linear layer (``examples/dynamo/torchao``)
* :ref:`quantize_linear_fp8_static`: TorchAO static FP8 (act + weight) quantization of a Linear layer (``examples/dynamo/torchao``)
* :ref:`torch_export_flux_fp8_woq`: Compiling FLUX.1-dev with TorchAO FP8 weight-only quantization (``examples/dynamo/torchao``)
* :ref:`quantize_linear_int4_woq`: TorchAO INT4 weight-only quantization of a Linear layer (``examples/dynamo/torchao``)
* :ref:`torch_export_flux_int4_woq`: Compiling FLUX.1-dev with TorchAO INT4 weight-only quantization (``examples/dynamo/torchao``)
* :ref:`torch_export_qwen3_int4_woq`: Compiling Qwen3-8B with TorchAO INT4 weight-only quantization (``examples/dynamo/torchao``)
* :ref:`quantize_linear_nvfp4_woq`: TorchAO NVFP4 weight-only quantization of a Linear layer (``examples/dynamo/torchao``)
* :ref:`torch_export_flux_nvfp4_woq`: Compiling FLUX.1-dev with TorchAO NVFP4 weight-only quantization (``examples/dynamo/torchao``)
* :ref:`quantize_linear_mxfp4`: TorchAO MXFP4 (dyn-act + MX weight) quantization of a Linear layer (``examples/dynamo/torchao``)
* :ref:`torch_export_flux_mxfp4`: Compiling FLUX.1-dev with TorchAO MXFP4 (``examples/dynamo/torchao``)
* :ref:`debugger_example`: Debugging Torch-TensorRT Compilation
* :ref:`torch_export_3d_rope`: Compiling a 3D RoPE video-transformer block with complex numerics support
* :ref:`engine_converter_binding_names`: Naming input / output bindings when emitting a raw serialized TRT engine