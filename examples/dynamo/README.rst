.. _torch_tensorrt_examples:

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
* :ref:`_torch_compile_phi3_vision`: Compiling a Phi 3 vision model from Hugging Face using ``torch.compile``