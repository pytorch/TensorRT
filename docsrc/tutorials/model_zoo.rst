Model Zoo
=========

End-to-end compilation examples for popular models across vision, language, and
generative AI. Each example shows how to compile the model with Torch-TensorRT
and benchmark against eager PyTorch.

Vision
------

* :doc:`ResNet (image classification) <_rendered_examples/dynamo/torch_compile_resnet_example>`
* :doc:`BERT (transformer) <_rendered_examples/dynamo/torch_compile_transformers_example>`

Generative AI
-------------

* :doc:`GPT-2 (text generation) <_rendered_examples/dynamo/torch_compile_gpt2>`
* :doc:`Stable Diffusion (image generation) <_rendered_examples/dynamo/torch_compile_stable_diffusion>`
* :doc:`FLUX.1-dev (image generation) <_rendered_examples/dynamo/torch_export_flux_dev>`
* :doc:`SAM2 (segmentation) <_rendered_examples/dynamo/torch_export_sam2>`

HuggingFace Models
------------------

* :doc:`Compiling HuggingFace models <compile_hf_models>`

Distributed Inference
---------------------

* :doc:`Data-parallel GPT-2 <_rendered_examples/distributed_inference/data_parallel_gpt2>`
* :doc:`Data-parallel Stable Diffusion <_rendered_examples/distributed_inference/data_parallel_stable_diffusion>`
* :doc:`Tensor-parallel simple example <_rendered_examples/distributed_inference/tensor_parallel_simple_example>`

Notebooks
---------

* :doc:`Jupyter notebooks <notebooks>`
