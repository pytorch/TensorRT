Model Zoo
=========

End-to-end compilation examples for popular models across vision, language, and
generative AI. Each example shows how to compile the model with Torch-TensorRT
and benchmark against eager PyTorch.

.. toctree::
   :maxdepth: 1
   :caption: Vision

   _rendered_examples/dynamo/torch_compile_resnet_example
   _rendered_examples/dynamo/torch_compile_transformers_example
   _rendered_examples/dynamo/engine_caching_bert_example

.. toctree::
   :maxdepth: 1
   :caption: Generative AI

   _rendered_examples/dynamo/torch_compile_gpt2
   _rendered_examples/dynamo/torch_export_sam2
   _rendered_examples/dynamo/llama2_flashinfer_rmsnorm

For diffusion models see :doc:`HuggingFace Models <huggingface/index>`.

.. toctree::
   :maxdepth: 1
   :caption: Notebooks

   notebooks
