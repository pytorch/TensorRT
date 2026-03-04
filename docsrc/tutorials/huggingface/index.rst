HuggingFace Models
==================

Compile and accelerate HuggingFace models with Torch-TensorRT: large language models
and visual language models via the ``tools/llm`` toolkit, Stable Diffusion via
``torch.compile``, Flux via ``torch.export``, and LoRA weight-swapping via
the Mutable Torch-TensorRT Module.

.. toctree::
   :maxdepth: 1

   compile_hf_models
   ../_rendered_examples/dynamo/torch_compile_stable_diffusion
   ../_rendered_examples/dynamo/torch_export_flux_dev
   ../_rendered_examples/dynamo/mutable_torchtrt_module_example
