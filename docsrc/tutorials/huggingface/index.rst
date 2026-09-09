HuggingFace Models
==================

Compile and accelerate HuggingFace models with Torch-TensorRT: large language models
and visual language models via the ``tools/llm`` toolkit, Stable Diffusion via
``torch.compile``, Flux via ``torch.export``, and LoRA weight-swapping via
the Mutable Torch-TensorRT Module.

.. toctree::
   :maxdepth: 1

   compile_hf_models
   Example: Compiling Stable Diffusion with torch.compile <../_rendered_examples/dynamo/torch_compile_stable_diffusion>
   Example: Compiling FLUX.1-dev with the dynamo backend <../_rendered_examples/dynamo/torch_export_flux_dev>
   Example: FLUX.1-dev INT4 WOQ <../_rendered_examples/dynamo/torchao/torch_export_flux_int4_woq>
   Example: FLUX.1-dev NVFP4 WOQ <../_rendered_examples/dynamo/torchao/torch_export_flux_nvfp4_woq>
   Example: Qwen3-8B INT4 WOQ <../_rendered_examples/dynamo/torchao/torch_export_qwen3_int4_woq>
   Example: Mutable Torch TensorRT Module <../_rendered_examples/dynamo/mutable_torchtrt_module_example>
