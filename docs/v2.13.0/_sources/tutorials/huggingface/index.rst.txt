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
   Example: Mutable Torch TensorRT Module <../_rendered_examples/dynamo/mutable_torchtrt_module_example>
