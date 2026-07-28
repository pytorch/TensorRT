Runtime Optimization
=====================

Optimize inference throughput and latency: CUDA Graphs for kernel-replay,
pre-allocated output buffers, multiple optimization profiles for distinct shape
regimes (e.g. LLM prefill/decode), and choosing the Python vs C++ TRT execution
path.

.. toctree::
   :maxdepth: 1

   cuda_graphs
   Example: Torch Export with Cudagraphs <../_rendered_examples/dynamo/torch_export_cudagraphs>
   Example: Pre-allocated output buffer <../_rendered_examples/dynamo/pre_allocated_output_example>
   multi_optimization_profiles
   Example: Multiple optimization profiles (prefill/decode) <../_rendered_examples/dynamo/multi_optimization_profiles>
   Python vs C++ runtime <python_runtime>
