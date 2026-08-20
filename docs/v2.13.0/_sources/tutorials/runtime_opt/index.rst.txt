Runtime Optimization
=====================

Optimize inference throughput and latency: CUDA Graphs for kernel-replay,
pre-allocated output buffers, and choosing the Python vs C++ TRT execution path.

.. toctree::
   :maxdepth: 1

   cuda_graphs
   Example: Torch Export with Cudagraphs <../_rendered_examples/dynamo/torch_export_cudagraphs>
   Example: Pre-allocated output buffer <../_rendered_examples/dynamo/pre_allocated_output_example>
   Python vs C++ runtime <python_runtime>
