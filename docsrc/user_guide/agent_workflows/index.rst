Agent workflows
===============

Torch-TensorRT exposes versioned reports for preflight analysis, compilation,
validation, and benchmarking. An assistant can inspect these reports without
interpreting terminal output. The existing ``compile()`` return type and
``dryrun`` interface remain available.

Start with :doc:`direct_api` to use the Python APIs in a provisioned
Torch-TensorRT environment. Core Torch-TensorRT has no agent framework or
protocol dependency.

The first release targets the Dynamo ahead-of-time ``ExportedProgram`` path.
Analysis lowers and partitions an isolated graph, and reports converter
eligibility separately from the final TensorRT assignment. Constant folding can
execute tensor operations on that copy; analysis requires the compiler's
Torch/CUDA/TensorRT environment. Eligibility does not guarantee a successful
engine build or a speedup.

Validation and benchmarking use prepared inputs and an explicit stateless
evaluation protocol. Persistence and orchestration policies remain the caller's
responsibility.

.. toctree::
   :maxdepth: 2

   direct_api
   contracts_testing
