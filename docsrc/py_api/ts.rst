.. _torch_tensorrt_ts_py:

TorchScript frontend removed
===========================

``torch_tensorrt.ts`` and the ``ir="ts"`` / ``ir="torchscript"`` compilation paths
have been removed following the removal of PyTorch's public TorchScript APIs.
Use ``torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)`` and
:doc:`save an ExportedProgram, AOTInductor package, or ExecuTorch program <../user_guide/runtime_performance/saving_models>`.

Existing TorchScript artifacts cannot be loaded with ``torch_tensorrt.load``.
Recompile the original PyTorch model with the Dynamo frontend.
