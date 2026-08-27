Run an ExecuTorch model with Torch-TensorRT
============================================

Torch-TensorRT can export TensorRT-accelerated models as ExecuTorch ``.pte``
programs.  The optional ``torch-tensorrt-executorch-runtime`` package provides
the Python runtime integration required to load and run those programs.  It is
intended for deployment environments: it does not require PyTorch or LibTorch
at runtime.

Prerequisites
-------------

This runtime is supported on Linux and requires a compatible NVIDIA driver and
TensorRT runtime installation.  The ``.pte`` program must be run on a GPU
compatible with the TensorRT engine embedded when it was exported.

Install the runtime package alongside the matching Torch-TensorRT release:

.. code-block:: bash

   pip install torch-tensorrt-executorch-runtime

Install a version that matches the version used to export the model when
pinning dependencies for deployment.  The package installs the ExecuTorch
TensorRT backend; it is separate from ``torch-tensorrt`` so export-only users
do not need to install ExecuTorch runtime components.

Load and run a ``.pte`` program
--------------------------------

Use :func:`torch_tensorrt.load` with ``format=\"executorch\"`` to load a saved
program.  The returned module accepts and returns PyTorch tensors in the same
way as the original exported method:

.. code-block:: python

   import torch
   import torch_tensorrt

   model = torch_tensorrt.load("model.pte", format="executorch")

   input_tensor = torch.randn(1, 3, 224, 224, device="cuda")
   with torch.inference_mode():
       output = model(input_tensor)

The TensorRT engine is loaded and executed by the native ExecuTorch backend.
Inputs should be placed on the CUDA device expected by the exported engine.

Exporting the model
-------------------

Export a TensorRT-compiled ``ExportedProgram`` or FX graph to ExecuTorch with
the one-step save API:

.. code-block:: python

   import torch_tensorrt

   torch_tensorrt.save(
       trt_compiled_model,
       "model.pte",
       output_format="executorch",
   )

For workflows that need to inspect or further transform the ExecuTorch edge
program before writing it, use :mod:`torch_tensorrt.executorch` instead.  See
:doc:`../user_guide/runtime_performance/saving_models` for the export options
and examples.

Deployment notes
----------------

* A ``.pte`` embeds serialized TensorRT engines.  Build the artifact for the
  TensorRT version and GPU architecture used in deployment.
* TensorRT engines are device-specific.  Re-export the model when moving to an
  incompatible GPU, TensorRT version, or CUDA environment.
* The runtime package is only needed to execute ``.pte`` programs.  Continue
  using ``torch-tensorrt`` for compilation and export.
