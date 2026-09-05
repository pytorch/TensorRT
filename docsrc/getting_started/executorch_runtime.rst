.. _executorch_runtime_getting_started:

Deploy an ExecuTorch Model with the Runtime Wheel
##################################################

Torch-TensorRT can package TensorRT-accelerated subgraphs in an ExecuTorch
``.pte`` program. Starting with the 2.14 release, install the separate
``torch-tensorrt-executorch-runtime`` wheel to load and run that program in
Python. The export environment needs Torch-TensorRT and ExecuTorch; the
inference environment needs the runtime wheel and its matching dependencies.

This workflow is currently supported on Linux x86 and aarch64. Windows support
is planned for the next release. Install all packages from the same release
matrix so that PyTorch, Torch-TensorRT, ExecuTorch, TensorRT, and the runtime
wheel are binary compatible.

Export a ``.pte`` model
------------------------

Install the ExecuTorch export dependencies:

.. code-block:: shell

    python -m pip install "torch-tensorrt[executorch]"

Use the included static-shape export example to compile a model and save it as
an ExecuTorch ``.pte`` program:

.. code-block:: shell

    python examples/torchtrt_executorch_example/export_static_shape.py \
        --model_path=model.pte

The example exports a static-shape model, compiles it with TensorRT, and embeds
the TensorRT engine in ``model.pte``.

Load and run with the runtime wheel
-----------------------------------

Install the runtime wheel in the inference environment:

.. code-block:: shell

    python -m pip install torch-tensorrt-executorch-runtime

Use the included loader example to load the ``.pte`` program with the runtime
wheel and run its ``forward`` method:

.. code-block:: shell

    python examples/executorch_reference_runner/load_model.py \
        --model_path=model.pte

