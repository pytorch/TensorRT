.. _aot_inductor:

AOTInductor Deployment
======================

**AOTInductor** (Ahead-of-Time Inductor) compiles a PyTorch model into a self-contained
``.pt2`` package at build time. That package can be loaded and executed in Python or C++
without a Torch-TensorRT dependency at runtime.

Torch-TensorRT integrates with AOTInductor: TRT-convertible subgraphs become TRT engines
embedded in the package; the remaining ops (PyTorch fallback subgraphs) are compiled by
AOTInductor into native CUDA kernels. The result is a single ``.pt2`` file that runs
end-to-end without Python.

**When to use AOTInductor**

* Deploying to C++ servers without a Python environment.
* Shipping a single self-contained artifact that bundles both TRT engines and PyTorch ops.
* When you want inference-time independence from Torch-TensorRT.

.. note::

    AOTInductor packaging is currently **Linux-only**.

----

Compile and Save
-----------------

The workflow is identical to the standard ``ir="dynamo"`` path, with two extra arguments
to ``torch_tensorrt.save``:

* ``output_format="aot_inductor"`` — selects the ``.pt2`` packager.
* ``retrace=True`` — re-exports the compiled graph through ``torch.export`` before
  passing it to ``torch._inductor.aoti_compile_and_package``. Required when the compiled
  module contains TRT engine subgraphs.

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    example_inputs = (torch.randn(8, 10, device="cuda"),)

    # Step 1 — export with optional dynamic shapes
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    exported = torch.export.export(
        model, example_inputs, dynamic_shapes={"x": {0: batch_dim}}
    )

    # Step 2 — compile with Torch-TensorRT
    trt_gm = torch_tensorrt.dynamo.compile(
        exported,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 10),
                opt_shape=(8, 10),
                max_shape=(1024, 10),
                dtype=torch.float32,
            )
        ],
        min_block_size=1,
    )

    # Step 3 — package into a .pt2 file
    torch_tensorrt.save(
        trt_gm,
        "model.pt2",
        output_format="aot_inductor",
        retrace=True,
        arg_inputs=example_inputs,
    )

The ``.pt2`` file embeds both the TRT engine(s) and AOTInductor-compiled kernels for any
ops that fell back to PyTorch.

----

Python Inference
-----------------

Load the package with ``torch._inductor.aoti_load_package``. No Torch-TensorRT import
is needed at inference time:

.. code-block:: python

    import torch

    model = torch._inductor.aoti_load_package("model.pt2")

    # Works with any batch size within the compiled range
    output = model(torch.randn(4, 10, device="cuda"))
    output = model(torch.randn(16, 10, device="cuda"))

----

C++ Inference
--------------

The same ``.pt2`` package runs in C++ via ``AOTIModelPackageLoader``, with no Python
or Torch-TensorRT dependency:

.. code-block:: cpp

    #include "torch/torch.h"
    #include "torch/csrc/inductor/aoti_package/model_package_loader.h"

    int main() {
        c10::InferenceMode mode;

        torch::inductor::AOTIModelPackageLoader loader("model.pt2");

        // Batch size 8
        std::vector<torch::Tensor> inputs = {torch::randn({8, 10}, at::kCUDA)};
        auto outputs = loader.run(inputs);

        // Dynamic batch — different size works within the compiled min/max range
        outputs = loader.run({torch::randn({1, 10}, at::kCUDA)});

        return 0;
    }

.. note::

    At runtime, ``libtorchtrt_runtime.so`` is not needed. Ensure your link flags
    exclude it (or use ``--as-needed``) to avoid a spurious dependency.

----

Comparison: PT2 vs ExportedProgram
------------------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - ``.ep`` (ExportedProgram)
     - ``.pt2`` (AOTInductor)
   * - Python load
     - ``torch_tensorrt.load("trt.ep").module()``
     - ``torch._inductor.aoti_load_package("trt.pt2")``
   * - C++ load
     - Not supported
     - ``AOTIModelPackageLoader``
   * - Torch-TensorRT at runtime
     - Required
     - Not required
   * - Non-TRT ops
     - Run via PyTorch eager
     - Compiled by AOTInductor (native CUDA)
   * - Platform
     - Linux + Windows
     - Linux only

----

See ``examples/torchtrt_aoti_example/`` for a complete end-to-end runnable example
(``model.py`` for compilation, ``inference.py`` for loading).
