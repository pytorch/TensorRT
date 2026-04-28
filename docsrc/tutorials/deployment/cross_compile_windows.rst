.. _cross_compile_windows:

Cross-Compiling for Windows
============================

:func:`torch_tensorrt.dynamo.cross_compile_for_windows` compiles TRT engines on a
**Linux x86-64** host and produces an ``ExportedProgram`` containing engines that can
be loaded and executed on **Windows x86-64** — without requiring a Linux GPU at
inference time.

This is the standard path for teams that build models on Linux (where TRT tooling is
more mature) and deploy on Windows (game engines, desktop applications, enterprise
software).

----

Requirements
------------

* **Build machine**: Linux x86-64 with CUDA and TensorRT installed.
* **Target machine**: Windows x86-64 with a compatible NVIDIA GPU (same or newer
  CUDA compute capability).
* ``enable_cross_compile_for_windows=True`` is automatically set by this API;
  do not set it manually on ``compile()``.

The following features are **disabled** during cross-compilation (they are not
available in the Windows TRT runtime or require OS-specific binaries):

* Python runtime (``use_python_runtime`` is forced to ``False``)
* Lazy engine initialization (``lazy_engine_init`` is forced to ``False``)
* Engine caching (``cache_built_engines`` / ``reuse_cached_engines`` disabled)

----

Workflow
--------

**Step 1 — Export on the Linux build machine**

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn(1, 3, 224, 224).cuda()]

    # Export to ExportedProgram
    exp_program = torch.export.export(model, tuple(inputs))

**Step 2 — Cross-compile for Windows**

.. code-block:: python

    trt_gm = torch_tensorrt.dynamo.cross_compile_for_windows(
        exp_program,
        arg_inputs=inputs,
    )

**Step 3 — Save the compiled module**

.. code-block:: python

    torch_tensorrt.save(trt_gm, "model_windows.ep", arg_inputs=inputs)

**Step 4 — Load and run on Windows**

Copy ``model_windows.ep`` to the Windows machine. Ensure
``libtorchtrt_runtime.so`` / ``torchtrt_runtime.dll`` is on the library path.

.. code-block:: python

    # On Windows:
    import torch_tensorrt
    trt_gm = torch_tensorrt.load("model_windows.ep").module()
    output = trt_gm(*inputs)

----

Dynamic Shapes
--------------

Dynamic shapes work the same as in normal ``compile()``:

.. code-block:: python

    from torch_tensorrt import Input

    trt_gm = torch_tensorrt.dynamo.cross_compile_for_windows(
        exp_program,
        arg_inputs=[
            Input(
                min_shape=(1,  3, 224, 224),
                opt_shape=(4,  3, 224, 224),
                max_shape=(16, 3, 224, 224),
            )
        ],
    )

----

Engine Compatibility
---------------------

The produced engines are compatible with the **same or newer CUDA compute capability**
as the GPU used during compilation. Use ``hardware_compatible=True`` if the Windows
deployment GPU may have a different architecture within the Ampere+ generation:

.. code-block:: python

    trt_gm = torch_tensorrt.dynamo.cross_compile_for_windows(
        exp_program,
        arg_inputs=inputs,
        hardware_compatible=True,  # engine runs on Ampere and newer
    )

----

Saving and Loading Cross-Compiled Programs
--------------------------------------------

The output of ``cross_compile_for_windows`` is a standard ``torch.fx.GraphModule``
containing ``TorchTensorRTModule`` submodules with Windows-compatible engine bytes.
Save and load via the standard Torch-TensorRT save/load API:

.. code-block:: python

    # Save (Linux)
    torch_tensorrt.save(trt_gm, "model_windows.ep", arg_inputs=inputs)

    # Load (Windows)
    trt_gm = torch_tensorrt.load("model_windows.ep").module()
    trt_gm(*inputs)

Alternatively, save as a raw ``.engine`` file for direct TRT deployment:

.. code-block:: python

    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exp_program,
        arg_inputs=inputs,
        enable_cross_compile_for_windows=False,  # use cross_compile_for_windows() instead
    )
    # Note: use cross_compile_for_windows() for the full workflow;
    # convert_exported_program_to_serialized_trt_engine() does not support cross-compilation.

----

Troubleshooting
---------------

``AssertionError: cross_compile_for_windows is only supported on Linux x86-64``
    You must run the compilation step on a Linux x86-64 machine. The ``@needs_cross_compile``
    decorator gates this function.

Engine fails to load on Windows
    Ensure the TRT version on Windows is ≥ the version used on Linux. Use
    ``version_compatible=True`` for forward compatibility within a TRT major version.

Output mismatch between Linux and Windows
    Floating-point results may differ slightly due to different driver/hardware
    implementations. Use ``optimization_level=0`` on Linux to minimize kernel
    specialization and improve cross-platform reproducibility.
