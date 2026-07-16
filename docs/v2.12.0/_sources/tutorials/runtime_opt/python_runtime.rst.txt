.. _python_runtime:

Python Runtime
==============

Torch-TensorRT provides two runtime backends for executing compiled TRT engines
inside a PyTorch graph:

* **C++ runtime** (default) — ``TorchTensorRTModule`` backed by a C++ TorchBind class.
  Fully serializable, supports CUDAGraphs, multi-device safe.
* **Python runtime** — ``PythonTorchTensorRTModule`` backed entirely by the TRT Python
  API. Simpler to instrument for debugging but **not serializable** to
  ``ExportedProgram``.

----

When to Use the Python Runtime
--------------------------------

Use ``use_python_runtime=True`` when:

* You need to run on a machine where the C++ Torch-TensorRT library is not installed
  (e.g., a minimal CI container with only the Python wheel).
* You want to attach Python-level callbacks to the engine execution (via
  :ref:`observer`) for debugging or profiling without building the C++ extension.
* You are debugging a conversion issue and want to step through TRT execution in Python.

Use the default C++ runtime in all other cases, especially:

* When saving a compiled module to disk (``torch_tensorrt.save()``).
* When using CUDAGraphs for low-latency inference.
* In production deployments.

----

Enabling the Python Runtime
-----------------------------

.. code-block:: python

    import torch_tensorrt

    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=inputs,
        use_python_runtime=True,
    )

Or via ``torch.compile``:

.. code-block:: python

    trt_model = torch.compile(
        model,
        backend="tensorrt",
        options={"use_python_runtime": True},
    )

----

Limitations
-----------

* **Not serializable**: ``PythonTorchTensorRTModule`` cannot be saved via
  ``torch_tensorrt.save()`` as an ``ExportedProgram`` or loaded back. The module is
  Python-only in-process.

  .. code-block:: python

      # This will raise an error with use_python_runtime=True:
      torch_tensorrt.save(trt_gm, "model.ep", arg_inputs=inputs)

* **No C++ deployment**: The compiled module cannot be exported to AOTInductor or used
  in a C++ application without re-compiling with the C++ runtime.

* **CUDAGraphs**: Whole-graph CUDAGraphs work with the Python runtime, but the
  per-submodule CUDAGraph recording in ``CudaGraphsTorchTensorRTModule`` is
  only available with the C++ runtime.

----

``PythonTorchTensorRTModule`` Direct Instantiation
----------------------------------------------------

You can instantiate ``PythonTorchTensorRTModule`` directly from raw engine bytes,
for example when integrating a TRT engine built outside of Torch-TensorRT:

.. code-block:: python

    from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule
    from torch_tensorrt.dynamo._settings import CompilationSettings

    # Load raw engine bytes (e.g., from trtexec output or torch_tensorrt.dynamo.convert_*)
    with open("model.engine", "rb") as f:
        engine_bytes = f.read()

    module = PythonTorchTensorRTModule(
        serialized_engine=engine_bytes,
        input_binding_names=["x"],
        output_binding_names=["output"],
        name="my_engine",
        settings=CompilationSettings(),
    )

    output = module(torch.randn(1, 3, 224, 224).cuda())

**Constructor arguments:**

``serialized_engine`` (``bytes``)
    The raw serialized TRT engine bytes.

``input_binding_names`` (``List[str]``)
    TRT input binding names in the order they are passed to ``forward()``.

``output_binding_names`` (``List[str]``)
    TRT output binding names in the order they should be returned.

``name`` (``str``, optional)
    Human-readable name for the module (used in logging).

``settings`` (``CompilationSettings``, optional)
    The compilation settings used to build the engine. Used to determine device
    placement and other runtime behaviors.

``weight_name_map`` (``dict``, optional)
    Mapping of TRT weight names to PyTorch state dict names. Required for refit
    support via :func:`~torch_tensorrt.dynamo.refit_module_weights`.

``requires_output_allocator`` (``bool``, default ``False``)
    Set to ``True`` if the engine contains data-dependent-shape ops (``nonzero``,
    ``unique``, etc.) that require TRT's output allocator.

----

Runtime Selection Logic
------------------------

When ``use_python_runtime`` is ``None`` (auto-select), Torch-TensorRT tries to import
the C++ TorchBind class. If the C++ extension is not available it silently falls back to
the Python runtime. Pass ``True`` or ``False`` to force a specific runtime.
