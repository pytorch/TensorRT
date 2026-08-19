.. _python_runtime:

Python vs C++ runtime
=====================

Torch-TensorRT uses a single module type, :class:`~torch_tensorrt.runtime.TorchTensorRTModule`,
to run TensorRT engines inside PyTorch. The **execution path** (which code actually drives
TensorRT execution) is selected automatically:

* **C++ path** — ``torch.classes.tensorrt.Engine`` and ``torch.ops.tensorrt.execute_engine``.
  Used when the Torch-TensorRT C++ extension (``libtorchtrt`` / runtime ``.so``) is loaded:
  TorchScript-friendly, and integrates with the full C++ runtime stack.
* **Python path** — Internal ``TRTEngine`` (``torch_tensorrt.dynamo.runtime._TRTEngine``)
  plus ``tensorrt::execute_engine`` registered from Python when the C++ runtime is not
  available (use ``PYTHON_ONLY=1`` when building Torch-TensorRT). Useful for minimal installs and for Python-level debugging.

Both the C++ and Python paths are invoked through the same ``TorchTensorRTModule`` class,
which dispatches to the appropriate runtime engine based on the build of Torch-TensorRT (Full build or PYTHON_ONLY build).

----

When the Python runtime is used
-----------------------------

The Python engine implementation is chosen automatically when the C++ Torch-TensorRT library
is not installed (enabled by setting ``PYTHON_ONLY=1`` when building Torch-TensorRT). You may still prefer that setup when:

* You need to run on a machine where the C++ Torch-TensorRT library is not installed
  (e.g., a minimal CI container with only the Python wheel).
* You want to attach Python-level callbacks to the engine execution (via
  :ref:`observer`) for debugging or profiling without building the C++ extension.
* You are debugging a conversion issue and want to step through TRT execution in Python.

Use the default C++ runtime in all other cases, especially:

* When using CUDAGraphs for low-latency inference.
* In production deployments.
----

Compile and run
-----------------

Use ``torch_tensorrt.dynamo.compile``, ``torch.compile(..., backend="tensorrt", ...)``, or
construct :class:`~torch_tensorrt.runtime.TorchTensorRTModule` directly. The module picks C++
vs Python execution based on the build of Torch-TensorRT (Full build or Python-only build).

----

Serialization
---------------

``TorchTensorRTModule`` are serializable in both the C++ and Python paths.
.. code-block::python
    torch_tensorrt.save(trt_module, trt_ep_path, retrace=True)
    trt_module = torch_tensorrt.load(trt_ep_path).module()

Cross-serialization (Python and C++)
-------------------------------------

One of the key features of ``TorchTensorRTModule`` is seamless cross serialization: 
**you can serialize an engine using the Python runtime and load it using the C++ runtime, or vice versa**. 
The engine file format and all core metadata are fully compatible across runtimes and platforms, ensuring flexibility for production and development workflows.

For example, you can:

- **Build and serialize in Python**, then deploy by loading the module in a C++-enabled environment (e.g. in TorchScript or when the C++ extension is present):
  
  .. code-block:: python

      # In an environment with only Python runtime (PYTHON_ONLY=1)
      torch_tensorrt.save(trt_module, "trt_module.ep")

      # --- Later, or on a different machine with C++ runtime enabled ---
      trt_module = torch_tensorrt.load("trt_module.ep").module()
      output = trt_module(input)

- **Build in C++ runtime environment**, save the engine, and then load it in a Python-only deployment or debugging context, with no changes needed.

This interoperability allows you to train, compile, and debug using the Python path, 
but deploy for maximum performance using the C++ runtime—or test and profile using Python tools with modules built from C++.
**No extra conversion is required and the serialization format is shared across both backends.**

----

Limitations
-----------
* **CUDAGraphs**: Whole-graph CUDAGraphs work with the Python runtime, but the
  per-submodule CUDAGraph recording in ``CudaGraphsTorchTensorRTModule`` is
  only available with the C++ runtime.
----

``TorchTensorRTModule`` from raw engine bytes
---------------------------------------------

You can build a module directly from a serialized TensorRT engine (for example, an engine
produced outside Torch-TensorRT):

.. code-block:: python

    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
    from torch_tensorrt.dynamo._settings import CompilationSettings

    with open("model.engine", "rb") as f:
        engine_bytes = f.read()

    module = TorchTensorRTModule(
        serialized_engine=engine_bytes,
        input_binding_names=["x"],
        output_binding_names=["output"],
        name="my_engine",
        settings=CompilationSettings(),
    )

    output = module(torch.randn(1, 3, 224, 224).cuda())

**Constructor arguments** (see class docstring for full detail):

``serialized_engine`` (``bytes``)
    Raw serialized TRT engine.

``input_binding_names`` / ``output_binding_names`` (``List[str]``)
    TRT input binding names in the order they are passed to ``forward()``.

``output_binding_names`` (``List[str]``)
    TRT output binding names in the order they are returned from ``forward()``.

``name`` (``str``, optional)
    Name for logging and serialization.

``settings`` (:class:`~torch_tensorrt.dynamo._settings.CompilationSettings`, optional)
    Device and runtime options (must match how the engine was built).

``weight_name_map`` (``dict``, optional)
    Mapping of TRT weight names to PyTorch state dict names. Required for refit
    support via :func:`~torch_tensorrt.dynamo.refit_module_weights`.

``requires_output_allocator`` (``bool``, default ``False``)
    Set to ``True`` if the engine contains data-dependent-shape ops (``nonzero``,
    ``unique``, etc.) that require TRT's output allocator.

----

Runtime selection summary
-------------------------

* ``TorchTensorRTModule`` uses the C++ engine path when the Torch-TensorRT extension is loaded;
  otherwise it uses the Python ``TRTEngine`` path.
* If the C++ extension is **not** built, only the Python path is available.
* To use the Python runtime, set ``PYTHON_ONLY=1`` when building Torch-TensorRT.
