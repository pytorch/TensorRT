.. _python_runtime:

Python vs C++ runtime
=====================

Torch-TensorRT uses a single module type, :class:`~torch_tensorrt.runtime.TorchTensorRTModule`,
to run TensorRT engines inside PyTorch. The **execution path** (which code actually drives
TensorRT execution) is selected automatically:

* **C++ path** — ``torch.classes.tensorrt.Engine`` and ``torch.ops.tensorrt.execute_engine``.
  Used when the Torch-TensorRT C++ extension (``libtorchtrt`` / runtime ``.so``) is loaded:
  TorchScript-friendly, and integrates with the full C++ runtime stack.
* **Python path** — Internal ``TRTEngine`` (``torch_tensorrt.dynamo.runtime._PythonTRTEngine``)
  plus ``tensorrt::execute_engine`` registered from Python when the C++ runtime is not
  available. Useful for minimal installs and for Python-level debugging.

There is no separate subclass or API to **pin** only the Python path: the same
``TorchTensorRTModule`` class is used in both cases.

----

When the Python path is used
-----------------------------

The Python engine implementation is chosen automatically when the C++ Torch-TensorRT library
is not installed. You may still prefer that setup when:

* You deploy environments without the compiled Torch-TensorRT extension.
* You want easier Python-level debugging and instrumentation around TRT execution.

Prefer the C++ path when:

* You rely on the default Torch-TensorRT deployment story and maximum parity with TorchScript export.
* You use whole-graph CUDAGraph wrappers that assume the C++ runtime (see :ref:`cuda_graphs`).

----

Compile and run
-----------------

Use ``torch_tensorrt.dynamo.compile``, ``torch.compile(..., backend="tensorrt", ...)``, or
construct :class:`~torch_tensorrt.runtime.TorchTensorRTModule` directly. The module picks C++
vs Python execution based on extension availability.

----

Serialization
---------------

``TorchTensorRTModule`` records serialized state compatible with ``torch.save`` /
``get_extra_state`` / ``set_extra_state``. Some **export** workflows (e.g. certain
``ExportedProgram`` save paths) may still assume a C++-only graph; validate your deployment path
if you rely on portable artifacts.

----

Limitations
-----------

* **C++ deployment**: Artifacts produced or run on the Python path still need TensorRT and the
  Torch-TensorRT Python package in-process unless you recompile for the C++ path.
* **CUDAGraphs**: Whole-graph CUDAGraph wrappers may assume the C++ runtime for some configurations;
  see :ref:`cuda_graphs`.
* **Explicit allocator engines**: Engines with data-dependent outputs may set
  ``requires_output_allocator=True``; ``TorchTensorRTModule`` supports output-allocator execution
  on the Python path. See :ref:`cuda_graphs` for interaction with CUDA graphs.

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
    Binding names in ``forward`` order.

``name`` (``str``, optional)
    Name for logging and serialization.

``settings`` (:class:`~torch_tensorrt.dynamo._settings.CompilationSettings`, optional)
    Device and runtime options (must match how the engine was built).

``weight_name_map`` (``dict``, optional)
    For refit workflows; see :func:`~torch_tensorrt.dynamo.refit_module_weights`.

``requires_output_allocator`` (``bool``)
    Set ``True`` for data-dependent-shape ops that need TRT's output allocator.

----

Runtime selection summary
-------------------------

* ``TorchTensorRTModule`` uses the C++ engine path when the Torch-TensorRT extension is loaded;
  otherwise it uses the Python ``TRTEngine`` path.
* If the C++ extension is **not** built, only the Python path is available.
