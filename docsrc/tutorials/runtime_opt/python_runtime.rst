.. _python_runtime:

Python vs C++ runtime
=====================

Torch-TensorRT uses a single module type, :class:`~torch_tensorrt.runtime.TorchTensorRTModule`,
to run TensorRT engines inside PyTorch. The **execution path** (which code actually drives
``execute_async``) is selected at runtime:

* **C++ path (default)** — ``torch.classes.tensorrt.Engine`` and ``torch.ops.tensorrt.execute_engine``.
  Preferred for production when the Torch-TensorRT C++ extension is available: TorchScript-friendly,
  and integrates with the full C++ runtime stack.
* **Python path** — When the C++ runtime is absent, use the internal ``TRTEngine`` plus
  ``torch.ops.tensorrt.execute_engine`` (registered from Python when the C++ runtime is absent). Useful when the C++ extension is absent, or when
  you want easier Python-level debugging and instrumentation.

:class:`~torch_tensorrt.runtime.PythonTorchTensorRTModule` is a **thin subclass** of
``TorchTensorRTModule`` that **pins** the Python path (same constructor and behavior, but always
resolves to the Python engine). Prefer ``TorchTensorRTModule`` plus the global backend APIs below
when you do not need that pin.

----

When to use the Python path
---------------------------

Use :func:`~torch_tensorrt.runtime.set_runtime_backend` (typically as a context manager) when:

* The C++ Torch-TensorRT library is not installed (e.g. a minimal environment with only the Python pieces).
* You want Python-level hooks (e.g. :ref:`observer`) without relying on the C++ extension.
* You are debugging conversion or execution and want to break inside the Python TRT wrapper.

Prefer the C++ path when:

* You rely on the default Torch-TensorRT deployment story and maximum parity with TorchScript export.
* You use whole-graph CUDAGraph wrappers that assume the C++ runtime (see :ref:`cuda_graphs`).

----

Enabling the Python path
------------------------

**Process-wide default (context manager)**

.. code-block:: python

    import torch_tensorrt as tt

    with tt.runtime.set_runtime_backend("python"):
        trt_gm = tt.dynamo.compile(exported_program, inputs)

**``torch.compile``** (same context manager around compile / first run)

.. code-block:: python

    import torch_tensorrt as tt

    with tt.runtime.set_runtime_backend("python"):
        trt_model = torch.compile(model, backend="tensorrt", options={})

The context manager does **not** replace :class:`~torch_tensorrt.runtime.PythonTorchTensorRTModule`,
which always requests the Python path via a class-level pin.

----

Serialization
---------------

Module state records which backend was used (``runtime_backend`` in packed metadata). After load,
``TorchTensorRTModule`` reconstructs either the C++ engine or the Python engine wrapper
as appropriate. Some **export** workflows (e.g. certain ``ExportedProgram`` save paths) may still
assume a C++-only graph; validate your deployment path if you mix Python execution with AOT export.

----

Limitations
-----------

* **C++ deployment**: A module that executed on the Python path still needs TensorRT and the
  Torch-TensorRT Python pieces available in-process unless you recompile targeting the C++ path.
* **CUDAGraphs**: Whole-graph CUDAGraph wrappers may assume the C++ runtime for some configurations;
  see :ref:`cuda_graphs`.
* **Explicit allocator engines**: Engines with data-dependent outputs may set
  ``requires_output_allocator=True``; the unified module supports the output-allocator execution
  mode on the Python path. See :ref:`cuda_graphs` for interaction with CUDA graphs.

----

``PythonTorchTensorRTModule`` direct instantiation
--------------------------------------------------

You can instantiate :class:`~torch_tensorrt.runtime.PythonTorchTensorRTModule` from raw engine bytes
when you need a **guaranteed** Python execution path (e.g. integrating an engine built outside
Torch-TensorRT):

.. code-block:: python

    from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule
    from torch_tensorrt.dynamo._settings import CompilationSettings

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

**Constructor arguments** (same as ``TorchTensorRTModule``):

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

* :func:`~torch_tensorrt.runtime.get_runtime_backend` / :func:`~torch_tensorrt.runtime.set_runtime_backend`
  — process default for newly created ``TorchTensorRTModule`` instances (unless a subclass pins a backend).
  Use ``set_runtime_backend`` as a context manager to scope C++ vs Python for compile and forward.
* If the C++ extension is **not** built, only the Python path is available.
