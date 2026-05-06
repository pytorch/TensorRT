.. _serialized_engine:

Extracting a Raw TensorRT Engine
==================================

:func:`~torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine`
compiles an ``ExportedProgram`` directly to raw TensorRT engine bytes, bypassing the
PyTorch wrapper. The output is a ``bytes`` object that can be:

* Saved to a ``.engine`` file and loaded by ``trtexec`` or any TRT-native runtime.
* Embedded in a C++ application via ``nvinfer1::IRuntime::deserializeCudaEngine``.
* Deployed without any Python or PyTorch dependency at inference time.

Use this API when you need a **self-contained TRT engine** rather than a compiled
``torch.fx.GraphModule``. For normal PyTorch-integrated inference, prefer
:func:`~torch_tensorrt.dynamo.compile`.

.. note::

   This API compiles the **entire** exported program as a single TRT engine. It does
   not perform graph partitioning — if any operator in the graph is unsupported, the
   conversion will fail. Use ``require_full_compilation=True`` with
   :func:`~torch_tensorrt.dynamo.compile` first to verify full coverage.

----

Basic Usage
-----------

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn(1, 3, 224, 224).cuda()]

    exported = torch.export.export(model, tuple(inputs))

    engine_bytes: bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exported,
        arg_inputs=inputs,
    )

    # Save to disk
    with open("model.engine", "wb") as f:
        f.write(engine_bytes)

----

Dynamic Shapes
--------------

Pass ``torch_tensorrt.Input`` objects to specify min/opt/max shape ranges:

.. code-block:: python

    from torch_tensorrt import Input

    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exported,
        arg_inputs=[
            Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
            )
        ],
    )

----

Loading the Engine
------------------

The returned bytes can be loaded back by any TRT-compatible runtime:

.. code-block:: python

    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_bytes)

Or via Torch-TensorRT's own deserializer:

.. code-block:: python

    from torch_tensorrt.dynamo._refit import get_engine_from_encoded_engine
    import base64

    # The bytes can be base64-encoded for storage:
    encoded = base64.b64encode(engine_bytes).decode()
    engine = get_engine_from_encoded_engine(encoded)

----

Compared to ``compile()``
--------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * -
     - ``compile()``
     - ``convert_exported_program_to_serialized_trt_engine()``
   * - Output
     - ``torch.fx.GraphModule`` (PyTorch-callable)
     - ``bytes`` (raw TRT engine)
   * - Partial TRT coverage
     - Yes — unsupported ops fall back to PyTorch
     - No — full TRT required
   * - Serialization
     - ``torch_tensorrt.save()`` → ``.ep`` file
     - ``open(..., "wb").write(bytes)`` → ``.engine`` file
   * - PyTorch at runtime
     - Required
     - Not required
   * - Multiple inputs/outputs
     - Full support
     - Full support
   * - Graph partitioning
     - Yes
     - No (single engine)

----

Key Parameters
--------------

All :ref:`CompilationSettings <compilation_settings>` parameters are accepted. The most
relevant for this API:

``inputs`` / ``arg_inputs``
    Input shape specifications. Accepts ``torch.Tensor`` (static shape inferred),
    ``torch_tensorrt.Input`` (explicit static or dynamic ranges), or a mix.

``immutable_weights``
    Default ``True``. Set to ``False`` to produce a refittable engine whose weights
    can be updated without recompilation.

``optimization_level``
    Integer 0–5 controlling compile time vs. runtime performance trade-off.

``hardware_compatible``
    ``True`` to build an engine deployable on different Ampere+ GPU SKUs.

``version_compatible``
    ``True`` to build an engine forward-compatible with newer TRT releases.

**Returns** ``bytes`` — the serialized TRT engine. Save or pass directly to
``IRuntime::deserializeCudaEngine``.
