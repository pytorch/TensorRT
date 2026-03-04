.. _torch_compile:

TensorRT Backend for ``torch.compile``
======================================================
.. currentmodule:: torch_tensorrt.dynamo

.. automodule:: torch_tensorrt.dynamo
   :members:
   :undoc-members:
   :show-inheritance:

This guide presents the Torch-TensorRT `torch.compile` backend: a deep learning compiler which uses TensorRT to accelerate JIT-style workflows across a wide variety of models.

Key Features
--------------------------------------------

The primary goal of the Torch-TensorRT `torch.compile` backend is to enable Just-In-Time compilation workflows by combining the simplicity of `torch.compile` API with the performance of TensorRT. Invoking the `torch.compile` backend is as simple as importing the `torch_tensorrt` package and specifying the backend:

.. code-block:: python

    import torch_tensorrt
    ...
    optimized_model = torch.compile(model, backend="torch_tensorrt", dynamic=False)

.. note:: Many additional customization options are available to the user. These will be discussed in further depth in this guide.

The backend can handle a variety of challenging model structures and offers a simple-to-use interface for effective acceleration of models. Additionally, it has many customization options to ensure the compilation process is fitting to the specific use case.

Customizable Settings
-----------------
.. autoclass:: CompilationSettings

Custom Setting Usage
^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch_tensorrt
    ...
    optimized_model = torch.compile(model, backend="torch_tensorrt", dynamic=False,
                                    options={"truncate_long_and_double": True,
                                             "enabled_precisions": {torch.float, torch.half},
                                             "debug": True,
                                             "min_block_size": 2,
                                             "torch_executed_ops": {"torch.ops.aten.sub.Tensor"},
                                             "optimization_level": 4,
                                             "use_python_runtime": False,})

.. note:: Torch-TensorRT supports FP32, FP16, and INT8 precision layers. For INT8 quantization, use the TensorRT Model Optimizer (modelopt) for post-training quantization (PTQ). See :ref:`vgg16_ptq` for an example.

Advanced Precision Control
^^^^^^^^^^^^^^^^^

For fine-grained control over mixed precision execution, TensorRT 10.12+ provides additional settings:

* ``use_explicit_typing``: Enable explicit type specification (required for TensorRT 10.12+)
* ``enable_autocast``: Enable rule-based autocast for automatic precision selection
* ``autocast_low_precision_type``: Target precision for autocast (e.g., ``torch.float16``)
* ``autocast_excluded_nodes``: Specific nodes to exclude from autocast
* ``autocast_excluded_ops``: Operation types to exclude from autocast

For detailed information and examples, see :ref:`mixed_precision`.

Compilation
-----------------
Compilation is triggered by passing inputs to the model, as so:

.. code-block:: python

    import torch_tensorrt
    ...
    # Causes model compilation to occur
    first_outputs = optimized_model(*inputs)

    # Subsequent inference runs with the same, or similar inputs will not cause recompilation
    # For a full discussion of this, see "Recompilation Conditions" below
    second_outputs = optimized_model(*inputs)

After Compilation
-----------------
The compilation object can be used for inference within the Python session, and will recompile according to the recompilation conditions detailed below. In addition to general inference, the compilation process can be a helpful tool in determining model performance, current operator coverage, and feasibility of serialization. Each of these points will be covered in detail below.

Model Performance
^^^^^^^^^^^^^^^^^
The optimized model returned from `torch.compile` is useful for model benchmarking since it can automatically handle changes in the compilation context, or differing inputs that could require recompilation. When benchmarking inputs of varying distributions, batch sizes, or other criteria, this can save time.

Operator Coverage
^^^^^^^^^^^^^^^^^
Compilation is also a useful tool in determining operator coverage for a particular model. For instance, the following compilation command will display the operator coverage for each graph, but will not compile the model - effectively providing a "dryrun" mechanism:

.. code-block:: python

    import torch_tensorrt
    ...
    optimized_model = torch.compile(model, backend="torch_tensorrt", dynamic=False,
                                    options={"debug": True,
                                             "min_block_size": float("inf"),})

If key operators for your model are unsupported, see :ref:`dynamo_conversion` to contribute your own converters, or file an issue here: https://github.com/pytorch/TensorRT/issues.

Feasibility of Serialization
^^^^^^^^^^^^^^^^^
Compilation can also be helpful in demonstrating graph breaks and the feasibility of serialization of a particular model. For instance, if a model has no graph breaks and compiles successfully with the Torch-TensorRT backend, then that model should be compilable and serializeable via the `torch_tensorrt` Dynamo IR, as discussed in :ref:`dynamic_shapes`. To determine the number of graph breaks in a model, the `torch._dynamo.explain` function is very useful:

.. code-block:: python

    import torch
    import torch_tensorrt
    ...
    explanation = torch._dynamo.explain(model)(*inputs)
    print(f"Graph breaks: {explanation.graph_break_count}")
    optimized_model = torch.compile(model, backend="torch_tensorrt", dynamic=False, options={"truncate_long_and_double": True})

Engine Caching
^^^^^^^^^^^^^^^^^
Engine caching can significantly reduce recompilation times by saving built TensorRT engines to disk and reusing them when possible. This is particularly useful for JIT workflows where graphs may be invalidated and recompiled. When enabled, engines are saved with a hash of their corresponding PyTorch subgraph and can be reloaded in subsequent compilations—even across different Python sessions.

To enable engine caching, use the ``cache_built_engines`` and ``reuse_cached_engines`` options:

.. code-block:: python

    import torch_tensorrt
    ...
    optimized_model = torch.compile(model, backend="torch_tensorrt", dynamic=False,
                                    options={"cache_built_engines": True,
                                             "reuse_cached_engines": True,
                                             "immutable_weights": False,
                                             "engine_cache_dir": "/tmp/torch_trt_cache",
                                             "engine_cache_size": 1 << 30})  # 1GB

.. note:: To use engine caching, ``immutable_weights`` must be set to ``False`` to allow engine refitting. When a cached engine is loaded, weights are refitted rather than rebuilding the entire engine, which can reduce compilation times by orders of magnitude.

For more details and examples, see :ref:`engine_caching_example`.

Dynamic Shape Support
-----------------

The Torch-TensorRT `torch.compile` backend now supports dynamic shapes, allowing models to handle varying input dimensions without recompilation. You can specify dynamic dimensions using the ``torch._dynamo.mark_dynamic`` API:

.. code-block:: python

    import torch
    import torch_tensorrt
    ...
    inputs = torch.randn((1, 3, 224, 224), dtype=torch.float32).cuda()
    # Mark dimension 0 (batch) as dynamic with range [1, 8]
    torch._dynamo.mark_dynamic(inputs, 0, min=1, max=8)
    optimized_model = torch.compile(model, backend="tensorrt")
    optimized_model(inputs)  # First compilation

    # No recompilation with different batch size in the dynamic range
    inputs_bs4 = torch.randn((4, 3, 224, 224), dtype=torch.float32).cuda()
    optimized_model(inputs_bs4)

Without dynamic shapes, the model will recompile for each new input shape encountered. For more control over dynamic shapes, consider using the AOT compilation path with ``torch_tensorrt.compile`` as described in :ref:`dynamic_shapes`. For a complete tutorial on dynamic shape compilation, see :ref:`compile_with_dynamic_inputs`.

Recompilation Conditions
-----------------

Once the model has been compiled, subsequent inference inputs with the same shape and data type, which traverse the graph in the same way will not require recompilation. Furthermore, each new recompilation will be cached for the duration of the Python session. For instance, if inputs of batch size 4 and 8 are provided to the model, causing two recompilations, no further recompilation would be necessary for future inputs with those batch sizes during inference within the same session.

To persist engine caches across Python sessions, use the ``cache_built_engines`` and ``reuse_cached_engines`` options as described in the Engine Caching section above.

Recompilation is generally triggered by one of two events: encountering inputs of different sizes or inputs which traverse the model code differently. The latter scenario can occur when the model code includes conditional logic, complex loops, or data-dependent-shapes. `torch.compile` handles guarding in both of these scenario and determines when recompilation is necessary.
