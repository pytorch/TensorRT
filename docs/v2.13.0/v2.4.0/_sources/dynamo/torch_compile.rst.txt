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

Customizeable Settings
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

.. note:: Quantization/INT8 support is slated for a future release; currently, we support FP16 and FP32 precision layers.

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
Compilation can also be helpful in demonstrating graph breaks and the feasibility of serialization of a particular model. For instance, if a model has no graph breaks and compiles successfully with the Torch-TensorRT backend, then that model should be compileable and serializeable via the `torch_tensorrt` Dynamo IR, as discussed in :ref:`dynamic_shapes`. To determine the number of graph breaks in a model, the `torch._dynamo.explain` function is very useful:

.. code-block:: python

    import torch
    import torch_tensorrt
    ...
    explanation = torch._dynamo.explain(model)(*inputs)
    print(f"Graph breaks: {explanation.graph_break_count}")
    optimized_model = torch.compile(model, backend="torch_tensorrt", dynamic=False, options={"truncate_long_and_double": True})

Dynamic Shape Support
-----------------

The Torch-TensorRT `torch.compile` backend will currently require recompilation for each new batch size encountered, and it is preferred to use the `dynamic=False` argument when compiling with this backend. Full dynamic shape support is planned for a future release.

Recompilation Conditions
-----------------

Once the model has been compiled, subsequent inference inputs with the same shape and data type, which traverse the graph in the same way will not require recompilation. Furthermore, each new recompilation will be cached for the duration of the Python session. For instance, if inputs of batch size 4 and 8 are provided to the model, causing two recompilations, no further recompilation would be necessary for future inputs with those batch sizes during inference within the same session. Support for engine cache serialization is planned for a future release.

Recompilation is generally triggered by one of two events: encountering inputs of different sizes or inputs which traverse the model code differently. The latter scenario can occur when the model code includes conditional logic, complex loops, or data-dependent-shapes. `torch.compile` handles guarding in both of these scenario and determines when recompilation is necessary.
