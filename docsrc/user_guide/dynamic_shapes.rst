.. _dynamic_shapes:

Dynamic shapes with Torch-TensorRT
====================================

By default, you can run a pytorch model with varied input shapes and the output shapes are determined eagerly.
However, Torch-TensorRT is an AOT compiler which requires some prior information about the input shapes to compile and optimize the model.

Dynamic shapes using torch.export (AOT)
------------------------------------

In the case of dynamic input shapes, we must provide the (min_shape, opt_shape, max_shape) arguments so that the model can be optimized for
this range of input shapes. An example usage of static and dynamic shapes is as follows.

NOTE: The following code uses Dynamo Frontend. In case of Torchscript Frontend, please swap out ``ir=dynamo`` with ``ir=ts`` and the behavior is exactly the same.

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    # Compile with static shapes
    inputs = torch_tensorrt.Input(shape=[1, 3, 224, 224], dtype=torch.float32)
    # or compile with dynamic shapes
    inputs = torch_tensorrt.Input(min_shape=[1, 3, 224, 224],
                                  opt_shape=[4, 3, 224, 224],
                                  max_shape=[8, 3, 224, 224],
                                  dtype=torch.float32)
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs)

Under the hood
--------------

There are two phases of compilation when we use ``torch_tensorrt.compile`` API with ``ir=dynamo`` (default).

- torch_tensorrt.dynamo.trace (which uses torch.export to trace the graph with the given inputs)

We use ``torch.export.export()`` API for tracing and exporting a PyTorch module into ``torch.export.ExportedProgram``. In the case of
dynamic shaped inputs, the ``(min_shape, opt_shape, max_shape)`` range provided via ``torch_tensorrt.Input`` API is used to construct ``torch.export.Dim`` objects
which is used in the ``dynamic_shapes`` argument for the export API.
Please take a look at ``_tracer.py`` file to understand how this works under the hood.

- torch_tensorrt.dynamo.compile (which compiles an torch.export.ExportedProgram object using TensorRT)

In the conversion to TensorRT, the graph already has the dynamic shape information in the node's metadata which will be used during engine building phase.

Custom Dynamic Shape Constraints
---------------------------------

Given an input ``x = torch_tensorrt.Input(min_shape, opt_shape, max_shape, dtype)``,
Torch-TensorRT attempts to automatically set the constraints during ``torch.export`` tracing by constructing
`torch.export.Dim` objects with the provided dynamic dimensions accordingly. Sometimes, we might need to set additional constraints and Torchdynamo errors out if we don't specify them.
If you have to set any custom constraints to your model (by using `torch.export.Dim`), we recommend exporting your program first before compiling with Torch-TensorRT.
Please refer to this `documentation <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html#constraints-dynamic-shapes>`_ to export the Pytorch module with dynamic shapes.
Here's a simple example that exports a matmul layer with some restrictions on dynamic dimensions.

.. code-block:: python

    import torch
    import torch_tensorrt

    class MatMul(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, query, key):
            attn_weight = torch.matmul(query, key.transpose(-1, -2))
            return attn_weight

    model = MatMul().eval().cuda()
    inputs = [torch.randn(1, 12, 7, 64).cuda(), torch.randn(1, 12, 7, 64).cuda()]
    seq_len = torch.export.Dim("seq_len", min=1, max=10)
    dynamic_shapes=({2: seq_len}, {2: seq_len})
    # Export the model first with custom dynamic shape constraints
    exp_program = torch.export.export(model, tuple(inputs), dynamic_shapes=dynamic_shapes)
    trt_gm = torch_tensorrt.dynamo.compile(exp_program, inputs)
    # Run inference
    trt_gm(*inputs)

Dynamic shapes using torch.compile (JIT)
------------------------------------

``torch_tensorrt.compile(model, inputs, ir="torch_compile")`` returns a torch.compile boxed function with the backend
configured to TensorRT. In the case of ``ir=torch_compile``, users can provide dynamic shape information for the inputs using ``torch._dynamo.mark_dynamic`` API (https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)
to avoid recompilation of TensorRT engines.

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = torch.randn((1, 3, 224, 224), dtype=float32)
    # This indicates the dimension 0 is dynamic and the range is [1, 8]
    torch._dynamo.mark_dynamic(inputs, 0, min=1, max=8)
    trt_gm = torch.compile(model, backend="tensorrt")
    # Compilation happens when you call the model
    trt_gm(inputs)

    # No recompilation of TRT engines with modified batch size
    inputs_bs2 = torch.randn((2, 3, 224, 224), dtype=torch.float32)
    trt_gm(inputs_bs2)


Saving and Loading Models with Dynamic Shapes
----------------------------------------------

When you compile a model with dynamic shapes and want to save it for later use, you need to preserve the dynamic shape
specifications. Torch-TensorRT provides two methods to accomplish this:

Method 1: Automatic Inference from torch_tensorrt.Input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest approach is to pass the same ``torch_tensorrt.Input`` objects (with min/opt/max shapes) to both ``compile()`` and ``save()``.
The dynamic shape specifications will be inferred automatically:

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()

    # Define Input with dynamic shapes once
    inputs = [
        torch_tensorrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(8, 3, 224, 224),
            max_shape=(32, 3, 224, 224),
            dtype=torch.float32,
            name="x"  # Optional: provides better dimension naming
        )
    ]

    # Compile with dynamic shapes
    trt_model = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)

    # Save - dynamic shapes inferred automatically!
    torch_tensorrt.save(trt_model, "model.ep", arg_inputs=inputs)

    # Load and use with different batch sizes
    loaded_model = torch_tensorrt.load("model.ep").module()
    output1 = loaded_model(torch.randn(4, 3, 224, 224).cuda())   # Works!
    output2 = loaded_model(torch.randn(16, 3, 224, 224).cuda())  # Works!


Method 2: Explicit torch.export.Dim Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For advanced use cases or when you need fine-grained control over dimension naming, you can explicitly provide ``dynamic_shapes``
using ``torch.export.Dim``:

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    example_input = torch.randn((2, 3, 224, 224)).cuda()

    # Define dynamic dimensions explicitly
    dyn_batch = torch.export.Dim("batch", min=1, max=32)
    dynamic_shapes = {"x": {0: dyn_batch}}

    # Export with dynamic shapes
    exp_program = torch.export.export(
        model, (example_input,),
        dynamic_shapes=dynamic_shapes,
        strict=False
    )

    # Compile
    trt_model = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=[torch_tensorrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(8, 3, 224, 224),
            max_shape=(32, 3, 224, 224),
        )]
    )

    # Save with explicit dynamic_shapes
    torch_tensorrt.save(
        trt_model,
        "model.ep",
        arg_inputs=[example_input],
        dynamic_shapes=dynamic_shapes  # Same as used during export
    )

    # Load and use
    loaded_model = torch_tensorrt.load("model.ep").module()

**When to use this method:**
  - You need specific dimension names for torch.export compatibility
  - You're working with existing torch.export workflows
  - You require fine-grained control over dynamic dimension specifications

Multiple Dynamic Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both methods support multiple dynamic dimensions (e.g., dynamic batch, height, and width):

.. code-block:: python

    # Method 1 (Automatic): Multiple dynamic dimensions
    inputs = [
        torch_tensorrt.Input(
            min_shape=(1, 3, 64, 64),
            opt_shape=(8, 3, 256, 256),
            max_shape=(16, 3, 512, 512),
            name="image"
        )
    ]

    trt_model = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
    torch_tensorrt.save(trt_model, "model.ep", arg_inputs=inputs)  # All 3 dims inferred!

    # Load and test with various sizes
    loaded = torch_tensorrt.load("model.ep").module()
    loaded(torch.randn(4, 3, 128, 128).cuda())
    loaded(torch.randn(12, 3, 384, 384).cuda())

Saving with Keyword Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your model uses keyword arguments with dynamic shapes, both methods support them:

.. code-block:: python

    # Define dynamic inputs for both args and kwargs
    arg_inputs = [
        torch_tensorrt.Input(
            min_shape=(1, 10),
            opt_shape=(4, 10),
            max_shape=(8, 10),
            name="x"
        )
    ]

    kwarg_inputs = {
        "mask": torch_tensorrt.Input(
            min_shape=(1, 5),
            opt_shape=(4, 5),
            max_shape=(8, 5),
            name="mask"
        )
    }

    # Compile
    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        arg_inputs=arg_inputs,
        kwarg_inputs=kwarg_inputs
    )

    # Save - both arg and kwarg dynamic shapes inferred automatically
    torch_tensorrt.save(
        trt_model,
        "model.ep",
        arg_inputs=arg_inputs,
        kwarg_inputs=kwarg_inputs
    )
