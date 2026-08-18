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
