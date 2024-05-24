.. _dynamic_shapes:

Dynamic shapes with Torch-TensorRT
====================================

By default, you can run a pytorch model with varied input shapes and the output shapes are determined eagerly.
However, Torch-TensorRT is an AOT compiler which requires some prior information about the input shapes to compile and optimize the model.
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

- aten_tracer.trace (which uses torch.export to trace the graph with the given inputs)

In the tracing phase, we use torch.export along with the constraints. In the case of
dynamic shaped inputs, the range can be provided to the tracing via constraints. Please
refer to this `docstring <https://github.com/pytorch/pytorch/blob/5dcee01c2b89f6bedeef9dd043fd8d6728286582/torch/export/__init__.py#L372-L434>`_
for detailed information on how to set constraints. In short, we create new inputs for
torch.export tracing and provide constraints on the min and max values(provided by the user), a particular dimension can take.
Please take a look at ``aten_tracer.py`` file to understand how this works under the hood.

- dynamo.compile (which compiles a torch.fx.GraphModule object using TensorRT)

In the conversion to TensorRT, we use the user provided dynamic shape inputs.
We perform shape analysis using dummy inputs (across min, opt and max shapes) and store the
intermediate output shapes which can be used in case the graph has a mix of Pytorch
and TensorRT submodules.

Custom Constraints
------------------

Given an input ``x = torch_tensorrt.Input(min_shape, opt_shape, max_shape, dtype)``,
Torch-TensorRT automatically sets the constraints during ``torch.export`` tracing as follows

.. code-block:: python

    for dim in constraint_dims:
        if min_shape[dim] > 1:
            constraints.append(min_shape[dim] <= dynamic_dim(trace_input, dim))
        if max_shape[dim] > 1:
            constraints.append(dynamic_dim(trace_input, dim) <= max_shape[dim])

Sometimes, we might need to set additional constraints and Torchdynamo errors out if we don't specify them.
For example, in the case of BERT model compilation, there are two inputs and a constraint has to be set involving the sequence length size of these two inputs.

.. code-block:: python

    constraints.append(dynamic_dim(trace_inputs[0], 0) == dynamic_dim(trace_inputs[1], 0))


If you have to provide any custom constraints to your model, the overall workflow for model compilation using ``ir=dynamo`` would involve a few steps.

.. code-block:: python

    import torch
    import torch_tensorrt
    from torch_tensorrt.dynamo.lowering import apply_lowering_passes, get_decompositions
    # Assume the model has two inputs
    model = MyModel()
    torch_input_1 = torch.randn((1, 14), dtype=torch.int32).cuda()
    torch_input_2 = torch.randn((1, 14), dtype=torch.int32).cuda()

    dynamic_inputs = [torch_tensorrt.Input(min_shape=[1, 14],
                        opt_shape=[4, 14],
                        max_shape=[8, 14],
                        dtype=torch.int32),
                      torch_tensorrt.Input(min_shape=[1, 14],
                        opt_shape=[4, 14],
                        max_shape=[8, 14],
                        dtype=torch.int32)]

    # Export the model with additional constraints
    constraints = []
    # The following constraints are automatically added by Torch-TensorRT in the
    # general case when you call torch_tensorrt.compile directly on MyModel()
    constraints.append(dynamic_dim(torch_input_1, 0) < 8)
    constraints.append(dynamic_dim(torch_input_2, 0) < 8)
    # This is an additional constraint as instructed by Torchdynamo
    constraints.append(dynamic_dim(torch_input_1, 0) == dynamic_dim(torch_input_2, 0))
    with unittest.mock.patch(
        "torch._export.DECOMP_TABLE", get_decompositions(experimental_decompositions)
    ):
        graph_module = export(
            model, (torch_input_1, torch_input_2), constraints=constraints
        ).module()

    # Use the dynamo.compile API
    trt_mod = torch_tensorrt.dynamo.compile(graph_module, inputs=dynamic_inputs, **compile_spec)

Limitations
-----------

If there are operations in the graph that use the dynamic dimension of the input, Pytorch
introduces ``torch.ops.aten.sym_size.int`` ops in the graph. Currently, we cannot handle these operators and
the compilation results in undefined behavior. We plan to add support for these operators and implement
robust support for shape tensors in the next release. Here is an example of the limitation described above

.. code-block:: python

    import torch
    import torch_tensorrt

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.avgpool(x)
            out = torch.flatten(x, 1)
            return out

    model = MyModel().eval().cuda()
    # Compile with dynamic shapes
    inputs = torch_tensorrt.Input(min_shape=(1, 512, 1, 1),
                         opt_shape=(4, 512, 1, 1),
                         max_shape=(8, 512, 1, 1),
                         dtype=torch.float32)
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs)


The traced graph of `MyModule()` looks as follows

.. code-block:: python

    Post export graph: graph():
    %arg0_1 : [num_users=2] = placeholder[target=arg0_1]
    %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%arg0_1, [-1, -2], True), kwargs = {})
    %sym_size : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%arg0_1, 0), kwargs = {})
    %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%mean, [%sym_size, 512]), kwargs = {})
    return (view,)


Here the ``%sym_size`` node captures the dynamic batch and uses it in the ``aten.view`` layer. This requires shape tensors support
which would be a part of our next release.

Workaround (BERT static compilation example)
------------------------------------------

In the case where you encounter the issues mentioned in the **Limitations** section,
you can compile the model (static mode) with max input size that can be provided. In the cases of smaller inputs,
we can pad them accordingly. This is only a workaround until we address the limitations.

.. code-block:: python

    import torch
    import torch_tensorrt
    from transformers.utils.fx import symbolic_trace as transformers_trace

    model = BertModel.from_pretrained("bert-base-uncased").cuda().eval()

    # Input sequence length is 20.
    input1 = torch.randint(0, 5, (1, 20), dtype=torch.int32).to("cuda")
    input2 = torch.randint(0, 5, (1, 20), dtype=torch.int32).to("cuda")

    model = transformers_trace(model, input_names=["input_ids", "attention_mask"]).eval().cuda()
    trt_mod = torch_tensorrt.compile(model, inputs=[input1, input2], **compile_spec)
    model_outputs = model(input, input2)

    # If you have a sequence of length 14, pad 6 zero tokens and run inference
    # or recompile for sequence length of 14.
    input1 = torch.randint(0, 5, (1, 14), dtype=torch.int32).to("cuda")
    input2 = torch.randint(0, 5, (1, 14), dtype=torch.int32).to("cuda")
    trt_mod = torch_tensorrt.compile(model, inputs=[input1, input2], **compile_spec)
    model_outputs = model(input, input2)


Dynamic shapes with ir=torch_compile
------------------------------------

``torch_tensorrt.compile(model, inputs, ir="torch_compile")`` returns a torch.compile boxed function with the backend
configured to Tensorrt. In the case of ``ir=torch_compile``, when the input size changes, Dynamo will trigger a recompilation
of the TensorRT engine automatically giving dynamic shape behavior similar to native PyTorch eager however with the cost of rebuilding
TRT engine. This limitation will be addressed in future versions of Torch-TensorRT.

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = torch.randn((1, 3, 224, 224), dtype=float32)
    trt_gm = torch_tensorrt.compile(model, ir="torch_compile", inputs)
    # Compilation happens when you call the model
    trt_gm(inputs)

    # Recompilation happens with modified batch size
    inputs_bs2 = torch.randn((2, 3, 224, 224), dtype=torch.float32)
    trt_gm = torch_tensorrt.compile(model, ir="torch_compile", inputs_bs2)
