.. _saving_models:

Saving models compiled with Torch-TensorRT
====================================
.. currentmodule:: torch_tensorrt.dynamo

.. automodule:: torch_tensorrt.dynamo
   :members:
   :undoc-members:
   :show-inheritance:

Saving models compiled with Torch-TensorRT can be done using `torch_tensorrt.save` API.

Dynamo IR
-------------

The output type of `ir=dynamo` compilation of Torch-TensorRT is `torch.fx.GraphModule` object by default.
We can save this object in either `TorchScript` (`torch.jit.ScriptModule`), `ExportedProgram` (`torch.export.ExportedProgram`) or `PT2` formats by
specifying the `output_format` flag. Here are the options `output_format` will accept

* `exported_program` : This is the default. We perform transformations on the graphmodule first and use `torch.export.save` to save the module.
* `torchscript` : We trace the graphmodule via `torch.jit.trace` and save it via `torch.jit.save`.
* `PT2 Format` : This is a next generation runtime for PyTorch models, allowing them to run in Python and in C++

a) ExportedProgram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's an example usage

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    # trt_ep is a torch.fx.GraphModule object
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
    torch_tensorrt.save(trt_gm, "trt.ep", inputs=inputs)

    # Later, you can load it and run inference
    model = torch.export.load("trt.ep").module()
    model(*inputs)

b) Torchscript
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    # trt_gm is a torch.fx.GraphModule object
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)
    torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", arg_inputs=inputs)

    # Later, you can load it and run inference
    model = torch.jit.load("trt.ts").cuda()
    model(*inputs)


Torchscript IR
-------------

In Torch-TensorRT 1.X versions, the primary way to compile and run inference with Torch-TensorRT is using Torchscript IR.
For `ir=ts`, this behavior stays the same in 2.X versions as well.

.. code-block:: python

  import torch
  import torch_tensorrt

  model = MyModel().eval().cuda()
  inputs = [torch.randn((1, 3, 224, 224)).cuda()]
  trt_ts = torch_tensorrt.compile(model, ir="ts", arg_inputs=inputs) # Output is a ScriptModule object
  torch.jit.save(trt_ts, "trt_model.ts")

  # Later, you can load it and run inference
  model = torch.jit.load("trt_model.ts").cuda()
  model(*inputs)


Loading the models
--------------------

We can load torchscript or exported_program models using `torch.jit.load` and `torch.export.load` APIs from PyTorch directly.
Alternatively, we provide a light wrapper `torch_tensorrt.load(file_path)` which can load either of the above model types.

Here's an example usage

.. code-block:: python

    import torch
    import torch_tensorrt

    # file_path can be trt.ep or trt.ts file obtained via saving the model (refer to the above section)
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    model = torch_tensorrt.load(<file_path>).module()
    model(*inputs)

b) PT2 Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PT2 is a new format that allows models to be run outside of Python in the future. It utilizes `AOTInductor <https://docs.pytorch.org/docs/main/torch.compiler_aot_inductor.html>`_
to generate kernels for components that will not be run in TensorRT.

Here's an example on how to save and load Torch-TensorRT Module using AOTInductor in Python

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    # trt_ep is a torch.fx.GraphModule object
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
    torch_tensorrt.save(trt_gm, "trt.pt2", arg_inputs=inputs, output_format="aot_inductor", retrace=True)

    # Later, you can load it and run inference
    model = torch._inductor.aoti_load_package("trt.pt2")
    model(*inputs)
