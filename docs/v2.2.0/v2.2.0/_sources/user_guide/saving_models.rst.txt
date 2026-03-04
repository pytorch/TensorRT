.. _saving_models:

Saving models compiled with Torch-TensorRT
====================================
.. currentmodule:: torch_tensorrt.dynamo

.. automodule:: torch_tensorrt.dynamo
   :members:
   :undoc-members:
   :show-inheritance:

Saving models compiled with Torch-TensorRT varies slightly with the `ir` that has been used for compilation.

Dynamo IR
-------------

The output type of `ir=dynamo` compilation of Torch-TensorRT is `torch.export.ExportedProgram` object by default. 
In addition, we provide a new parameter `output_format` in the `CompilationSetting` object provided before compilation.
The `output_format` can take the following options 

* `exported_program` (or) `ep` : This is the default. Returns an ExportedProgram 
* `torchscript` (or) `ts` : This returns a TorchScript module
* `graph_module` (or) `fx` : This returns a torch.fx.GraphModule which can be traced into Torchscript to save to disk.

a) Torchscript
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you set the `output_format="torchscript"`, this will return a `ScriptModule` which can be serialized via torch.jit.save

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    # trt_ts is a torch.jit.ScriptModule object
    trt_ts = torch_tensorrt.compile(model, ir="dynamo", inputs, output_format="torchscript")
    torch.jit.save(trt_ts, "trt_model.ts")

    # Later, you can load it and run inference
    model = torch.jit.load("trt_model.ts").cuda()
    model(*inputs)

b) ExportedProgram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`torch.export.ExportedProgram`, a new format introduced in Pytorch 2.X is the default return type of Torch-TensorRT compilation.

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    # trt_ep is a torch.export.ExportedProgram object
    trt_ep = torch_tensorrt.compile(model, ir="dynamo", inputs) 
    torch.export.save(trt_ep, "trt_model.ep")

    # Later, you can load it and run inference
    model = torch.export.load("trt_model.ep")
    model(*inputs)

c) GraphModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can also return a `torch.fx.GraphModule` object as the output of Torch-TensorRT compilation by setting `output_format="graph_module"`.
Internally, partitioning, lowering, conversion phases operate using GraphModule objects. These can be either traced into a Torchscript modules or 
exported into `ExportedProgram` objects

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    # trt_gm is a torch.fx.GraphModule object
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs, output_format="graph_module") 

Torchscript IR
-------------

In Torch-TensorRT 1.X versions, the primary way to compile and run inference with Torch-TensorRT is using Torchscript IR.
For `ir=ts`, this behavior stays the same in 2.X versions as well.

.. code-block:: python

  import torch
  import torch_tensorrt

  model = MyModel().eval().cuda()
  inputs = [torch.randn((1, 3, 224, 224)).cuda()]
  trt_ts = torch_tensorrt.compile(model, ir="ts", inputs) # Output is a ScriptModule object
  torch.jit.save(trt_ts, "trt_model.ts")

  # Later, you can load it and run inference
  model = torch.jit.load("trt_model.ts").cuda()
  model(*inputs)

