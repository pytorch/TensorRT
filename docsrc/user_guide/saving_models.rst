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

Starting with 2.1 release of Torch-TensorRT, we are switching the default compilation to be dynamo based.
The output of `ir=dynamo` compilation is a `torch.fx.GraphModule` object. There are two ways to save these objects

a) Converting to Torchscript
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`torch.fx.GraphModule` objects cannot be serialized directly. Hence we use `torch.jit.trace` to convert this into a `ScriptModule` object which can be saved to disk.
The following code illustrates this approach.

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs) # Output is a torch.fx.GraphModule
    trt_traced_model = torch.jit.trace(trt_gm, inputs)
    torch.jit.save(trt_traced_model, "trt_model.ts")

    # Later, you can load it and run inference
    model = torch.jit.load("trt_model.ts").cuda()
    model(*inputs)

b) ExportedProgram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`torch.export.ExportedProgram` is a new format introduced in Pytorch 2.1. After we compile a Pytorch module using Torch-TensorRT, the resultant 
`torch.fx.GraphModule` along with additional metadata can be used to create `ExportedProgram` which can be saved and loaded from disk.

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    exp_program = torch_tensorrt.dynamo.trace(model, inputs)
    trt_gm = torch_tensorrt.dynamo.compile(exp_program, inputs) # Output is a torch.fx.GraphModule
    # Transform and create an exported program
    trt_exp_program = torch_tensorrt.dynamo.export(trt_gm, inputs, exp_program.call_spec, ir="exported_program")
    torch.export.save(trt_exp_program, "trt_model.ep")

    # Later, you can load it and run inference 
    model = torch.export.load("trt_model.ep")
    model(*inputs)

`torch_tensorrt.dynamo.export` inlines the submodules within a GraphModule to their corresponding nodes, stiches all the nodes together and creates an ExportedProgram. 
This is needed as `torch.export` serialization cannot handle serializing and deserializing of submodules (`call_module` nodes). 

.. note:: This way of saving the models using `ExportedProgram` is experimental. Here is a known issue : https://github.com/pytorch/TensorRT/issues/2341


Torchscript IR
-------------

In Torch-TensorRT 1.X versions, the primary way to compile and run inference with Torch-TensorRT is using Torchscript IR.
This behavior stays the same in 2.X versions as well.

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

