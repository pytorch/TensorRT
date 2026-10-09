.. _quick_start:

Quick Start
##################

Option 1: torch.compile
-------------------------

You can use Torch-TensorRT anywhere you use torch.compile:

.. code-block:: py

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda() # define your model here
    x = torch.randn((1, 3, 224, 224)).cuda() # define what the inputs to the model will look like

    optimized_model = torch.compile(model, backend="tensorrt")
    optimized_model(x) # compiled on first run

    optimized_model(x) # this will be fast!


Option 2: Export
-------------------------

If you want to optimize your model ahead-of-time and/or deploy in a C++ environment, Torch-TensorRT provides an export-style workflow that serializes an optimized module. This module can be deployed in PyTorch or with libtorch (i.e. without a Python dependency).

Step 1: Optimize + serialize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: py

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda() # define your model here
    inputs = [torch.randn((1, 3, 224, 224)).cuda()] # define a list of representative inputs here

    trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs)
    torch_tensorrt.save(trt_gm, "trt.ep", inputs=inputs) # PyTorch only supports Python runtime for an ExportedProgram. For C++ deployment, use a TorchScript file
    torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", inputs=inputs)

Step 2: Deploy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deployment in Python:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: py

    import torch
    import torch_tensorrt

    inputs = [torch.randn((1, 3, 224, 224)).cuda()] # your inputs go here

    # You can run this in a new python session!
    model = torch.export.load("trt.ep").module()
    # model = torch_tensorrt.load("trt.ep").module() # this also works
    model(*inputs)

Deployment in C++:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

    #include "torch/script.h"
    #include "torch_tensorrt/torch_tensorrt.h"

    auto trt_mod = torch::jit::load("trt.ts");
    auto input_tensor = [...]; // fill this with your inputs
    auto results = trt_mod.forward({input_tensor});