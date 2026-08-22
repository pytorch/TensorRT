.. _getting_started_with_python_api:

Using Torch-TensorRT in Python
*******************************

Torch-TensorRT Python API accepts a ```torch.nn.Module`` as an input. Under the hood, it uses ``torch.jit.script`` to convert the input module into a
TorchScript module. To compile your input ```torch.nn.Module`` with Torch-TensorRT, all you need to do is provide the module and inputs
to Torch-TensorRT and you will be returned an optimized TorchScript module to run or add into another PyTorch module. Inputs
is a list of ``torch_tensorrt.Input`` classes which define input's shape, datatype and memory format. You can also specify settings such as
operating precision for the engine or target device. After compilation you can save the module just like any other module
to load in a deployment application. In order to load a TensorRT/TorchScript module, make sure you first import ``torch_tensorrt``.

.. code-block:: python

    import torch_tensorrt

    ...

    model = MyModel().eval() # torch module needs to be in eval (not training) mode

    inputs = [torch_tensorrt.Input(
                min_shape=[1, 1, 16, 16],
                opt_shape=[1, 1, 32, 32],
                max_shape=[1, 1, 64, 64],
                dtype=torch.half,
            )]
    enabled_precisions = {torch.float, torch.half} # Run with fp16

    trt_ts_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions=enabled_precisions)

    input_data = input_data.to('cuda').half()
    result = trt_ts_module(input_data)
    torch.jit.save(trt_ts_module, "trt_ts_module.ts")

.. code-block:: python

    # Deployment application
    import torch
    import torch_tensorrt

    trt_ts_module = torch.jit.load("trt_ts_module.ts")
    input_data = input_data.to('cuda').half()
    result = trt_ts_module(input_data)

Torch-TensorRT python API also provides ``torch_tensorrt.ts.compile`` which accepts a TorchScript module as input.
The torchscript module can be obtained via scripting or tracing (refer to :ref:`creating_torchscript_module_in_python`). ``torch_tensorrt.ts.compile`` accepts a Torchscript module
and a list of ``torch_tensorrt.Input`` classes.
