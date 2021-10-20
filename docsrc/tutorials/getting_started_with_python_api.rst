.. _getting_started:

Using TRTorch in Python
************************

TRTorch Python API accepts a `torch.nn.Module` as an input. Under the hood, it uses `torch.jit.script` to convert the input module into a
TorchScript module. To compile your input `torch.nn.Module` with TRTorch embedded into Python, all you need to do is provide the module and a compile spec
to TRTorch and you will be returned an optimized TorchScript module to run or add into another PyTorch module. The
only required setting is a list of `trtorch.Input` classes which define input's shape, datatype and memory format. You can also specify settings such as
operating precision for the engine or target device. After compilation you can save the module just like any other module
to load in a deployment application. In order to load a TensorRT/TorchScript module, make sure you first import ``trtorch``.

.. code-block:: python

    import trtorch

    ...

    model = MyModel().eval() # torch module needs to be in eval (not training) mode

    compile_spec = {
        "inputs": [trtorch.Input(
                min_shape=[1, 1, 16, 16],
                opt_shape=[1, 1, 32, 32],
                max_shape=[1, 1, 64, 64],
                dtype=torch.half,
            ),
        ],
        "enabled_precisions": {torch.float, torch.half} # Run with fp16
    }

    trt_ts_module = trtorch.compile(model, compile_spec)

    input_data = input_data.to('cuda').half()
    result = trt_ts_module(input_data)
    torch.jit.save(trt_ts_module, "trt_ts_module.ts")

.. code-block:: python

    # Deployment application
    import torch
    import trtorch

    trt_ts_module = torch.jit.load("trt_ts_module.ts")
    input_data = input_data.to('cuda').half()
    result = trt_ts_module(input_data)

TRTorch python API also provides ``trtorch.compile_ts`` which accepts a TorchScript module as input.
The torchscript module can be obtained via scripting or tracing. ``trtorch.compile_ts`` accepts a Torchscript module along
with a compile spec similar to the one listed above.
