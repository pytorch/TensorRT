.. _getting_started_with_python_api:

Using Torch-TensorRT in Python
*******************************

The Torch-TensorRT Python API supports a number of unique usecases compared to the CLI and C++ APIs which solely support TorchScript compilation.

Torch-TensorRT Python API can accept a ``torch.nn.Module``, ``torch.jit.ScriptModule``, or ``torch.fx.GraphModule`` as an input.
Depending on what is provided one of the two frontends (TorchScript or FX) will be selected to compile the module. Provided the
module type is supported, users may explicitly set which frontend they would like to use using the ``ir`` flag for ``compile``.
If given a ``torch.nn.Module`` and the ``ir`` flag is set to either ``default`` or ``torchscript`` the module will be run through
``torch.jit.script`` to convert the input module into a  TorchScript module.


To compile your input ``torch.nn.Module`` with Torch-TensorRT, all you need to do is provide the module and inputs
to Torch-TensorRT and you will be returned an optimized TorchScript module to run or add into another PyTorch module. Inputs
is a list of ``torch_tensorrt.Input`` classes which define input Tensors' shape, datatype and memory format. Alternatively, if your input is a more complex data type, such as a tuple or list of Tensors, you can use the ``input_signature`` argument to specify a collection-based input, such as ``(List[Tensor], Tuple[Tensor, Tensor])``. See the second sample below for an example. You can also specify settings such as operating precision for the engine or target device. After compilation you can save the module just like any other module
to load in a deployment application. In order to load a TensorRT/TorchScript module, make sure you first import ``torch_tensorrt``.

.. code-block:: python

    import torch_tensorrt

    ...

    model = MyModel().eval()  # torch module needs to be in eval (not training) mode

    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 1, 16, 16],
            opt_shape=[1, 1, 32, 32],
            max_shape=[1, 1, 64, 64],
            dtype=torch.half,
        )
    ]
    enabled_precisions = {torch.float, torch.half}  # Run with fp16

    trt_ts_module = torch_tensorrt.compile(
        model, inputs=inputs, enabled_precisions=enabled_precisions
    )

    input_data = input_data.to("cuda").half()
    result = trt_ts_module(input_data)
    torch.jit.save(trt_ts_module, "trt_ts_module.ts")

.. code-block:: python

    # Sample using collection-based inputs via the input_signature argument
    import torch_tensorrt

    ...

    model = MyModel().eval()

    # input_signature expects a tuple of individual input arguments to the module
    # The module below, for example, would have a docstring of the form:
    # def forward(self, input0: List[torch.Tensor], input1: Tuple[torch.Tensor, torch.Tensor])
    input_signature = (
        [torch_tensorrt.Input(shape=[64, 64], dtype=torch.half), torch_tensorrt.Input(shape=[64, 64], dtype=torch.half)],
        (torch_tensorrt.Input(shape=[64, 64], dtype=torch.half), torch_tensorrt.Input(shape=[64, 64], dtype=torch.half)),
    )
    enabled_precisions = {torch.float, torch.half}

    trt_ts_module = torch_tensorrt.compile(
        model, input_signature=input_signature, enabled_precisions=enabled_precisions
    )

    input_data = input_data.to("cuda").half()
    result = trt_ts_module(input_data)
    torch.jit.save(trt_ts_module, "trt_ts_module.ts")

.. code-block:: python

    # Deployment application
    import torch
    import torch_tensorrt

    trt_ts_module = torch.jit.load("trt_ts_module.ts")
    input_data = input_data.to("cuda").half()
    result = trt_ts_module(input_data)

Torch-TensorRT Python API also provides ``torch_tensorrt.ts.compile`` which accepts a TorchScript module as input and ``torch_tensorrt.fx.compile`` which accepts a FX GraphModule as input.
