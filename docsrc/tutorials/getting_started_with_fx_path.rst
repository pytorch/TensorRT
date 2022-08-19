.. _getting_started_with_fx:

Torch-TensorRT (FX Frontend) User Guide
========================
Torch-TensorRT (FX Frontend) is a tool that can convert a PyTorch model through ``torch.fx`` to an
TensorRT engine optimized targeting running on Nvidia GPUs. TensorRT is the inference engine
developed by NVIDIA which composed of various kinds of optimization including kernel fusion,
graph optimization, low precision, etc.. This tool is developed in Python environment which allows this
workflow to be very accessible to researchers and engineers. There are a few stages that a
user want to use this tool and we will introduce them here.

> Torch-TensorRT (FX Frontend) is in ``Beta`` and currently it is recommended to work with PyTorch nightly.

.. code-block:: shell

    # Test an example by
    $ python py/torch_tensorrt/fx/example/lower_example.py


Converting a PyTorch Model to TensorRT Engine
---------------------------------------------
In general, users are welcome to use the ``compile()`` to finish the conversion from a model to tensorRT engine. It is a
wrapper API that consists of the major steps needed to finish this converison. Please refer to ``lower_example.py`` file in ``examples/fx``.

In this section, we will go through an example to illustrate the major steps that fx path uses.
Users can refer to ``fx2trt_example.py`` file in ``examples/fx``.

* **Step 1: Trace the model with acc_tracer**
Acc_tracer is a tracer inheritated from FX tracer. It comes with args normalizer to convert all args to kwargs and pass to TRT converters.

.. code-block:: shell

   import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer

   # Build the model which needs to be a PyTorch nn.Module.
   my_pytorch_model = build_model()

   # Prepare inputs to the model. Inputs have to be a List of Tensors
   inputs = [Tensor, Tensor, ...]

   # Trace the model with acc_tracer.
   acc_mod = acc_tracer.trace(my_pytorch_model, inputs)

*Common Errors:*

symbolically traced variables cannot be used as inputs to control flow
This means the model contains dynamic control flow. Please refer to section “Dynamic Control Flow” in `FX guide <https://pytorch.org/docs/stable/fx.html#dynamic-control-flow>`_.

* **Step 2: Build TensorRT engine**
There are `two different modes <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch>`_ for how TensorRT handles batch dimension, explicit batch dimension and implicit batch dimension. This mode was used by early versions of TensorRT, and is now deprecated but continues to be supported for backwards compatibility. In explicit batch mode, all dimensions are explicit and can be dynamic, that is their length can change at execution time. Many new features, such as dynamic shapes and loops, are available only in this mode. User can still choose to use implicit batch mode when they set ``explicit_batch_dimension=False`` in ``compile()``. We do not recommend to use it since it will lack of support in future TensorRT versions.

Explicit batch is the default mode and it must be set for dynamic shape. For most of vision task, user can choose to enable ``dynamic_batch`` in ``compile()`` if they want to get the similar effects as implicit mode where only batch dimension changes. It has some requirements:
1. Shapes of inputs, outputs and activations are fixed except batch dimension.
2. Inputs, outputs and activations have batch dimension as the major dimension.
3. All the operators in the model do not modify batch dimension (permute, transpose, split, etc.) or compute over batch dimension (sum, softmax, etc.).

For examples of the last path, if we have a 3D tensor t shaped as (batch, sequence, dimension), operations such as torch.transpose(0, 2). If any of these three are not satisfied, we’ll need to specify InputTensorSpec as inputs with dynamic range.

.. code-block:: shell

    import deeplearning.trt.fx2trt.converter.converters
    from torch.fx.experimental.fx2trt.fx2trt import InputTensorSpec, TRTInterpreter

    # InputTensorSpec is a dataclass we use to store input information.
    # There're two ways we can build input_specs.
    # Option 1, build it manually.
    input_specs = [
      InputTensorSpec(shape=(1, 2, 3), dtype=torch.float32),
      InputTensorSpec(shape=(1, 4, 5), dtype=torch.float32),
    ]
    # Option 2, build it using sample_inputs where user provide a sample
    inputs = [
    torch.rand((1,2,3), dtype=torch.float32),
    torch.rand((1,4,5), dtype=torch.float32),
    ]
    input_specs = InputTensorSpec.from_tensors(inputs)

    # IMPORTANT: If dynamic shape is needed, we need to build it slightly differently.
    input_specs = [
        InputTensorSpec(
            shape=(-1, 2, 3),
            dtype=torch.float32,
            # Currently we only support one set of dynamic range. User may set other dimensions but it is not promised to work for any models
            # (min_shape, optimize_target_shape, max_shape)
            # For more information refer to fx/input_tensor_spec.py
            shape_ranges = [
                ((1, 2, 3), (4, 2, 3), (100, 2, 3)),
            ],
        ),
        InputTensorSpec(shape=(1, 4, 5), dtype=torch.float32),
    ]

    # Build a TRT interpreter. Set explicit_batch_dimension accordingly.
    interpreter = TRTInterpreter(
        acc_mod, input_specs, explicit_batch_dimension=True/False
    )

    # The output of TRTInterpreter run() is wrapped as TRTInterpreterResult.
    # The TRTInterpreterResult contains required parameter to build TRTModule,
    # and other informational output from TRTInterpreter run.
    class TRTInterpreterResult(NamedTuple):
        engine: Any
        input_names: Sequence[str]
        output_names: Sequence[str]
        serialized_cache: bytearray

    #max_batch_size: set accordingly for maximum batch size you will use.
    #max_workspace_size: set to the maximum size we can afford for temporary buffer
    #lower_precision: the precision model layers are running on (TensorRT will choose the best perforamnce precision).
    #sparse_weights: allow the builder to examine weights and use optimized functions when weights have suitable sparsity
    #force_fp32_output: force output to be fp32
    #strict_type_constraints: Usually we should set it to False unless we want to control the precision of certain layer for numeric #reasons.
    #algorithm_selector: set up algorithm selection for certain layer
    #timing_cache: enable timing cache for TensorRT
    #profiling_verbosity: TensorRT logging level
    trt_interpreter_result = interpreter.run(
        max_batch_size=64,
        max_workspace_size=1 << 25,
        sparse_weights=False,
        force_fp32_output=False,
        strict_type_constraints=False,
        algorithm_selector=None,
        timing_cache=None,
        profiling_verbosity=None,
    )


*Common Errors:*

RuntimeError: Conversion of function xxx not currently supported!
- This means we don’t have the support for this xxx operator. Please refer to section “How to add a missing op” below for further instructions.

* **Step 3: Run the model**
One way is using TRTModule, which is basically a PyTorch nn.Module.

.. code-block:: shell

    from torch_tensorrt.fx import TRTModule
    mod = TRTModule(
        trt_interpreter_result.engine,
        trt_interpreter_result.input_names,
        trt_interpreter_result.output_names)
    # Just like all other PyTorch modules
    outputs = mod(*inputs)
    torch.save(mod, "trt.pt")
    reload_trt_mod = torch.load("trt.pt")
    reload_model_output = reload_trt_mod(*inputs)

So far, we give a detailed explanation of major steps in convterting a PyTorch model into TensorRT engine. Users are welcome to refer to the source code for some parameters explanations. In the converting scheme, there are two important actions in it. One is acc tracer which helps us to convert a PyTorch model to acc graph. The other is FX path converter which helps to convert the acc graph's operation to corresponding TensorRT operation and build up the TensoRT engine for it.

Acc Tracer
---------

Acc tracer is a custom FX symbolic tracer. It does a couple more things compare to the vanilla FX symbolic tracer. We mainly depend on it to convert PyTorch ops or builtin ops to acc ops. There are two main purposes for fx2trt to use acc ops:

1. there’re many ops that do similar things in PyTorch ops and builtin ops such like torch.add, builtin.add and torch.Tensor.add. Using acc tracer, we normalize these three ops to a single acc_ops.add. This helps reduce the number of converters we need to write.
2. acc ops only have kwargs which makes writing converter easier as we don’t need to add additional logic to find arguments in args and kwargs.

FX2TRT
--------
After symbolic tracing, we have the graph representation of a PyTorch model. fx2trt leverages the power of fx.Interpreter. fx.Interpreter goes through the whole graph node by node and calls the function that node represents. fx2trt overrides the original behavior of calling the function with invoking corresponding converts for each node. Each converter function adds corresponding TensorRT layer(s).

Below is an example of a converter function. The decorator is used to register this converter function with the corresponding node. In this example, we register this converter to a fx node whose target is acc_ops.sigmoid.

.. code-block:: shell

    @tensorrt_converter(acc_ops.sigmoid)
    def acc_ops_sigmoid(network, target, args, kwargs, name):
        """
        network: TensorRT network. We'll be adding layers to it.

        The rest arguments are attributes of fx node.
        """
        input_val = kwargs['input']

        if not isinstance(input_val, trt.tensorrt.ITensor):
            raise RuntimeError(f'Sigmoid received input {input_val} that is not part '
                            'of the TensorRT region!')

        layer = network.add_activation(input=input_val, type=trt.ActivationType.SIGMOID)
        layer.name = name
        return layer.get_output(0)

How to Add a Missing Op
****************

You can actually add it wherever you want just need to remember import the file so that all acc ops and mapper will be registered before tracing with acc_tracer.

* **Step 1. Add a new acc op**

TODO: Need to explain more on the logistic of acc op like when we want to break down an op and when we want to reuse other ops.

In `acc tracer <https://github.com/pytorch/TensorRT/blob/master/py/torch_tensorrt/fx/tracer/acc_tracer/acc_tracer.py>`_, we convert nodes in the graph to acc ops if there’s a mapping registered for the node to an acc op.

In order to make the conversion to acc ops to happen, there’re two things required. One is that there should be an acc op function defined and the other is there should be a mapping registered.

Defining an acc op is simple, we first just need a function and register the function as an acc op via this decorator `acc_normalizer.py <https://github.com/pytorch/TensorRT/blob/master/py/torch_tensorrt/fx/tracer/acc_tracer/acc_normalizer.py>`_. e.g. the following code adds an acc op named foo() which adds two given inputs.

.. code-block:: shell

    # NOTE: all acc ops should only take kwargs as inputs, therefore we need the "*"
    # at the beginning.
    @register_acc_op
    def foo(*, input, other, alpha):
        return input + alpha * other

There’re two ways to register a mapping. One is `register_acc_op_mapping() <https://github.com/pytorch/TensorRT/blob/1a22204fecec690bc3c2a318dab4f57b98c57f05/py/torch_tensorrt/fx/tracer/acc_tracer/acc_normalizer.py#L164>`_. Let’s register a mapping from torch.add to foo() we just created above. We need to add decorator register_acc_op_mapping to it.

.. code-block:: shell

    this_arg_is_optional = True

    @register_acc_op_mapping(
        op_and_target=("call_function", torch.add),
        arg_replacement_tuples=[
            ("input", "input"),
            ("other", "other"),
            ("alpha", "alpha", this_arg_is_optional),
        ],
    )
    @register_acc_op
    def foo(*, input, other, alpha=1.0):
        return input + alpha * other

``op_and_target`` determines which node will trigger this mapping. op and target are the attributes of FX node. In acc_normalization when we see a node with the same op and target as set in the ``op_and_target``, we will trigger the mapping. Since we want to map from ``torch.add``, then op would be call_function and target would be ``torch.add``. ``arg_replacement_tuples`` determines how we construct kwargs for new acc op node using args and kwargs from original node. Each tuple in ``arg_replacement_tuples`` represents one argument mapping rule. It contains two or three elements. The third element is a boolean variable that determines whether this kwarg is optional in *original node*. We only need to specify the third element if it’s True. The first element is the argument name in original node which will be used as the acc op node’s argument whose name is the second element in the tuple. The sequence of the tuples does matter because the position of the tuple determines where the argument is in original node’s args. We use this information to map args from original node to kwargs in acc op node.
We don’t have to specify arg_replacement_tuples if none of the followings are true.

1. kwargs of original nodes and acc op nodes have different name.
2. there’re optional arguments.

The other way to register a mapping is through `register_custom_acc_mapper_fn() <https://github.com/pytorch/TensorRT/blob/1a22204fecec690bc3c2a318dab4f57b98c57f05/py/torch_tensorrt/fx/tracer/acc_tracer/acc_normalizer.py#L206>`_. This one is designed to reduce the redundant op registration as it allows you to use a function to map to one or more existing acc ops throught some combinations. In the function, you can do basically whatever you want. Let’s use an example to explain how it works.

.. code-block:: shell

    @register_acc_op
    def foo(*, input, other, alpha=1.0):
        return input + alpha * other

    @register_custom_acc_mapper_fn(
        op_and_target=("call_function", torch.add),
        arg_replacement_tuples=[
            ("input", "input"),
            ("other", "other"),
            ("alpha", "alpha", this_arg_is_optional),
        ],
    )
    def custom_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
        """
        `node` is original node, which is a call_function node with target
        being torch.add.
        """
        alpha = 1
        if "alpha" in node.kwargs:
            alpha = node.kwargs["alpha"]
        foo_kwargs = {"input": node["input"], "other": node["other"], "alpha": alpha}
        with node.graph.inserting_before(node):
            foo_node = node.graph.call_function(foo, kwargs=foo_kwargs)
            foo_node.meta = node.meta.copy()
            return foo_node


In the custom mapper function, we construct an acc op node and return it. The node we returns here would take over all the children nodes of original nodes `acc_normalizer.py <https://github.com/pytorch/TensorRT/blob/1a22204fecec690bc3c2a318dab4f57b98c57f05/py/torch_tensorrt/fx/tracer/acc_tracer/acc_normalizer.py#L379>`_.

The last step would be *adding unit test* for the new acc op or mapper function we added. The place to add the unit test is here `test_acc_tracer.py <https://github.com/pytorch/TensorRT/blob/master/py/torch_tensorrt/fx/test/tracer/test_acc_tracer.py>`_.

* **Step 2. Add a new converter**

All the developed converters for acc ops are all in `acc_op_converter.py <https://github.com/pytorch/TensorRT/blob/master/py/torch_tensorrt/fx/converters/acc_ops_converters.py>`_. It could give you a good example of how the converter is added.

Essentially, the converter is the mapping mechanism that maps the acc ops to a TensorRT layer. If we are able to find all the TensorRT layers we need we can get start to add a converter for the node using `TensorRT APIs <https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Network.html>`_.

.. code-block:: shell

    @tensorrt_converter(acc_ops.sigmoid)
    def acc_ops_sigmoid(network, target, args, kwargs, name):
        """
        network: TensorRT network. We'll be adding layers to it.

        The rest arguments are attributes of fx node.
        """
        input_val = kwargs['input']

        if not isinstance(input_val, trt.tensorrt.ITensor):
            raise RuntimeError(f'Sigmoid received input {input_val} that is not part '
                            'of the TensorRT region!')

        layer = network.add_activation(input=input_val, type=trt.ActivationType.SIGMOID)
        layer.name = name
        return layer.get_output(0)

We need to use ``tensorrt_converter`` decorator to register the converter. The argument for the decorator is the target of the fx node that we need to convert. In the converter, we can find the inputs to the fx node in kwargs. As in the example, the original node is `acc_ops.sigmoid` which only has one argument “input” in acc_ops.py. We get the input and check if it’s a TensorRT tensor. After that, we add a sigmoid layer to TensorRT network and return the output of the layer. The output we returned will be passed to the children nodes of acc_ops.sigmoid by fx.Interpreter.

**What if we can not find corresponding layers in TensorRT that do the same thing as the node.**

In this case, we would need to do a bit more work. TensorRT provides plugins which serves as custom layers. *We have not implement this feature yet. We will update once it is enabled*.

Last step would be adding the unit test for the new converter we added. User could add corresponding unit test in this `folder <https://github.com/pytorch/TensorRT/tree/master/py/torch_tensorrt/fx/test/converters/acc_op>`_.
