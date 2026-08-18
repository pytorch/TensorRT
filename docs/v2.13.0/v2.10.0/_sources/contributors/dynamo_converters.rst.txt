.. _dynamo_converters:

Writing Dynamo Converters
=============================
The dynamo converter library in Torch-TensorRT is located in ``TensorRT/py/torch_tensorrt/dynamo/conversion``.

Converter implementation
------------------------

Registration
^^^^^^^^^^^^^^^^

A converter is a function decrorated with  ``torch_tensorrt.dynamo.dynamo_tensorrt_converter`` that follows the function signature:


.. code-block:: python

    @torch_tensorrt.dynamo.conversion.dynamo_tensorrt_converter(torch.ops.aten.leaky_relu.default)
    def leaky_relu_converter(
        ctx: torch_tensorrt.dynamo.conversion.ConversionCtx,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[tensorrt.ITensor, Sequence[tensorrt.ITensor]]:



The decorator takes a number of arguments:

    * ``key``: Node target for which the converter is implemented for (for example, torch.ops.aten.leaky_relu.default)
    * ``enabled``: Whether the converter should be enabled as a converter that can be used in the converter registry
    * ``capability_validator``: A lambda that can take a ``torch.fx.Node`` and determine if the converter can properly handle this Node. If the validator returns ``False``, the subgraph partitioner will make sure this Node is run in PyTorch in the compiled graph.
    * ``priority``: Allows developers to override existing converters in the converter registry

All that is required for a converter is the key.

The function body is responsible for taking the current state of the network and adding the next subgraph to perform the op specified in the decorator with TensorRT operations.
The function is provided arguments as the native PyTorch op would be provided with the added case of numpy arrays for frozen Tensor attributes or TensorRT ITensors which are output Tensors of previous nodes, corresponding to edges/output Tensors of intermediate operations in the graph.
To determine the types expected as well as the return type of the converter, look at the definition of the op being converted. In the case of ``aten`` operations, this file will be the source of truth: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
Since many converters a developer may write are a composition of lower level operators, instead of needing to implement the converter in raw TensorRT, the ``torch_tensorrt.dynamo.conversion.impl`` subpackage contains many implementations of operations that can be chained to create a TensorRT subgraph.

    * ``ctx`` : The current state of the compiler. Converters primarily will manipulate ctx.net which is the ``tensorrt.INetworkDefinition`` being constructed. Additional metadata including user provided settings is available in this struct as well.
    * ``target``: Target key in the ``call_module`` or ``call_function`` above. eg: ``torch.ops.aten_.leaky_relu.default``. Note that ``torch.ops.aten._leaky_relu`` is the ``OpOverloadPacket`` while ``torch.ops.aten_.leaky_relu.default`` is ``OpOverload``.
    * ``args``: The arguments being passed to a particular Node (as collected by the ``torch_tensorrt.dynamo.conversion.TRTInterpreter``). These arguments along with the kwargs are to be used to construct a specific TensorRT subgraph representing the current node in the INetworkDefinition.
    * ``kwargs``: The arguments being passed to a particular Node (as collected by the ``torch_tensorrt.dynamo.conversion.TRTInterpreter``).
    * ``name``: String containing the name of the target

The function is expected to return the ``tensorrt.ITensor`` or some collection of ``tensorrt.ITensor`` for use in the ``torch_tensorrt.dynamo.conversion.TRTInterpreter`` matching the output signature of the operation being converted

Capability Validation
^^^^^^^^^^^^^^^^^^^^^^^

There are some converters which have special cases to be accounted for. In those cases, one should use ``capability_validators`` to register the converter using ``@dynamo_tensorrt_converter``
We illustrate this through ``torch.ops.aten.embedding.default``. It has parameters - ``scale_grad_by_freq`` and ``sparse`` which are not currently supported by the implementation.
In such cases we can write validator ``embedding_param_validator`` which implements that given those parameters the converter is not supported and register the converter by


Type Contract
^^^^^^^^^^^^^^^

The function is expected to follow the type contract established by the signature. This includes accepting the union of valid PyTorch types + numpy arrays for constant tensors and TensorRT ITensors.
In the case that only a subset of types is supported in the converter, you can also add the ``torch_tensorrt.dynamo.conversion.converter_utils.enforce_tensor_types``, which allows you to specify a dictionary mapping between input positions and types that those inputs can take. Where possible the decorator will convert inputs to match these types preferring the order provided.
``int`` keys in the dictionary will refer to positional arguments in ``args``. ``str`` keys will refer to keyword arguments in ``kwargs``.


Example: ``Convolution``
^^^^^^^^^^^^^^^^^^^^^^^^^

The default convolution converter both uses a capability validator and type enforcement to prevent being run in unsupported situations.
The capability validator is run during partitioning to determine if a particular convolution node can be converted to TensorRT or needs to run in PyTorch. Here the validator ensures that the convolution is no greater than 3D.
The type enforcer will autocast before the converter is called, inputs to the supported type in the converter, thereby limiting the number of cases an author must handle.

.. code-block:: python

    @dynamo_tensorrt_converter(
        torch.ops.aten.convolution.default, capability_validator=lambda conv_node: conv_node.args[7] in ([0], [0, 0], [0, 0, 0])
    )  # type: ignore[misc]
    @enforce_tensor_types(
        {
            0: (TRTTensor,),
            1: (np.ndarray, torch.Tensor, TRTTensor),
            2: (np.ndarray, torch.Tensor, TRTTensor),
        }
    )  # type: ignore[misc]
    def aten_ops_convolution(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[TRTTensor, Sequence[TRTTensor]]:

Evaluators
------------------------

Some operations do not produce TensorRT subgraphs as a side-effect. These are termed evaluators.

    Example: ``operator.getitem``

    Evaluators are categorized as so since they do not make any modification to the graph. This is implemented in ``py/torch_tensorrt/dynamo/conversion/op_evaluators.py``, with the corresponding ``capbility_validator``.
    The opcode is ``operator.getitem``.


Operator Decomposition
-----------------------

There are some converters which can be decomposed into suboperations in PyTorch and need not have separate converter registration.
Such converters can be implemented via a decomposition

Example: ``addmm``
^^^^^^^^^^^^^^^^^^^^^^^

The decompositions are registered via ``register_torch_trt_decomposition`` decorator
We define ``addmm_replacement`` and replace it with the torch ops, which will have their corresponding converters called.

.. code-block:: python

    @torch_tensorrt.dynamo.lowering.register_torch_trt_decomposition(torch.ops.aten.addmm)
    def addmm_replacement(
        input_: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *, beta=1, alpha=1
    ) -> torch.Tensor:
        return torch.add(
            torch.mul(input_, beta), torch.mul(torch.matmul(mat1, mat2), alpha)
        )

You can modify the decompositions run by editing ``torch_tensorrt.dynamo.lowering.torch_enabled_decompositions`` and ``torch_tensorrt.dynamo.lowering.torch_disabled_decompositions``

    Note: ``torch_tensorrt.dynamo.lowering.torch_enabled_decompositions`` and ``torch_tensorrt.dynamo.lowering.torch_disabled_decompositions`` must be disjoint sets and that the decompositions already defined in ``torch_tensorrt.dynamo.lowering`` will take precedence over torch lowering ops.

Much of the time, this is significantly easier than implementing a converter. So where possible, this is what should be tried first.