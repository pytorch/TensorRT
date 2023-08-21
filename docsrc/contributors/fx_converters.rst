.. _conversion:

FX Converters
==================
The FX converter library in Torch-TensorRT is located in ``TensorRT/py/torch_tensorrt/fx/converters`` (Converters present in FX will soon be deprecated) and ``TensorRT/py/torch_tensorrt/dynamo/conversion``.
FX converters are categorized into - ``aten_ops_converters``, ``acc_ops_converters`` and ``nn_ops_converters``, while dynamo converters only cover ``aten_ops_converters``
The individual converters present in the above folders are useful for the quantization workflow.

The dynamo converters are registered using the ``dynamo_tensorrt_converter`` and the FX converters are registered using the ``tensorrt_converter``.
Since FX converters will be deprecated soon, this document will focus more on the dynamo converters.


Steps
==================

Operation Sets
-------------------
There are three different converter sets for FX in torch_tensorrt. Depending on whether the operation is generated using acc_trace, aten_trace or fx_trace, the converters are categorized to one of the three operation sets - 
``aten_ops_converters``, ``acc_ops_converters`` or ``nn_ops_converters``.  The converters are registered using ``tensorrt_converter`` decorator for FX and ``dynamo_tensorrt_converter`` for dynamo. The function decorated
has the arguments - ``network, target, args, kwargs, name``,  which is common across all the operators schema.
These functions are mapped in the ``aten`` converter registry dictionary (at present a compilation of FX and dynamo converters, FX will be deprecated soon), with key as the function target name.
    
    * acc_ops_converters
        *  acc_trace is produced by ``torch_tensorrt.fx.tracer.acc_tracer.acc_tracer.trace``.
    * aten_ops
        There are two options at present for this
            #.   Dynamo: aten_trace is produced by ``torch_tensorrt.dynamo.backend.compile``. The second round of trace is produced by ``aot_torch_tensorrt_aten_backend`` by invoking ``aot_module_simplified`` from ``torch._functorch.aot_autograd``
            #.   FX: aten_trace is produced by ``torch_tensorrt.fx.tracer.dispatch_tracer.aten_tracer.trace``. This flow is more common currently, but this will soon be deprecated in torch_tensorrt.
    * nn_ops
        *  symbolic_trace is produced by ``torch.fx._symbolic_trace``.

As mentioned above, if you would like to add a new converter, its implementation will be included in ``TensorRT/py/torch_tensorrt/dynamo/conversion/impl``
Although there is a corresponding implementation of the converters  included in the common implementation library present in ``TensorRT/py/torch_tensorrt/fx/impl`` for FX converters, this documentation focuses on the implementation of the ``aten_ops`` converters in dynamo. There might be some steps involved in reorganizing files for ``acc_ops`` converters. This is discussed in more detail in the next section.


Converter implementation
------------------------
In this section, we illustrate the steps to be implemented for writing a converter. We divide them according to activation, operator, lowering pass implementation or evaluator.
Each of them is detailed with the help of an example

    * Registration
        
        The converter needs to be registered with the appropriate op code in the ``tensorrt_converter`` and ``dynamo_tensorrt_converter``. 
        
        * Activation type

            Example: ``leaky_relu``
            
            * acc_ops_converters

                * FX_converters (soon to be deprecated)
                
                    Define in ``py/torch_tensorrt/fx/converters/acc_ops_converters``. One needs to register the opcode generated in the trace, with ``tensorrt_converter`` decorator. Op code to be used for the registration or the converter registry key in this case is ``acc_ops.leaky_relu``

                    .. code-block:: python

                        @tensorrt_converter(acc_ops.leaky_relu)
                        def acc_ops_leaky_relu(
                            network: TRTNetwork,
                            target: Target,
                            args: Tuple[Argument, ...],
                            kwargs: Dict[str, Argument],
                            name: str,
                        ) -> Union[TRTTensor, Sequence[TRTTensor]]:
                                input_val = kwargs["input"]
                                negative_slope = kwargs["negative_slope"]
                                operation_type = trt.ActivationType.LEAKY_RELU
                                return activation.leaky_relu(
                                    network, target, SourceIR.ACC, name, kwargs["input"], kwargs["negative_slope"]
                                )

                    Note since the above is deprecated, you may need to revisit these files only for file reorganization.

                * Dynamo_converters

                    The ``acc_ops`` are not present in dynamo converters
            
            * aten_ops_converters

                * FX_converters (soon to be deprecated)

                    Define in ``py/torch_tensorrt/fx/converters/aten_ops_converters``. One needs to register the opcode generated in the trace with ``tensorrt_converter`` decorator. Op code to be used for the registration or the converter registry key in this case is ``torch.ops.aten.leaky_relu.default``

                        .. code-block:: python
            
                            @tensorrt_converter(torch.ops.aten.leaky_relu.default)
                                def aten_ops_leaky_relu(
                                    network: TRTNetwork,
                                    target: Target,
                                    args: Tuple[Argument, ...],
                                    kwargs: Dict[str, Argument],
                                    name: str,
                                ) -> Union[TRTTensor, Sequence[TRTTensor]]:
                                        return activation.leaky_relu(network, target, SourceIR.ATEN, name, args[0], args[1])

                * Dynamo_converters

                    Define in ``py/torch_tensorrt/dynamo/conversion/aten_ops_converters``. One needs to register the opcode generated in the trace with ``dynamo_tensorrt_converter`` decorator. Op code to be used for the registration or the converter registry key in this case is ``torch.ops.aten.leaky_relu.default``

                        .. code-block:: python
            
                            @dynamo_tensorrt_converter(torch.ops.aten.leaky_relu.default)
                                def aten_ops_leaky_relu(
                                    network: TRTNetwork,
                                    target: Target,
                                    args: Tuple[Argument, ...],
                                    kwargs: Dict[str, Argument],
                                    name: str,
                                ) -> Union[TRTTensor, Sequence[TRTTensor]]:
                                        return activation.leaky_relu(network, target, SourceIR.ATEN, name, args[0], args[1])

            The ``tensorrt_converter`` and ``dynamo_tensorrt_converter`` are similar decorator functions with some differences. 
            
            #. Both register the converters in the registeries (python dictionaries) - ``CONVERTERS`` and ``DYNAMO_CONVERTERS`` respectively. These are two dictioneries which are concatenated to form the overall converter registry 
            #. The dictionary is keyed on the ``OpOverLoad`` which is mentioned in more detail below with examples 
            #. Both return the decorated converter implementation
            #. The ``CONVERTERS`` directly registers the decorated ``converter_implementation`` function, while ``DYNAMO_CONVERTERS`` has additionational arguments and registers the ``ConverterSupport`` object
            #. The additional arguments are:

                .. code-block:: python
                    def dynamo_tensorrt_converter(
                        key: Target,
                        enabled: bool = True,
                        capability_validator: Optional[Callable[[Node], bool]] = None,
                        priority: ConverterPriority = ConverterPriority.STANDARD,
                    ) -> Callable[[Any], Union[TRTTensor, Sequence[TRTTensor]]]:

                #. key: Node target for which the converter is implemented for (for example, torch.ops.aten.leaky_relu.Tensor)
                #. enabled: Whether the converter should be enabled/cached or not
                #. capability_validator: Function which evaluates whether a node is valid for conversion by the decorated converter. It defaults to None, implying the capability_validator function is always true. This means all nodes of "key" kind can be supported by this converter by default. See ``embedding`` example for more details
                #. priority: Converter's level of priority relative to other converters with the same target

            #. The ``ConverterSupport`` is a compilation of ``converter_implementation`` and ``capability_validator``.


            The function decorated by ``tensorrt_converter`` and ``dynamo_tensorrt_converter`` has the following arguments which are automatically generated by the trace functions mentioned above.
            
            #. network : Node in the form of ``call_module`` or ``call_function`` having the target as the key
            #. target: Target key in the ``call_module`` or ``call_function`` above. eg: ``torch.ops.aten_.leaky_relu.default``. Note that ``torch.ops.aten._leaky_relu`` is the ``OpOverloadPacket`` while ``torch.ops.aten_.leaky_relu.default`` is ``OpOverload``. The 
            #. args: The arguments passed in the ``call_module`` or ``call_function`` above
            #. kwargs: The kwargs passed in the ``call_module`` or ``call_function`` above
            #. name: String containing the name of the target

            As a user writing new converters, one just needs to take care that the approriate arguments are extracted from the trace generated to the implementation function in the implementation lib function ``activation.leaky_relu`` (which we will discuss below in detail). As one can see in the example above, the trace for ``acc_op`` and ``aten_op`` is different.
            ``Acc_ops`` has arguments in the ``args`` whereas ``aten_ops`` has arguments in the ``kwargs`` in the trace.


        * Operation type

            Example: ``fmod``

            It follows the same steps as the above converter. In this case the opcode is ``torch.ops.aten.fmod.Scalar`` or ``torch.ops.aten.fmod.Tensor``. 
            Hence both the opcodes are registered in ``py/torch_tensorrt/fx/converters/aten_ops_converters`` and ``py/torch_tensorrt/dynamo/conversion/aten_ops_converters``.  The opcode is ``acc_ops.fmod`` in ``py/torch_tensorrt/fx/converters/acc_ops_converters``.
            Note that ``torch.ops.aten.fmod`` is the ``OpOverLoadPacket`` while the registry is keyed on ``torch.ops.aten.fmod.Scalar`` or ``torch.ops.aten.fmod.Tensor``, which is ``OpOverLoad``

            Example: ``embedding``

            It follows the same steps as the above converter. In this case the opcode is ``torch.ops.aten.embedding.default``. 
            There are some converters which have special cases to be accounted for. In those cases, one should use ``capability_validators`` to register the converter using ``@dynamo_tensorrt_converter``
            We illustrate this through ``torch.ops.aten.embedding.default``. It has parameters - ``scale_grad_by_freq`` and ``sparse`` which are not currently supported by the implementation.
            In such cases we can write validator ``embedding_param_validator`` which implements that given those paramters the converter is not supported and register the converter by 
                
                .. code-block:: python
                    @dynamo_tensorrt_converter(
                        torch.ops.aten.embedding.default, capability_validator=embedding_param_validator
                    )

            So if there is a new converted in which certain special cases are not to be supported then they can be specified in the ``capability_validator``.
        
        * Evaluator type

            Example: ``operator.getitem``

            Evaluators are categorized as so since they do not make any modification to the graph. This is implemented in ``py/torch_tensorrt/dynamo/conversion/op_evaluators.py``, with the corresponding ``capbility_validator``.
            The opcode is ``operator.getitem``.


    * Implementation Library

        The converters across all the above three opsets have the common implementation library. FX converters would be ``py/torch_tensorrt/fx/converters/impl`` and dynamo converters would be ``py/torch_tensorrt/dynamo/conversion/impl``
        Again as mentioned above, one should focus on the dynamo converters which are implemented in ``py/torch_tensorrt/dynamo/conversion/impl``
        
        * Activation

            Example: ``leaky_relu``
        
            The implementation is to be placed in present in ``py/torch_tensorrt/fx/impl/activation.py``. This is where all the activation functions are defined and implemented.
            
            .. code-block:: python

                def leaky_relu(
                    network: TRTNetwork,
                    target: Target,
                    source_ir: Optional[SourceIR],
                    name: str,
                    input_val: TRTTensor,
                    alpha: Optional[Any],
                ):
                    #implementation

            The implementation function has the following arguments.

            #. network : ``network`` passed from the decorated function registration
            #. target: ``target`` passed from the decorated function registration
            #. source_ir: Enum attribute. ``SourceIR`` enum is defined in ``py/torch_tensorrt/fx/converters/impl/converter_utils``
            #. name: ``name`` passed from the decorated function registration
            #. input_val: Approriate arguments extracted from the decorated function registration from args or kwargs
            #. alpha: Approriate arguments extracted from the decorated function registration from args or kwargs. If not None, it will set the alpha attribute of the created TensorRT activation layer eg: Used in leaky_relu, elu, hardtanh           
            #. beta: Approriate arguments extracted from the decorated function registration from args or kwargs. If not None, it will set the beta attribute of the created TensorRT activation layer eg: Used in hardtanh
            #. dyn_range_fn: A optional function which takes the dynamic range of a TensorRT Tensor and returns the output dynamic range

            The implementation functions call the ``convert_activation`` function in ``py/torch_tensorrt/fx/impl/activation.py``. This function will add the approriate activation layer via ``network.add_activation``.
        
        * Operator
        
            The implementation is to be placed in ``py/torch_tensorrt/fx/impl/elementwise/ops.py`` for FX and ``py/torch_tensorrt/dynamo/conversion/impl`` for dynamo. This is where all the elementwise functions are defined and implemented.
            For a new operator, one should identify the category to which it belongs. Following are some examples

            #. Elementwise operators like ``fmod`` is present in ``py/torch_tensorrt/dynamo/conversion/impl/elementwise``. The ``py/torch_tensorrt/fx/impl/elementwise/base`` contains base functions for elementwise operator.
            #. Unary operators like ``sqrt`` will be present in ``py/torch_tensorrt/dynamo/conversion/impl/unary``. The ``py/torch_tensorrt/fx/impl/unary/base`` contains base functions for unary operator.
            #. Normalization operators like ``softmax``, ``layer_norm``, ``batch_norm`` will be present in ``py/torch_tensorrt/dynamo/conversion/impl/normalization``. Since there are no base operations common to all, there is no base file. But one can choose to implement a base file, if there are common functions across all normalization operations
            #. Individual operators like ``slice``, ``select``, ``where``, ``embedding`` will be present in ``py/torch_tensorrt/dynamo/conversion/impl/*.py``. They will have individual operator implementation with the same API structure as above but with different individual arguments
            
            Please note that the above operators would have common functions to be implemented which should be placed in 
            ``py/torch_tensorrt/dynamo/conversion/impl/converter_utils.py``


    * Lowering type

        There are some converters which can be decomposed into suboperations and need not have seperate converter registration.
        Such converters can be implemented via ``lowering passes``

        Example: ``addmm``
        
        The decompositions are registered via ``register_decomposition`` in ``py/torch_tensorrt/dynamo/backend/lowering/_decompositions.py``
        We define ``addmm_replacement`` and replace it with the torch ops, which will have their corresponding converters called.

        .. code-block:: python

            @register_decomposition(torch.ops.aten.addmm, registry=DECOMPOSITIONS)
            def addmm_replacement(
                input_: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *, beta=1, alpha=1
            ) -> torch.Tensor:
                return torch.add(
                    torch.mul(input_, beta), torch.mul(torch.matmul(mat1, mat2), alpha)
                )


       
Tests
-----

* FX testing: 
    
    Implement the fx tests in ``py/torch_tensorrt/fx/test/converters/aten_op/test_<operator_name>_aten.py``. Derive the test class from ``DispatchTestCase``, with parameterized testing to implement different test cases. Check for the following two conditions
    
    #. Compare the results for ``dispatch_tracer.aten_trace`` and torch.
    #. Test the ``expected_op``. You can find examples in the above tests. This op will be called by the model and needs to be specified in the test so that the test checks that the approriate converter is invoked
        
The tests should fail if any of the above two conditions fail

* Dynamo testing: 
    
    Dynamo tests are present for the lowering ops in ``py/torch_tensorrt/dynamo/backend/test/test_decompositions.py``. The above converters will soon be ported to dynamo tests
    
    #. Compare the results for ``fx.symbolic_trace`` and ``torch_tensorrt.dynamo.compile``.
    #. Test for the ``expected_op`` and the ``unexpected_op``. 
        
        #. ``expected_op``: Operations the operations are lowered to. eg: ``mul`` and ``add`` for ``addmm``
        #. Note that specify that ``disable_passes= True`` for cases where you would not want lowering passes (which should be the default when testing converters)
        #. ``unexpected_op``: Original operation. eg: ``addmm`` for ``addmm``
        
The tests should fail if any of the above two conditions fail
