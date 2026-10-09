.. _mixed_precision:

Compile Mixed Precision models with Torch-TensorRT
===================================================
.. currentmodule:: torch_tensorrt.dynamo

.. automodule:: torch_tensorrt.dynamo
   :members:
   :undoc-members:
   :show-inheritance:

Explicit Typing
---------------

Consider the following PyTorch model which explicitly casts intermediate layer to run in FP16. 

.. code-block:: python

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10,10)
            self.linear2 = torch.nn.Linear(10,30).half()
            self.linear3 = torch.nn.Linear(30,40)

        def forward(self, x):
            x = self.linear1(x)
            x = x.to(torch.float16)
            x = self.linear2(x)
            x = x.to(torch.float32)
            x = self.linear3(x)
            return x


If we compile the above model using Torch-TensorRT, layer profiling logs indicate that all the layers are 
run in FP32. This is because TensorRT picks the kernels for layers which result in the best performance. 

.. code-block:: python

    inputs = [torch.randn((1, 10), dtype=torch.float32).cuda()]
    mod = MyModule().eval().cuda()
    ep = torch.export.export(mod, tuple(inputs))
    with torch_tensorrt.logging.debug():
        trt_gm = torch_tensorrt.dynamo.compile(ep, 
                                            inputs=inputs, 
                                            debug=True)

    # Debug log info
    # Layers:
    # Name: __myl_MulSum_myl0_0, LayerType: kgen, Inputs: [ { Name: __mye116_dconst, Dimensions: [10,10], Format/Datatype: Float }, { Name: x, Dimensions: [10,1], Format/Datatype: Float }], Outputs: [ { Name: __myln_k_arg__bb1_2, Dimensions: [1,10], Format/Datatype: Float }], TacticName: __myl_MulSum_0xfa6c1858aea1b13b03f90165d7149ec6, StreamId: 0, Metadata: 
    # Name: __myl_AddResMulSum_myl0_1, LayerType: kgen, Inputs: [ { Name: __mye131_dconst, Dimensions: [10,30], Format/Datatype: Float }, { Name: __myln_k_arg__bb1_2, Dimensions: [1,10], Format/Datatype: Float }, { Name: linear1/addmm_constant_0 _ linear1/addmm_add_broadcast_to_same_shape_lhs_broadcast_constantFloat, Dimensions: [1,10], Format/Datatype: Float }], Outputs: [ { Name: __myln_k_arg__bb1_3, Dimensions: [1,30], Format/Datatype: Float }], TacticName: __myl_AddResMulSum_0xb3915d7ebfe48be45b6d49083479e12f, StreamId: 0, Metadata: 
    # Name: __myl_AddResMulSumAdd_myl0_2, LayerType: kgen, Inputs: [ { Name: __mye146_dconst, Dimensions: [30,40], Format/Datatype: Float }, { Name: linear3/addmm_2_constant_0 _ linear3/addmm_2_add_broadcast_to_same_shape_lhs_broadcast_constantFloat, Dimensions: [1,40], Format/Datatype: Float }, { Name: __myln_k_arg__bb1_3, Dimensions: [1,30], Format/Datatype: Float }, { Name: linear2/addmm_1_constant_0 _ linear2/addmm_1_add_broadcast_to_same_shape_lhs_broadcast_constantFloat, Dimensions: [1,30], Format/Datatype: Float }], Outputs: [ { Name: output0, Dimensions: [1,40], Format/Datatype: Float }], TacticName: __myl_AddResMulSumAdd_0xcdd0085ad25f5f45ac5fafb72acbffd6, StreamId: 0, Metadata: 


In order to respect the types specified by the user in the model (eg: in this case, ``linear2`` layer to run in FP16), users can enable 
the compilation setting ``use_explicit_typing=True``. Compiling with this option results in the following TensorRT logs

.. note:: If you enable ``use_explicit_typing=True``, only torch.float32 is supported in the enabled_precisions.


.. code-block:: python

    inputs = [torch.randn((1, 10), dtype=torch.float32).cuda()]
    mod = MyModule().eval().cuda()
    ep = torch.export.export(mod, tuple(inputs))
    with torch_tensorrt.logging.debug():
        trt_gm = torch_tensorrt.dynamo.compile(ep, 
                                            inputs=inputs, 
                                            use_explicit_typing=True,
                                            debug=True)

    # Debug log info
    # Layers:
    # Name: __myl_MulSumAddCas_myl0_0, LayerType: kgen, Inputs: [ { Name: linear1/addmm_constant_0 _ linear1/addmm_add_broadcast_to_same_shape_lhs_broadcast_constantFloat, Dimensions: [1,10], Format/Datatype: Float }, { Name: __mye112_dconst, Dimensions: [10,10], Format/Datatype: Float }, { Name: x, Dimensions: [10,1], Format/Datatype: Float }], Outputs: [ { Name: __myln_k_arg__bb1_2, Dimensions: [1,10], Format/Datatype: Half }], TacticName: __myl_MulSumAddCas_0xacf8f5dd9be2f3e7bb09cdddeac6c936, StreamId: 0, Metadata: 
    # Name: __myl_ResMulSumAddCas_myl0_1, LayerType: kgen, Inputs: [ { Name: __mye127_dconst, Dimensions: [10,30], Format/Datatype: Half }, { Name: linear2/addmm_1_constant_0 _ linear2/addmm_1_add_broadcast_to_same_shape_lhs_broadcast_constantHalf, Dimensions: [1,30], Format/Datatype: Half }, { Name: __myln_k_arg__bb1_2, Dimensions: [1,10], Format/Datatype: Half }], Outputs: [ { Name: __myln_k_arg__bb1_3, Dimensions: [1,30], Format/Datatype: Float }], TacticName: __myl_ResMulSumAddCas_0x5a3b318b5a1c97b7d5110c0291481337, StreamId: 0, Metadata: 
    # Name: __myl_ResMulSumAdd_myl0_2, LayerType: kgen, Inputs: [ { Name: __mye142_dconst, Dimensions: [30,40], Format/Datatype: Float }, { Name: linear3/addmm_2_constant_0 _ linear3/addmm_2_add_broadcast_to_same_shape_lhs_broadcast_constantFloat, Dimensions: [1,40], Format/Datatype: Float }, { Name: __myln_k_arg__bb1_3, Dimensions: [1,30], Format/Datatype: Float }], Outputs: [ { Name: output0, Dimensions: [1,40], Format/Datatype: Float }], TacticName: __myl_ResMulSumAdd_0x3fad91127c640fd6db771aa9cde67db0, StreamId: 0, Metadata: 

Now the ``linear2`` layer runs in FP16 as shown in the above logs. 



FP32 Accumulation
-----------------

When ``use_fp32_acc=True`` is set, Torch-TensorRT will attempt to use FP32 accumulation for matmul layers, even if the input and output tensors are in FP16. This is particularly useful for models that are sensitive to numerical errors introduced by lower-precision accumulation.

.. important::

    When enabling ``use_fp32_acc=True``, **explicit typing must be enabled** by setting ``use_explicit_typing=True``. Without ``use_explicit_typing=True``, the accumulation type may not be properly respected, and you may not see the intended numerical benefits.

.. code-block:: python

    inputs = [torch.randn((1, 10), dtype=torch.float16).cuda()]
    mod = MyModule().eval().cuda()
    ep = torch.export.export(mod, tuple(inputs))
    with torch_tensorrt.logging.debug():
        trt_gm = torch_tensorrt.dynamo.compile(
            ep,
            inputs=inputs,
            use_fp32_acc=True,
            use_explicit_typing=True,  # Explicit typing must be enabled
            debug=True
        )

    # Debug log info
    # Layers:
    # Name: __myl_MulSumAddCas_myl0_0, LayerType: kgen, Inputs: [ ... ], Outputs: [ ... ], Format/Datatype: Half, Accumulation: Float
    # ...

For more information on these settings, see the explicit typing examples above.