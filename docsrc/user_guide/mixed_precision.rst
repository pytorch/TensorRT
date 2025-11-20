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


Before TensorRT 10.12, if we compile the above model using Torch-TensorRT with the following settings, 
layer profiling logs indicate that all the layers are run in FP32. This is because old TensorRT picks 
the kernels for layers which result in the best performance (i.e., weak typing in old TensorRT). 

.. code-block:: python

    inputs = [torch.randn((1, 10), dtype=torch.float32).cuda()]
    mod = MyModule().eval().cuda()
    ep = torch.export.export(mod, tuple(inputs))
    trt_gm = torch_tensorrt.dynamo.compile(ep, inputs=inputs)

    # Debug log info
    # Layers:
    # Name: __myl_MulSum_myl0_0, LayerType: kgen, Inputs: [ { Name: __mye116_dconst, Dimensions: [10,10], Format/Datatype: Float }, { Name: x, Dimensions: [10,1], Format/Datatype: Float }], Outputs: [ { Name: __myln_k_arg__bb1_2, Dimensions: [1,10], Format/Datatype: Float }], TacticName: __myl_MulSum_0xfa6c1858aea1b13b03f90165d7149ec6, StreamId: 0, Metadata: 
    # Name: __myl_AddResMulSum_myl0_1, LayerType: kgen, Inputs: [ { Name: __mye131_dconst, Dimensions: [10,30], Format/Datatype: Float }, { Name: __myln_k_arg__bb1_2, Dimensions: [1,10], Format/Datatype: Float }, { Name: linear1/addmm_constant_0 _ linear1/addmm_add_broadcast_to_same_shape_lhs_broadcast_constantFloat, Dimensions: [1,10], Format/Datatype: Float }], Outputs: [ { Name: __myln_k_arg__bb1_3, Dimensions: [1,30], Format/Datatype: Float }], TacticName: __myl_AddResMulSum_0xb3915d7ebfe48be45b6d49083479e12f, StreamId: 0, Metadata: 
    # Name: __myl_AddResMulSumAdd_myl0_2, LayerType: kgen, Inputs: [ { Name: __mye146_dconst, Dimensions: [30,40], Format/Datatype: Float }, { Name: linear3/addmm_2_constant_0 _ linear3/addmm_2_add_broadcast_to_same_shape_lhs_broadcast_constantFloat, Dimensions: [1,40], Format/Datatype: Float }, { Name: __myln_k_arg__bb1_3, Dimensions: [1,30], Format/Datatype: Float }, { Name: linear2/addmm_1_constant_0 _ linear2/addmm_1_add_broadcast_to_same_shape_lhs_broadcast_constantFloat, Dimensions: [1,30], Format/Datatype: Float }], Outputs: [ { Name: output0, Dimensions: [1,40], Format/Datatype: Float }], TacticName: __myl_AddResMulSumAdd_0xcdd0085ad25f5f45ac5fafb72acbffd6, StreamId: 0, Metadata: 


However, since TensorRT 10.12, TensorRT has deprecated weak typing, we must set ``use_explicit_typing=True`` 
to enable strong typing, which means users must specify the precision of the nodes in the model. For example,
in the case above, we set ``linear2`` layer to run in FP16, so if we compile the model with the following settings,
the ``linear2`` layer will run in FP16 and other layers will run in FP32 as shown in the following TensorRT logs:

.. code-block:: python

    inputs = [torch.randn((1, 10), dtype=torch.float32).cuda()]
    mod = MyModule().eval().cuda()
    ep = torch.export.export(mod, tuple(inputs))
    trt_gm = torch_tensorrt.dynamo.compile(ep, inputs=inputs, use_explicit_typing=True)

    # Debug log info
    # Layers:
    # Name: __myl_MulSumAddCas_myl0_0, LayerType: kgen, Inputs: [ { Name: linear1/addmm_constant_0 _ linear1/addmm_add_broadcast_to_same_shape_lhs_broadcast_constantFloat, Dimensions: [1,10], Format/Datatype: Float }, { Name: __mye112_dconst, Dimensions: [10,10], Format/Datatype: Float }, { Name: x, Dimensions: [10,1], Format/Datatype: Float }], Outputs: [ { Name: __myln_k_arg__bb1_2, Dimensions: [1,10], Format/Datatype: Half }], TacticName: __myl_MulSumAddCas_0xacf8f5dd9be2f3e7bb09cdddeac6c936, StreamId: 0, Metadata: 
    # Name: __myl_ResMulSumAddCas_myl0_1, LayerType: kgen, Inputs: [ { Name: __mye127_dconst, Dimensions: [10,30], Format/Datatype: Half }, { Name: linear2/addmm_1_constant_0 _ linear2/addmm_1_add_broadcast_to_same_shape_lhs_broadcast_constantHalf, Dimensions: [1,30], Format/Datatype: Half }, { Name: __myln_k_arg__bb1_2, Dimensions: [1,10], Format/Datatype: Half }], Outputs: [ { Name: __myln_k_arg__bb1_3, Dimensions: [1,30], Format/Datatype: Float }], TacticName: __myl_ResMulSumAddCas_0x5a3b318b5a1c97b7d5110c0291481337, StreamId: 0, Metadata: 
    # Name: __myl_ResMulSumAdd_myl0_2, LayerType: kgen, Inputs: [ { Name: __mye142_dconst, Dimensions: [30,40], Format/Datatype: Float }, { Name: linear3/addmm_2_constant_0 _ linear3/addmm_2_add_broadcast_to_same_shape_lhs_broadcast_constantFloat, Dimensions: [1,40], Format/Datatype: Float }, { Name: __myln_k_arg__bb1_3, Dimensions: [1,30], Format/Datatype: Float }], Outputs: [ { Name: output0, Dimensions: [1,40], Format/Datatype: Float }], TacticName: __myl_ResMulSumAdd_0x3fad91127c640fd6db771aa9cde67db0, StreamId: 0, Metadata: 

Autocast
---------------

Weak typing behavior in TensorRT is deprecated. However mixed precision is a good way to maximize performance. 
Therefore, in Torch-TensorRT, we want to provide a way to enable mixed precision behavior like weak typing in 
old TensorRT, which is called `Autocast`. 

Before we dive into Torch-TensorRT Autocast, let's first take a look at PyTorch Autocast. PyTorch Autocast is a 
context-based autocast, which means it will affect the precision of the nodes inside the context. For example,
in PyTorch, we can do the following:

.. code-block:: python

    x = self.linear1(x)
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        x = self.linear2(x)
    x = self.linear3(x)

This will run ``linear2`` in FP16 and other layers remain in FP32. Please refer to `PyTorch Autocast documentation <https://docs.pytorch.org/docs/stable/amp.html#torch.autocast>`_ for more details.

Unlike PyTorch Autocast, Torch-TensorRT Autocast is a rule-based autocast, which intelligently selects nodes to 
keep in FP32 precision to maintain model accuracy while benefiting from reduced precision on the rest of the nodes. 
Torch-TensorRT Autocast also supports users to specify which nodes to exclude from Autocast, considering some nodes 
might be more sensitive to affecting accuracy. In addition, Torch-TensorRT Autocast can cooperate with PyTorch Autocast, 
allowing users to use both PyTorch Autocast and Torch-TensorRT Autocast in the same model. Torch-TensorRT Autocast 
respects the precision of the nodes within PyTorch Autocast context.

To enable Torch-TensorRT Autocast, we need to set both ``enable_autocast=True`` and ``use_explicit_typing=True``. 
On top of them, we can also specify the precision of the nodes to reduce to by ``autocast_low_precision_type``, 
and exclude certain nodes/ops from Torch-TensorRT Autocast by ``autocast_excluded_nodes`` or ``autocast_excluded_ops``.
For example,

.. code-block:: python

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10,10)
            self.linear2 = torch.nn.Linear(10,30)
            self.linear3 = torch.nn.Linear(30,40)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            return x

    inputs = [torch.randn((1, 10), dtype=torch.float32).cuda()]
    mod = MyModule().eval().cuda()
    ep = torch.export.export(mod, tuple(inputs))
    trt_gm = torch_tensorrt.dynamo.compile(
        ep, 
        inputs=inputs, 
        enable_autocast=True, 
        use_explicit_typing=True,
        autocast_low_precision_type=torch.float16,
        autocast_excluded_nodes={"^linear2$"},
    )

This model excludes ``linear2`` from Autocast, so it will run ``linear2`` in FP32 and other layers in FP16. 

In summary, now there are two ways in Torch-TensorRT to choose the precision of the nodes:
1. User specifies precision (strong typing):                ``use_explicit_typing=True + enable_autocast=False``
2. Autocast chooses precision (autocast + strong typing):   ``use_explicit_typing=True + enable_autocast=True``

FP32 Accumulation
-----------------

When ``use_fp32_acc=True`` is set, Torch-TensorRT will attempt to use FP32 accumulation for matmul layers, even if the input and output tensors are in FP16. This is particularly useful for models that are sensitive to numerical errors introduced by lower-precision accumulation.

.. important::

    When enabling ``use_fp32_acc=True``, **explicit typing must be enabled** by setting ``use_explicit_typing=True``. Without ``use_explicit_typing=True``, the accumulation type may not be properly respected, and you may not see the intended numerical benefits.

.. code-block:: python

    inputs = [torch.randn((1, 10), dtype=torch.float16).cuda()]
    mod = MyModule().eval().cuda()
    ep = torch.export.export(mod, tuple(inputs))
    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        inputs=inputs,
        use_fp32_acc=True,
        use_explicit_typing=True,  # Explicit typing must be enabled
    )

    # Debug log info
    # Layers:
    # Name: __myl_MulSumAddCas_myl0_0, LayerType: kgen, Inputs: [ ... ], Outputs: [ ... ], Format/Datatype: Half, Accumulation: Float
    # ...

For more information on these settings, see the explicit typing examples above.