.. _using_dla:

DLA
=================================

``DLA`` NVIDIA Deep Learning Accelerator is a fixed-function accelerator engine targeted for deep learning operations. DLA is designed to do full hardware acceleration of convolutional neural networks. DLA supports various layers such as convolution, deconvolution, fully-connected, activation, pooling, batch normalization, etc. ``trtorch`` supports compilation of TorchScript Module and deployment pipeline on the DLA hardware available on NVIDIA embedded platforms.

NOTE: DLA supports fp16 and int8 precision only.

Using DLA with trtorchc

.. code-block:: shell

    trtorchc [input_file_path] [output_file_path] [input_shapes...] -p f16 -d dla {OPTIONS}

Using DLA in a C++ application

.. code-block:: shell

    std::vector<std::vector<int64_t>> input_shape = {{32, 3, 32, 32}};
    auto compile_spec = trtorch::CompileSpec({input_shape});

    # Set a precision. DLA supports fp16 or int8 only
    compile_spec.op_precision = torch::kF16;
    compile_spec.device.device_type = trtorch::CompileSpec::DeviceType::kDLA;

    # Make sure the gpu id is set to Xavier id for DLA
    compile_spec.device.gpu_id = 0;

    # Set the DLA core id
    compile_spec.device.dla_core = 1;

    # If a layer fails to run on DLA it will fallback to GPU
    compile_spec.device.allow_gpu_fallback = true;

    # Set the workspace size
    compile_spec.workspace_size = 1 << 28;


Using DLA in a python application

.. code-block:: shell

    compile_spec = {
            "input_shapes": [self.input.shape],
            "device": {
                "device_type": trtorch.DeviceType.DLA,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": True
            },
            "op_precision": torch.half
    }

    trt_mod = trtorch.compile(self.scripted_model, compile_spec)
