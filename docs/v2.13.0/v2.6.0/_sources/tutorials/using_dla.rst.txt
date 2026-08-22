.. _using_dla:

DLA
=================================

``DLA`` NVIDIA Deep Learning Accelerator is a fixed-function accelerator engine targeted for deep learning operations. DLA is designed to do full hardware acceleration of convolutional neural networks. DLA supports various layers such as convolution, deconvolution, fully-connected, activation, pooling, batch normalization, etc. ``torch_tensorrt`` supports compilation of TorchScript Module and deployment pipeline on the DLA hardware available on NVIDIA embedded platforms.

NOTE: DLA supports fp16 and int8 precision only.

Using DLA with torchtrtc

.. code-block:: shell

    torchtrtc [input_file_path] [output_file_path] [input_shapes...] -p f16 -d dla {OPTIONS}

Using DLA in a C++ application

.. code-block:: c++

    std::vector<std::vector<int64_t>> input_shape = {{32, 3, 32, 32}};
    auto compile_spec = torch_tensorrt::CompileSpec({input_shape});

    # Set a precision. DLA supports fp16 or int8 only
    compile_spec.enabled_precisions = {torch::kF16};
    compile_spec.device.device_type = torch_tensorrt::CompileSpec::DeviceType::kDLA;

    # Make sure the gpu id is set to Xavier id for DLA
    compile_spec.device.gpu_id = 0;

    # Set the DLA core id
    compile_spec.device.dla_core = 1;

    # If a layer fails to run on DLA it will fallback to GPU
    compile_spec.device.allow_gpu_fallback = true;


Using DLA in a python application

.. code-block:: python

    compile_spec = {
        "inputs": [torch_tensorrt.Input(self.input.shape)],
        "device": torch_tensorrt.Device("dla:0", allow_gpu_fallback=True),
        "enalbed_precisions": {torch.half},
    }

    trt_mod = torch_tensorrt.compile(self.scripted_model, compile_spec)
