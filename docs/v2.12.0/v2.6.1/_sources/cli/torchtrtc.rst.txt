.. _torchtrtc:

torchtrtc
=================================

``torchtrtc`` is a CLI application for using the Torch-TensorRT compiler. It serves as an easy way to compile a
TorchScript Module with Torch-TensorRT from the command-line to quickly check support or as part of
a deployment pipeline. All basic features of the compiler are supported including post training
quantization (though you must already have a calibration cache file to use the PTQ feature). The compiler can
output two formats, either a TorchScript program with the TensorRT engine embedded or
the TensorRT engine itself as a PLAN file.

All that is required to run the program after compilation is for C++ linking against ``libtorchtrt.so``
or in Python importing the torch_tensorrt package. All other aspects of using compiled modules are identical
to standard TorchScript. Load with ``torch.jit.load()`` and run like you would run any other module.

.. code-block:: txt

    torchtrtc [input_file_path] [output_file_path]
      [input_specs...] {OPTIONS}

      torchtrtc is a compiler for TorchScript, it will compile and optimize
      TorchScript programs to run on NVIDIA GPUs using TensorRT

    OPTIONS:

        -h, --help                        Display this help menu
        Verbiosity of the compiler
          -v, --verbose                     Dumps debugging information about the
                                            compilation process onto the console
          -w, --warnings                    Disables warnings generated during
                                            compilation onto the console (warnings
                                            are on by default)
          --i, --info                       Dumps info messages generated during
                                            compilation onto the console
        --build-debuggable-engine         Creates a debuggable engine
        --allow-gpu-fallback              (Only used when targeting DLA
                                          (device-type)) Lets engine run layers on
                                          GPU if they are not supported on DLA
        --require-full-compilation        Require that the model should be fully
                                          compiled to TensorRT or throw an error
        --check-method-support=[method_name]
                                          Check the support for end to end
                                          compilation of a specified method in the
                                          TorchScript module
        --disable-tf32                    Prevent Float32 layers from using the
                                          TF32 data format
        --sparse-weights                  Enable sparsity for weights of conv and
                                          FC layers
        -p[precision...],
        --enable-precision=[precision...] (Repeatable) Enabling an operating
                                          precision for kernels to use when
                                          building the engine (Int8 requires a
                                          calibration-cache argument) [ float |
                                          float32 | f32 | fp32 | half | float16 |
                                          f16 | fp16 | int8 | i8 | char ]
                                          (default: float)
        -d[type], --device-type=[type]    The type of device the engine should be
                                          built for [ gpu | dla ] (default: gpu)
        --gpu-id=[gpu_id]                 GPU id if running on multi-GPU platform
                                          (defaults to 0)
        --dla-core=[dla_core]             DLACore id if running on available DLA
                                          (defaults to 0)
        --engine-capability=[capability]  The type of device the engine should be
                                          built for [ standard | safety |
                                          dla_standalone ]
        --calibration-cache-file=[file_path]
                                          Path to calibration cache file to use
                                          for post training quantization
        --teo=[op_name...],
        --torch-executed-op=[op_name...]  (Repeatable) Operator in the graph that
                                          should always be run in PyTorch for
                                          execution (partial compilation must be
                                          enabled)
        --tem=[module_name...],
        --torch-executed-mod=[module_name...]
                                          (Repeatable) Module that should always
                                          be run in Pytorch for execution (partial
                                          compilation must be enabled)
        --mbs=[num_ops],
        --min-block-size=[num_ops]        Minimum number of contiguous TensorRT
                                          supported ops to compile a subgraph to
                                          TensorRT
        --embed-engine                    Whether to treat input file as a
                                          serialized TensorRT engine and embed it
                                          into a TorchScript module (device spec
                                          must be provided)
        --num-avg-timing-iters=[num_iters]
                                          Number of averaging timing iterations
                                          used to select kernels
        --workspace-size=[workspace_size] Maximum size of workspace given to
                                          TensorRT
        --dla-sram-size=[dla_sram_size]   Fast software managed RAM used by DLA
                                          to communicate within a layer.
        --dla-local-dram-size=[dla_local_dram_size]  Host RAM used by DLA to share
                                          intermediate tensor data across operations.
        --dla-global-dram-size=[dla_global_dram_size] Host RAM used by DLA to store
                                          weights and metadata for execution
        --atol=[atol]                     Absolute tolerance threshold for acceptable
                                          numerical deviation from standard torchscript
                                          output (default 1e-8)
        --rtol=[rtol]                     Relative tolerance threshold for acceptable
                                          numerical deviation from standard torchscript
                                          output  (default 1e-5)
        --no-threshold-check              Skip checking threshold compliance
        --truncate-long-double,
        --truncate, --truncate-64bit      Truncate weights that are provided in
                                          64bit to 32bit (Long, Double to Int,
                                          Float)
        --save-engine                     Instead of compiling a full a
                                          TorchScript program, save the created
                                          engine to the path specified as the
                                          output path
        --custom-torch-ops                (repeatable) Shared object/DLL containing custom torch operators
        --custom-converters               (repeatable) Shared object/DLL containing custom converters
        input_file_path                   Path to input TorchScript file
        output_file_path                  Path for compiled TorchScript (or
                                          TensorRT engine) file
        input_specs...                    Specs for inputs to engine, can either
                                          be a single size or a range defined by
                                          Min, Optimal, Max sizes, e.g.
                                          "(N,..,C,H,W)"
                                          "[(MIN_N,..,MIN_C,MIN_H,MIN_W);(OPT_N,..,OPT_C,OPT_H,OPT_W);(MAX_N,..,MAX_C,MAX_H,MAX_W)]".
                                          Data Type and format can be specified by
                                          adding an "@" followed by dtype and "%"
                                          followed by format to the end of the
                                          shape spec. e.g. "(3, 3, 32,
                                          32)@f16%NHWC"
        "--" can be used to terminate flag options and force all following
        arguments to be treated as positional options

e.g.

.. code-block:: shell

    torchtrtc tests/modules/ssd_traced.jit.pt ssd_trt.ts "[(1,3,300,300); (1,3,512,512); (1, 3, 1024, 1024)]@f16%contiguous" -p f16

- To include a set of custom operators

.. code-block:: shell

    torchtrtc tests/modules/ssd_traced.jit.pt ssd_trt.ts --custom-torch-ops=<path to custom library .so file> "[(1,3,300,300); (1,3,512,512); (1, 3, 1024, 1024)]@fp16%contiguous" -p f16


- To include a set of custom converters

.. code-block:: shell

    torchtrtc tests/modules/ssd_traced.jit.pt ssd_trt.ts --custom-converters=<path to custom library .so file> "[(1,3,300,300); (1,3,512,512); (1, 3, 1024, 1024)]@fp16%contiguous" -p f16
