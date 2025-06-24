.. _runtime:

Deploying Torch-TensorRT Programs
====================================

After compiling and saving Torch-TensorRT programs there is no longer a strict dependency on the full
Torch-TensorRT library. All that is required to run a compiled program is the runtime. There are therefore a couple
options to deploy your programs other than shipping the full Torch-TensorRT compiler with your applications.

Torch-TensorRT package / libtorchtrt.so
--------------------------------------------

Once a program is compiled, you run it using the standard PyTorch APIs. All that is required is that the package
must be imported in python or linked in C++.

Runtime Library
-----------------

Distributed with the C++ distribution is ``libtorchtrt_runtime.so``. This library only contains the components
necessary to run Torch-TensorRT programs. Instead of linking ``libtorchtrt.so`` or importing ``torch_tensorrt`` you can
link ``libtorchtrt_runtime.so`` in your deployment programs or use ``DL_OPEN`` or ``LD_PRELOAD``. For python
you can load the runtime with ``torch.ops.load_library("libtorchtrt_runtime.so")``. You can then continue to use
programs just as you would otherwise via PyTorch API.

.. note:: If you are linking ``libtorchtrt_runtime.so``, likely using the following flags will help ``-Wl,--no-as-needed -ltorchtrt -Wl,--as-needed`` as there's no direct symbol dependency to anything in the Torch-TensorRT runtime for most Torch-TensorRT runtime applications

An example of how to use ``libtorchtrt_runtime.so`` can be found here: https://github.com/pytorch/TensorRT/tree/master/examples/torchtrt_aoti_example

Plugin Library
---------------

In the case you use Torch-TensorRT as a converter to a TensorRT engine and your engine uses plugins provided by Torch-TensorRT, Torch-TensorRT
ships the library ``libtorchtrt_plugins.so`` which contains the implementation of the TensorRT plugins used by Torch-TensorRT during
compilation. This library can be ``DL_OPEN`` or ``LD_PRELOAD`` similarly to other TensorRT plugin libraries.

Multi Device Safe Mode
---------------

Multi-device safe mode is a setting in Torch-TensorRT which allows the user to determine whether
the runtime checks for device consistency prior to every inference call.

There is a non-negligible, fixed cost per-inference call when multi-device safe mode is enabled, which is why
it is now disabled by default. It can be controlled via the following convenience function which
doubles as a context manager.

.. code-block:: python

    # Enables Multi Device Safe Mode
    torch_tensorrt.runtime.set_multi_device_safe_mode(True)

    # Disables Multi Device Safe Mode [Default Behavior]
    torch_tensorrt.runtime.set_multi_device_safe_mode(False)

    # Enables Multi Device Safe Mode, then resets the safe mode to its prior setting
    with torch_tensorrt.runtime.set_multi_device_safe_mode(True):
        ...

TensorRT requires that each engine be associated with the CUDA context in the active thread from which it is invoked.
Therefore, if the device were to change in the active thread, which may be the case when invoking
engines on multiple GPUs from the same Python process, safe mode will cause Torch-TensorRT to display
an alert and switch GPUs accordingly. If safe mode is not enabled, there could be a mismatch in the engine
device and CUDA context device, which could lead the program to crash.

One technique for managing multiple TRT engines on different GPUs while not sacrificing performance for
multi-device safe mode is to use Python threads. Each thread is responsible for all of the TRT engines
on a single GPU, and the default CUDA device on each thread corresponds to the GPU for which it is
responsible (can be set via ``torch.cuda.set_device(...)``). In this way, multiple threads can be used in the same
Python script without needing to switch CUDA contexts and incur performance overhead.

Cudagraphs Mode
---------------

Cudagraphs mode is a setting in Torch-TensorRT which allows the user to determine whether
the runtime uses cudagraphs to accelerate inference in certain cases.

Cudagraphs can accelerate certain models by reducing kernel overheads, as documented further [here](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

.. code-block:: python

    # Enables Cudagraphs Mode
    torch_tensorrt.runtime.set_cudagraphs_mode(True)

    # Disables Cudagraphs Mode [Default Behavior]
    torch_tensorrt.runtime.set_cudagraphs_mode(False)

    # Enables Cudagraphs Mode, then resets the mode to its prior setting
    with torch_tensorrt.runtime.enable_cudagraphs(trt_module):
        ...

In the current implementation, use of a new input shape (for instance in dynamic shape
cases), will cause the cudagraph to be re-recorded. Cudagraph recording is generally
not latency intensive, and future improvements include caching cudagraphs for multiple input shapes.

Dynamic Output Allocation Mode
------------------------------

Dynamic output allocation is a feature in Torch-TensorRT which allows the output buffer of TensorRT engines to be
dynamically allocated. This is useful for models with dynamic output shapes, especially ops with data-dependent shapes.
Dynamic output allocation mode cannot be used in conjunction with CUDA Graphs nor pre-allocated outputs feature.
Without dynamic output allocation, the output buffer is allocated based on the inferred output shape based on input size.

There are two scenarios in which dynamic output allocation is enabled:

1. The model has been identified at compile time to require dynamic output allocation for at least one TensorRT subgraph.
These models will engage the runtime mode automatically (with logging) and are incompatible with other runtime modes
such as CUDA Graphs.

Converters can declare that subgraphs that they produce will require the output allocator using `requires_output_allocator=True`
there by forcing any model which utilizes the converter to automatically use the output allocator runtime mode. e.g.,

.. code-block:: python

    @dynamo_tensorrt_converter(
        torch.ops.aten.nonzero.default,
        supports_dynamic_shapes=True,
        requires_output_allocator=True,
    )
    def aten_ops_nonzero(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[TRTTensor, Sequence[TRTTensor]]:
        ...

2. Users may manually enable dynamic output allocation mode via the ``torch_tensorrt.runtime.enable_output_allocator`` context manager.

.. code-block:: python

    # Enables Dynamic Output Allocation Mode, then resets the mode to its prior setting
    with torch_tensorrt.runtime.enable_output_allocator(trt_module):
        ...

Deploying Torch-TensorRT Programs without Python
--------------------------------------------------------

AOT-Inductor
~~~~~~~~~~~~~~~~

AOTInductor is a specialized version of TorchInductor, designed to process exported PyTorch models, optimize them, and produce shared
libraries as well as other relevant artifacts. These compiled artifacts are specifically crafted for deployment in non-Python environments,
which are frequently employed for inference deployments on the server side.

Torch-TensorRT is able to accelerate subgraphs within AOTInductor exports in the same way it does in Python.

.. code-block:: py

    dynamo_model = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=[...])
    torch_tensorrt.save(
        dynamo_model,
        file_path=os.path.join(os.getcwd(), "model.pt2"),
        output_format="aot_inductor",
        retrace=True,
        arg_inputs=[...],
    )

This artifact then can be loaded in a C++ application to be executed with out a Python dependency.

.. code-block:: c++

    #include <iostream>
    #include <vector>

    #include "torch/torch.h"
    #include "torch/csrc/inductor/aoti_package/model_package_loader.h"

    int main(int argc, const char* argv[]) {
    // Check for correct number of command-line arguments
    std::string trt_aoti_module_path = "model.pt2";

    if (argc == 2) {
        trt_aoti_module_path = argv[1];
    }

        std::cout << trt_aoti_module_path << std::endl;

        // Get the path to the TRT AOTI model package from the command line
        c10::InferenceMode mode;

        torch::inductor::AOTIModelPackageLoader loader(trt_aoti_module_path);
        // Assume running on CUDA
        std::vector<torch::Tensor> inputs = {torch::randn({8, 10}, at::kCUDA)};
        std::vector<torch::Tensor> outputs = loader.run(inputs);
        std::cout << "Result from the first inference:"<< std::endl;
        std::cout << outputs << std::endl;

        // The second inference uses a different batch size and it works because we
        // specified that dimension as dynamic when compiling model.pt2.
        std::cout << "Result from the second inference:"<< std::endl;
        // Assume running on CUDA
        std::cout << loader.run({torch::randn({1, 10}, at::kCUDA)}) << std::endl;

        return 0;
    }

Note: Similar to Python, at runtime, no Torch-TensorRT APIs are used to operate the model. Therefore typically additional
flags are needed to make sure that ``libtorchtrt_runtime.so`` gets optimized out (see above).

See: ``//examples/torchtrt_aoti_example`` for a full end to end demo of this workflow


TorchScript
~~~~~~~~~~~~~~

TorchScript is a legacy compiler stack for PyTorch that includes a Python-less interpreter for TorchScript programs.
It has historically been used by Torch-TensorRT to execute models without Python. Even after the transition to TorchDynamo,
the TorchScript interpreter can continue to be used to run PyTorch models with TensorRT engines outside of Python.

.. code-block:: py

    dynamo_model = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=[...])
    ts_model = torch.jit.trace(dynamo_model, inputs=[...])
    torch.jit.save(ts_model, os.path.join(os.getcwd(), "model.ts"),)

This artifact then can be loaded in a C++ application to be executed with out a Python dependency.

.. code-block:: c++

    #include <fstream>
    #include <iostream>
    #include <memory>
    #include <sstream>
    #include <vector>
    #include "torch/script.h"

    int main(int argc, const char* argv[]) {
        if (argc < 2) {
            std::cerr << "usage: samplertapp <path-to-pre-built-trt-ts module>\n";
            return -1;
        }

        std::string trt_ts_module_path = argv[1];

        torch::jit::Module trt_ts_mod;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            trt_ts_mod = torch::jit::load(trt_ts_module_path);
        } catch (const c10::Error& e) {
            std::cerr << "error loading the model from : " << trt_ts_module_path << std::endl;
            return -1;
        }

        std::cout << "Running TRT engine" << std::endl;
        std::vector<torch::jit::IValue> trt_inputs_ivalues;
        trt_inputs_ivalues.push_back(at::randint(-5, 5, {1, 3, 5, 5}, {at::kCUDA}).to(torch::kFloat32));
        torch::jit::IValue trt_results_ivalues = trt_ts_mod.forward(trt_inputs_ivalues);
        std::cout << "==================TRT outputs================" << std::endl;
        std::cout << trt_results_ivalues << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "TRT engine execution completed. " << std::endl;
    }

Note: Similar to Python, at runtime, no Torch-TensorRT APIs are used to operate the model. Therefore typically additional
flags are needed to make sure that ``libtorchtrt_runtime.so`` gets optimized out (see above).

See: ``//examples/torchtrt_runtime_example`` for a full end to end demo of this workflow
