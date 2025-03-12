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

.. note:: If you are using the standard distribution of PyTorch in Python on x86, likely you will need the pre-cxx11-abi variant of ``libtorchtrt_runtime.so``, check :ref:`Installation` documentation for more details.

.. note:: If you are linking ``libtorchtrt_runtime.so``, likely using the following flags will help ``-Wl,--no-as-needed -ltorchtrt -Wl,--as-needed`` as there's no direct symbol dependency to anything in the Torch-TensorRT runtime for most Torch-TensorRT runtime applications

An example of how to use ``libtorchtrt_runtime.so`` can be found here: https://github.com/pytorch/TensorRT/tree/master/examples/torchtrt_runtime_example

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
Without dynamic output allocation, the output buffer is statically allocated and the size is the maximum possible size 
required by the op. This can lead to inefficient memory usage if the actual output size is smaller than the maximum possible size.

There are two scenarios in which dynamic output allocation is enabled:

1. When the model contains submodules that require a dynamic output allocator at runtime, users don't have to manually enable dynamic output allocation mode.

To specify if a module requires a dynamic output allocator, users can set the ``requires_output_allocator=True`` flag in the ``@dynamo_tensorrt_converter`` decorator of converters. e.g.,

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

2. When users manually enable dynamic output allocation via the ``torch_tensorrt.runtime.enable_output_allocator`` context manager.

.. code-block:: python

    # Enables Dynamic Output Allocation Mode, then resets the mode to its prior setting
    with torch_tensorrt.runtime.enable_output_allocator(trt_module):
        ...
