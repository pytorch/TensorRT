.. _runtime:

Deploying Torch-TensorRT Programs
====================================

After compiling and saving Torch-TensorRT programs there is no longer a strict dependency on the full
Torch-TensorRT library. All that is required to run a compiled program is the runtime. There are therfore a couple
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

.. note:: If you are linking ``libtorchtrt_runtime.so``, likely using the following flags will help ``-Wl,--no-as-needed -ltorchtrt -Wl,--as-needed`` as theres no direct symbol dependency to anything in the Torch-TensorRT runtime for most Torch-TensorRT runtime applications

An example of how to use ``libtorchtrt_runtime.so`` can be found here: https://github.com/pytorch/TensorRT/tree/master/examples/torchtrt_runtime_example

Plugin Library
---------------

In the case you use Torch-TensorRT as a converter to a TensorRT engine and your engine uses plugins provided by Torch-TensorRT, Torch-TensorRT
ships the library ``libtorchtrt_plugins.so`` which contains the implementation of the TensorRT plugins used by Torch-TensorRT during
compilation. This library can be ``DL_OPEN`` or ``LD_PRELOAD`` similar to other TensorRT plugin libraries.
