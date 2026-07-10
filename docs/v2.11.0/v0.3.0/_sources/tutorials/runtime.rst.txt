.. _runtime:

Deploying TRTorch Programs
===========================

After compiling and saving TRTorch programs there is no longer a strict dependency on the full
TRTorch library. All that is required to run a compiled program is the runtime. There are therfore a couple
options to deploy your programs other than shipping the full trtorch compiler with your applications.

TRTorch package / libtrtorch.so
---------------------------------

Once a program is compiled, you run it using the standard PyTorch APIs. All that is required is that the package
must be imported in python or linked in C++.

Runtime Library
-----------------

Distributed with the C++ distribution is ``libtrtorchrt.so``. This library only contains the components
necessary to run TRTorch programs. Instead of linking ``libtrtorch.so`` or importing ``trtorch`` you can
link ``libtrtorchrt.so`` in your deployment programs or use ``DL_OPEN`` or ``LD_PRELOAD``. For python
you can load the runtime with ``torch.ops.load_library("libtrtorchrt.so")``. You can then continue to use
programs just as you would otherwise via PyTorch API.

.. note:: If you are using the standard distribution of PyTorch in Python on x86, likely you will need the pre-cxx11-abi variant of ``libtrtorchrt.so``, check :ref:`Installation` documentation for more details.