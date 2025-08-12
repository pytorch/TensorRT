.. _Torch-TensorRT_in_RTX:

Torch-TensorRT in RTX
=====================

Overview
--------

TensorRT-RTX
~~~~~~~~~~~~

TensorRT for RTX builds on the proven performance of the NVIDIA TensorRT inference library, and simplifies the deployment of AI models on NVIDIA RTX GPUs across desktops, laptops, and workstations.

TensorRT for RTX is a drop-in replacement for NVIDIA TensorRT in applications targeting NVIDIA RTX GPUs from Turing through Blackwell generations. It introduces a Just-In-Time (JIT) optimizer in the runtime that compiles improved inference engines directly on the end-user’s RTX-accelerated PC in under 30 seconds. This eliminates the need for lengthy pre-compilation steps and enables rapid engine generation, improved application portability, and cutting-edge inference performance.

For detailed information about TensorRT-RTX, refer to:

* `TensorRT-RTX Documentation <https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html>`_

Currently, Torch-TensorRT only supports TensorRT-RTX for experimental purposes.  
Torch-TensorRT by default uses standard TensorRT during the build and run.

To use TensorRT-RTX:

- Build the wheel with the ``--use-rtx`` flag or set ``USE_TRT_RTX=true``.
- During runtime, set the ``USE_TRT_RTX=true`` environment variable to invoke TensorRT-RTX.

Prerequisites
-------------

Install Bazel
~~~~~~~~~~~~~

Bazel is required to build the wheel with TensorRT-RTX.

**In Linux:**

.. code-block:: sh

   curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-amd64 \
    -o bazelisk \
    && mv bazelisk /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

**In Windows:**

.. code-block:: sh

   choco install bazelisk -y

Install TensorRT-RTX Tarball
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorRT-RTX tarball can be downloaded from https://developer.nvidia.com/tensorrt-rtx.  
Currently, Torch-TensorRT uses TensorRT-RTX version **1.0.0.21**.

Once downloaded:

**In Linux:**

Make sure you add the lib path to the ``LD_LIBRARY_PATH`` environment variable.

.. code-block:: sh

    # If TensorRT-RTX is downloaded in /your_local_download_path/TensorRT-RTX-1.0.0.21
    export LD_LIBRARY_PATH=/your_local_download_path/TensorRT-RTX-1.0.0.21/lib:$LD_LIBRARY_PATH
    echo $LD_LIBRARY_PATH | grep TensorRT-RTX

**In Windows:**

Make sure you add the lib path to the Windows system variable ``PATH``.

.. code-block:: sh

    # If TensorRT-RTX is downloaded in C:\your_local_download_path\TensorRT-RTX-1.0.0.21
    set PATH="%PATH%;C:\your_local_download_path\TensorRT-RTX-1.0.0.21\lib"
    echo %PATH% | findstr TensorRT-RTX

Install TensorRT-RTX Wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, the `tensorrt_rtx` wheel is not published on PyPI.  
You must install it manually from the downloaded tarball.

.. code-block:: sh

   # If the tarball is downloaded in /your_local_download_path/TensorRT-RTX-1.0.0.21
   python -m pip install /your_local_download_path/TensorRT-RTX-1.0.0.21/python/tensorrt_rtx-1.0.0.21-cp39-none-linux_x86_64.whl

Build Torch-TensorRT with TensorRT-RTX
--------------------------------------

Build Locally with TensorRT-RTX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

    # If you have previously built with standard TensorRT, make sure to clean the build environment,
    # otherwise it will use the existing .so built with standard TensorRT, which is not compatible with TensorRT-RTX.
    python setup.py clean
    bazel clean --expunge
    #remove everything under build directory, 
    rm -rf build/*

    # Build wheel with TensorRT-RTX
    python setup.py bdist_wheel --use-rtx

    # Install the wheel
    python -m pip install dist/torch-tensorrt-*.whl

Quick Start
-----------

.. code-block:: python

    # You must set USE_TRT_RTX=true to use TensorRT-RTX
    USE_TRT_RTX=true python examples/dynamo/torch_compile_resnet_example.py

Troubleshooting
---------------

If you encounter load or link errors, check if `tensorrt_rtx` is linked correctly.  
If not, clean up the environment and rebuild.

**In Linux:**

.. code-block:: sh

    # Ensure only tensorrt_rtx is installed (no standard tensorrt wheels)
    python -m pip list | grep tensorrt 

    # Check if libtorchtrt.so links to the correct tensorrt_rtx shared object
    trt_install_path=$(python -m pip show torch-tensorrt | grep "Location" | awk '{print $2}')/torch_tensorrt

    # Verify libtensorrt_rtx.so.1 is linked, and libnvinfer.so.10 is NOT
    ldd $trt_install_path/lib/libtorchtrt.so

**In Windows:**

.. code-block:: sh

    # Check if tensorrt_rtx_1_0.dll is linked, and libnvinfer.dll is NOT
    cd py/torch_tensorrt
    dumpbin /DEPENDENTS torchtrt.dll
