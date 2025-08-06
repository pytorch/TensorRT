.. _Torch-TensorRT_in_RTX:

Torch-TensorRT in RTX
#############################

Overview
********

TensorRT-RTX
===========
TensorRT for RTX builds on the proven performance of the NVIDIA TensorRT inference library, and simplifies the deployment of AI models on NVIDIA RTX GPUs across desktops, laptops, and workstations.

TensorRT for RTX is a drop-in replacement for NVIDIA TensorRT in applications targeting NVIDIA RTX GPUs from Turing through Blackwell generations. It introduces a Just-In-Time (JIT) optimizer in the runtime that compiles improved inference engines directly on the end-user’s RTX-accelerated PC in under 30 seconds. This eliminates the need for lengthy pre-compilation steps and enables rapid engine generation, improved application portability, and cutting-edge inference performance.

For detailed information about TensorRT-RTX, refer to:
* `TensorRT-RTX Documentation <https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html>`_

Currenlty, Torch-TensorRT only supports TensorRT-RTX for the experiment purpose.
Torch-TensorRT by default uses TensorRT during the build and run.

In order to use TensorRT-RTX, you need to build the wheel with ``--use-rtx`` flag.
And then set the ``USE_TRT_RTX=1`` environment variable during run.




Prerequisites
*************

System Preparation
==================
1. **Install TensorRT-RTX**:
   TensorRT-RTX can be downloaded from https://developer.nvidia.com/tensorrt-rtx.
   .. code-block:: sh
    # if TensorRT-RTX is downloaded in /usr/local/tensorrt-rtx
   export LD_LIBRARY_PATH=/usr/local/tensorrt-rtx/lib:$LD_LIBRARY_PATH


Build Torch-TensorRT with TensorRT-RTX
=====================================

.. code-block:: sh
    # if you have previously build with Standard TensorRT, make sure to clean the build environment
    python setup.py clean
    # build wheel with TensorRT-RTX
    python setup.py bdist_wheel --use-rtx

    # install the wheel
    cd dist
    python -m pip install torch-tensorrt-*.whl

    # make sure the tensorrt_rtx.so file is linked to the tensorrt_rtx.so file in the TensorRT-RTX installation directory
    trt_install_path=$(python -m pip show torch-tensorrt | grep "Location" | awk '{print $2}')/torch_tensorrt

    # check if the libtensorrt_rtx.so.1 is linked
    ldd $trt_install_path/lib/libtorchtrt.so


Quick Start
===========

.. code-block:: py
    # you have to set USE_TRT_RTX=1 to use TensorRT-RTX
    USE_TRT_RTX=1 python examples/dynamo/torch_compile_resnet_example.py

