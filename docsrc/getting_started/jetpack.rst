.. _Torch_TensorRT_in_JetPack_6.2:

Torch-TensorRT in JetPack 6.2
#############################

Overview
********

JetPack 6.2
===========
NVIDIA JetPack 6.2 is the latest production release for Jetson platforms, featuring:
- CUDA 12.6
- TensorRT 10.3
- cuDNN 9.3

For detailed information about JetPack 6.2, refer to:
* `JetPack 6.2 Release Notes <https://docs.nvidia.com/jetson/jetpack/release-notes/index.html>`_
* `PyTorch for Jetson Platform <https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html>`_

Prerequisites
*************

System Preparation
==================
1. **Flash your Jetson device** 

   with JetPack 6.2 using SDK Manager:
   - `SDK Manager Guide <https://developer.nvidia.com/sdk-manager>`_

2. **Verify JetPack installation**:

   .. code-block:: sh
   
      apt show nvidia-jetpack

3. **Install development components**:
   .. code-block:: sh
   
      sudo apt-get update
      sudo apt-get install nvidia-jetpack

4. **Confirm CUDA 12.6 installation**:

   .. code-block:: sh
   
      nvcc --version
      # If missing or incorrect version:
      sudo apt-get install cuda-toolkit-12-6

5. **Validate cuSPARSELt library**:

   .. code-block:: sh
   
      # Check library presence
      ls /usr/local/cuda/lib64/libcusparseLt.so
      
      # Install if missing
      wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
      tar xf libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
      sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/include/* /usr/local/cuda/include/
      sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib/* /usr/local/cuda/lib64/

Building Torch-TensorRT
***********************

Build Environment Setup
=======================
1. **Install Build Dependencies**:

   .. code-block:: sh
   
      wget https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-arm64
      sudo mv bazelisk-linux-arm64 /usr/bin/bazel
      sudo chmod +x /usr/bin/bazel

   .. code-block:: sh
      
      apt-get install ninja-build vim libopenblas-dev git

2. **Install Python dependencies**:

   .. code-block:: sh
      
      wget https://bootstrap.pypa.io/get-pip.py
      python get-pip.py
      python -m pip install pyyaml

3. **Install PyTorch**:

   .. code-block:: sh
      
      # Can only install the torch and torchvision wheel from the JPL repo which is built specifically for JetPack 6.2
      python -m pip install torch==2.7.0 torchvision==0.22.0  --index-url=https://pypi.jetson-ai-lab.dev/jp6/cu126/


Building the Wheel
==================

1. **Configure `MODULE.bazel` for JetPack 6.2**:

   Replace the `http_archive` dependency for libtorch with a `new_local_repository`.

   Comment out:

   .. code-block:: python

      http_archive(
         name = "libtorch",
         build_file = "@//third_party/libtorch:BUILD",
         strip_prefix = "libtorch",
         urls = ["https://download.pytorch.org/libtorch/nightly/cu128/libtorch-cxx11-abi-shared-with-deps-latest.zip"],
      )

   Uncomment or add:

   .. code-block:: python

      new_local_repository(
         name = "libtorch",
         path = "/usr/local/lib/python3.10/dist-packages/torch",
         build_file = "third_party/libtorch/BUILD"
      )

   Confirm the path using:

   .. code-block:: sh

      python -c "import torch, os; print(os.path.dirname(torch.__file__))"

2. **Build the wheel**:

   .. code-block:: sh

      python setup.py bdist_wheel --jetpack

Installation
============

.. code-block:: sh

   cd dist
   python -m pip install torch-tensorrt-2.8.0.dev0+4da152843-cp310-none-linux_aarch64.whl

Post-Installation Verification
==============================

Verify installation by importing in Python:
.. code-block:: python

   # verify whether the torch-tensorrt can be imported
   import torch
   import torch_tensorrt
   print(torch_tensorrt.__version__)

   # verify whether the examples can be run
   python examples/dynamo/torch_compile_resnet_example.py