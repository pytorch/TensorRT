.. _Torch_TensorRT_in_JetPack:

Torch-TensorRT in JetPack
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
      python -m pip install torch==2.8.0 torchvision==0.23.0  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126


Building the Wheel
==================

.. code-block:: sh
   python setup.py bdist_wheel --jetpack

Installation
============

.. code-block:: sh
   # you will be able to find the wheel in the dist directory, has platform name linux_tegra_aarch64
   cd dist
   python -m pip install torch_tensorrt-2.8.0.dev0+d8318d8fc-cp310-cp310-linux_tegra_aarch64.whl

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
