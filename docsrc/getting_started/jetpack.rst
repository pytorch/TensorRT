.. _Torch_TensorRT_in_JetPack_6.2

Overview
##################

JetPack 6.2
---------------------
Nvida JetPack 6.2 is the latest production release of JetPack 6.
With this release it incorporates:
CUDA 12.6
TensorRT 10.3
cuDNN 9.3
DLFW 25.04

You can find more details for the JetPack 6.2:

    * https://docs.nvidia.com/jetson/jetpack/release-notes/index.html
    * https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html


Prerequisites
~~~~~~~~~~~~~~


Ensure your jetson developer kit has been flashed with the latest JetPack 6.1. You can find more details on how to flash Jetson board via sdk-manager:

    * https://developer.nvidia.com/sdk-manager


check the current jetpack version using

.. code-block:: sh

    apt show nvidia-jetpack

Ensure you have installed JetPack Dev components. This step is required if you need to build on jetson board.

You can only install the dev components that you require: ex, tensorrt-dev would be the meta-package for all TRT development or install everthing.

.. code-block:: sh
    # install all the nvidia-jetpack dev components
    sudo apt-get update
    sudo apt-get install nvidia-jetpack

Ensure you have cuda 12.6 installed(this should be installed automatically from nvidia-jetpack)

.. code-block:: sh

    # check the cuda version
    nvcc --version
    # if not installed or the version is not 12.6, install via the below cmd:
    sudo apt-get update
    sudo apt-get install cuda-toolkit-12-6

Ensure libcusparseLt.so exists at /usr/local/cuda/lib64/:

.. code-block:: sh

    # if not exist, download and copy to the directory
    wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
    tar xf libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
    sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/include/* /usr/local/cuda/include/
    sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib/* /usr/local/cuda/lib64/


Build torch_tensorrt wheel file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

you can build torch_tensorrt wheel file with or without docker.

if you want to build torch_tensorrt wheel file with docker, you can get the docker image from:
* https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch

.. code-block:: sh
# with this image you should have all the jetpack 6.2 environment ready 
# with cuda 12.6 and tensorrt 10.3 pre-installed
sudo docker run  -it  --gpus all --user root --shm-size=10.24g \
-ulimit stack=67108864 --ulimit memlock=-1 --cap-add SYS_ADMIN \
-v $(pwd):/workspace \
-net host nvcr.io/nvidia/l4t-jetpack:r36.4.0 bash


Install bazel

.. code-block:: sh

    wget -v https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-arm64
    sudo mv bazelisk-linux-arm64 /usr/bin/bazel
    sudo chmod +x /usr/bin/bazel

Install pip and required python packages:
    * https://pip.pypa.io/en/stable/installation/

.. code-block:: sh

    # install pip
    wget https://bootstrap.pypa.io/get-pip.py
    python get-pip.py


.. code-block:: sh
    # build torch_tensorrt wheel file
    # make sure you have the latest Torch_tensorrt source code and git configured
    # apt-get install git
    # git config --global --add safe.directory /workspace/TensorRT (your tensorrt source code path)
    export BUILD_VERSION="2.8.0.dev20250511" && ./packaging/pre_build_script.sh
    python setup.py bdist_wheel --jetpack


.. code-block:: sh
    # install torch_tensorrt wheel file built above
    cd dist
    python install torch-tensorrt-2.8.0.dev0+4da152843