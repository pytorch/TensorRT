.. _Torch_TensorRT_in_JetPack_6.2

Overview
##################

JetPack 6.2
---------------------
Nvida JetPack 6.2 is the latest production release ofJetPack 6.
With this release it incorporates:
CUDA 12.6
TensorRT 10.3
cuDNN 9.3
DLFW 24.09

You can find more details for the JetPack 6.2:

    * https://docs.nvidia.com/jetson/jetpack/release-notes/index.html
    * https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html


Prerequisites
~~~~~~~~~~~~~~


Ensure your jetson developer kit has been flashed with the latest JetPack 6.2. You can find more details on how to flash Jetson board via sdk-manager:

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
    wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz
    tar xf libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz
    sudo cp -a libcusparse_lt-linux-aarch64-0.7.1.0-archive/include/* /usr/local/cuda/include/
    sudo cp -a libcusparse_lt-linux-aarch64-0.7.1.0-archive/lib/* /usr/local/cuda/lib64/


Build torch_tensorrt
~~~~~~~~~~~~~~


Install bazel

.. code-block:: sh

    wget -v https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-arm64
    sudo mv bazelisk-linux-arm64 /usr/bin/bazel
    chmod +x /usr/bin/bazel

Install pip and required python packages:
    * https://pip.pypa.io/en/stable/installation/

.. code-block:: sh

    # install pip
    wget https://bootstrap.pypa.io/get-pip.py
    python get-pip.py

.. code-block:: sh

   # install pytorch from nvidia jetson distribution: https://pypi.jetson-ai-lab.dev/jp6/cu126/
    pip3 install torch --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126/

.. code-block:: sh

    # install required python packages
    python -m pip install -r toolchains/jp_workspaces/requirements.txt

    # if you want to run the test cases, then install the test required python packages
    python -m pip install -r toolchains/jp_workspaces/test_requirements.txt


Build and Install torch_tensorrt wheel file


Since torch_tensorrt version has dependencies on torch version. torch version supported by JetPack6.2 is from DLFW 24.08/24.09(torch 2.6.0).

Please make sure to build torch_tensorrt wheel file from source release/2.6 branch
(TODO: lanl to update the branch name once release/ngc branch is available)

.. code-block:: sh

    cuda_version=$(nvcc --version | grep Cuda | grep release | cut -d ',' -f 2 | sed -e 's/ release //g')
    export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
    export SITE_PACKAGE_PATH=${TORCH_INSTALL_PATH::-6}
    export CUDA_HOME=/usr/local/cuda-${cuda_version}/
    # replace the MODULE.bazel with the jetpack one
    cat toolchains/jp_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel
    # build and install torch_tensorrt wheel file
    python setup.py install --user

