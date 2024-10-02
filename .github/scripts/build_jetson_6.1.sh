#!/usr/bin/env bash

# how to run the jetson build on jetpack6.1:
# ./.github/scripts/build_jetson_6.1.sh

set -euxo pipefail

# get jetpack version: eg: Version: 6.1+b123 ---> 6.1
jetpack_version=$(apt show nvidia-jetpack 2>/dev/null | grep Version: | cut -d ' ' -f 2 | cut -d '+' -f 1)
python_version=$(python --version)
cuda_version=$(nvcc --version | grep Cuda | grep release | cut -d ',' -f 2 | sed -e 's/ release //g')
echo "Current jetpack_version: ${jetpack_version} cuda_version: ${cuda_version} python_version: ${python_version} "

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/aarch64-linux-gnu:/usr/include/aarch64-linux-gnu:/usr/local/cuda-${cuda_version}/lib64

# make sure nvidia-jetpack dev package is installed:
# go to /usr/include/aarch64-linux-gnu/ if you can see NvInfer.h(tensorrt related header files) which means dev package is installed
# if not installed, install via the below cmd:
# sudo apt update
# sudo apt install nvidia-jetpack

# make sure cuda is installed:
# nvcc --version or go to /usr/local/cuda/bin to see whether it is installed
# the install nvidia-jetpack dev package step will automatically install the cuda toolkit
# if not installed, install via the below cmd:
# sudo apt update
# sudo apt install cuda-toolkit-12-6

# make sure bazel is installed via the below cmd:
# wget -v https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-arm64
# sudo mv bazelisk-linux-arm64 /usr/bin/bazel
# chmod +x /usr/bin/bazel

# make sure pip is installed:
# sudo apt install python3-pip

# make sure setuptools is installed with the version < 71.*.*
# version 71.*.* will give the following error during build
# TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
# python -m pip install setuptools==70.2.0

# make sure torch is installed via the below cmd:
# wget https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
# python -m pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# make sure libcusparseLt.so exists if not download and copy via the below cmd:
# wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
# tar xf libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
# sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/include/* /usr/local/cuda/include/
# sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib/* /usr/local/cuda/lib64/

export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
export SITE_PACKAGE_PATH=${TORCH_INSTALL_PATH::-6}
export CUDA_HOME=/usr/local/cuda-${cuda_version}/

# replace the Module file with jetpack one
cat toolchains/jp_workspaces/MODULE.bazel.jp61 | envsubst > MODULE.bazel

# build on jetpack
python setup.py  --use-cxx11-abi  install --user

