#!/usr/bin/env bash

set -euxo pipefail

# get jetpack version: eg: Version: 6.0+b106 ---> 6.0
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
# the install nvidia-jetpack dev package step will automatically install the cuda tool
# if not installed, install via the below cmd:
# sudo apt update
# sudo apt install cuda-toolkit-12-2

# make sure bazel is installed via the below cmd:
# wget -v https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-arm64
# sudo mv bazelisk-linux-arm64 /usr/bin/bazel
# chmod +x /usr/bin/bazel

# make sure setuptools is installed
# sudo apt install python3-pip
# make sure setuptools is upgraded via the below cmd:
# pip install -U pip setuptools

# make sure torch is installed via the below cmd:
# wget https://developer.download.nvidia.cn/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl
# python -m pip install torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl

# make sure libcusparseLt.so exists if not download and copy via the below cmd:
# wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
# tar xf libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
# sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/include/* /usr/local/cuda/include/
# sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib/* /usr/local/cuda/lib64/

# make sure tensorrt is upgraded from 8.6.2 to tensorrt10.1.0
# wget http://cuda-repo/release-candidates/Libraries/TensorRT/v10.1/10.1.0.16-e64cb73a/12.4-r550/l4t-aarch64/tar/TensorRT-10.1.0.16.Ubuntu-22.04.aarch64-gnu.cuda-12.4.tar.gz
# gunzip TensorRT-10.1.0.16.Ubuntu-22.04.aarch64-gnu.cuda-12.4.tar.gz
# tar xvf TensorRT-10.1.0.16.Ubuntu-22.04.aarch64-gnu.cuda-12.4.tar

# copy tensorrt 10.1.0 header files to /usr/include/aarch64-linux-gnu/:
# cd ~/Desktop/lan/Downloads/TensorRT-10.1.0.16/include
# sudo cp * /usr/include/aarch64-linux-gnu/

# copy tensorrt 10.1.0 .so files to /usr/lib/aarch64-linux-gnu/:
# cd ~/Desktop/lan/Downloads/TensorRT-10.1.0.16/lib
# sudo cp libnvinfer.so.10.1.0 /usr/lib/aarch64-linux-gnu/
# sudo cp libnvinfer_plugin.so.10.1.0 /usr/lib/aarch64-linux-gnu/
# sudo cp libnvinfer_vc_plugin.so.10.1.0 /usr/lib/aarch64-linux-gnu/
# sudo cp libnvonnxparser.so.10.1.0 /usr/lib/aarch64-linux-gnu/
# sudo cp libnvinfer_lean.so.10.1.0  /usr/lib/aarch64-linux-gnu/
# sudo cp libnvinfer_dispatch.so.10.1.0 /usr/lib/aarch64-linux-gnu/
# sudo cp libnvinfer_builder_resource.so.10.1.0 /usr/lib/aarch64-linux-gnu/

# copy tensorrt 10.1.0 *_static.a and stubs to /usr/lib/aarch64-linux-gnu/:
# cd ~/Desktop/lan/Downloads/TensorRT-10.1.0.16/lib
# sudo cp stub/* /usr/lib/aarch64-linux-gnu/stub/
# sudo cp *_static.a /usr/lib/aarch64-linux-gnu/
# sudo cp libonnx_proto.a /usr/lib/aarch64-linux-gnu/

# create symbolic link under /usr/lib/aarch64-linux-gnu/:
# cd /usr/lib/aarch64-linux-gnu/
# sudo ln -s libnvinfer.so.10.1.0 libnvinfer.so
# sudo ln -s libnvinfer_plugin.so.10.1.0 libnvinfer_plugin.so
# sudo ln -s libnvinfer_vc_plugin.so.10.1.0 libnvinfer_vc_plugin.so
# sudo ln -s libnvonnxparser.so.10.1.0 libnvonnxparser.so
# sudo ln -s libnvinfer_lean.so.10.1.0 libnvinfer_lean.so
# sudo ln -s libnvinfer_dispatch.so.10.1.0 libnvinfer_dispatch.so


export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
export SITE_PACKAGE_PATH=${TORCH_INSTALL_PATH::-6}
# replace the WORKSPACE file with jetpack one
cat WORKSPACE > WORKSPACE.orig
cat toolchains/jp_workspaces/WORKSPACE.jp60 | envsubst > WORKSPACE

# build on jetpack
python setup.py  --use-cxx11-abi  install --user

