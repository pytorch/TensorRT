#!/bin/bash

# Install dependencies
python3 -m pip install pyyaml
CUDNN_VERSION=$(python3 -c "import versions; print(versions.__cudnn_version__.split('.')[0])")
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum check-update
yum install -y ninja-build gettext libcudnn${CUDNN_VERSION} libcudnn${CUDNN_VERSION}-devel
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-12.2.tar.gz
mkdir -p /usr/tensorrt
tar -xzvf tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-12.2.tar.gz -C /usr/tensorrt --strip-components=1
mkdir -p /usr/lib
cp /usr/tensorrt/lib/* /usr/lib/ || :
mkdir -p /usr/lib64
cp /usr/tensorrt/lib/* /usr/lib64/ || :
mkdir -p /usr/include
cp /usr/tensorrt/include/* /usr/include/ || :

mkdir -p /usr/lib/aarch64-linux-gnu
cp /usr/tensorrt/targets/aarch64-linux-gnu/lib/* /usr/lib/aarch64-linux-gnu/ || :
mkdir -p /usr/include/aarch64-linux-gnu
cp /usr/tensorrt/targets/aarch64-linux-gnu/include/* /usr/include/aarch64-linux-gnu/ || :

mkdir -p /usr/lib/x86_64-linux-gnu
cp /usr/tensorrt/targets/x86_64-linux-gnu/lib/* /usr/lib/x86_64-linux-gnu/ || :
mkdir -p /usr/include/x86_64-linux-gnu
cp /usr/tensorrt/targets/x86_64-linux-gnu/include/* /usr/include/x86_64-linux-gnu/ || :
# cp /usr/lib/x86_64-linux-gnu/libcudnn* /usr/lib64/ || :

rm tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-12.2.tar.gz
rm -rf /usr/tensorrt

export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")

cat toolchains/ci_workspaces/WORKSPACE.x86_64.release.rhel.tmpl | envsubst > WORKSPACE
export CI_BUILD=1
