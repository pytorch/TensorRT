#!/bin/bash

# Install dependencies
python3 -m pip install pyyaml
TRT_VERSION=$(python3 -c "import versions; versions.tensorrt_version()")
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum check-update
yum install -y ninja-build gettext tensorrt-${TRT_VERSION}.*
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-12.2.tar.gz && mkdir -p /usr/tensorrt/lib64 && tar -xzvf tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-12.2.tar.gz -C /usr/tensorrt/lib64 TensorRT-9.2.0.5/targets/x86_64-linux-gnu/lib/stubs --strip-components=5 && export LD_LIBRARY_PATH=/usr/tensorrt/lib64/:$LD_LIBRARY_PATH

export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")

cat toolchains/ci_workspaces/WORKSPACE.x86_64.release.rhel.tmpl | envsubst > WORKSPACE
export CI_BUILD=1
