#!/bin/bash

# Install dependencies
python3 -m pip install pyyaml
yum install -y ninja-build gettext
TRT_VERSION=$(python3 -c "import versions; versions.tensorrt_version()")
wget -P /opt/torch-tensorrt-builds/ https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.0/TensorRT-10.0.0.6.Linux.x86_64-gnu.cuda-12.4.tar.gz
tar -xvzf /opt/torch-tensorrt-builds/TensorRT-10.0.0.6.Linux.x86_64-gnu.cuda-12.4.tar.gz -C /opt/torch-tensorrt-builds/
export LD_LIBRARY_PATH=/opt/torch-tensorrt-builds/TensorRT-10.0.0.6/lib:$LD_LIBRARY_PATH
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel
python -m pip install -r py/requirements.txt
export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")

cat toolchains/ci_workspaces/WORKSPACE.x86_64.release.rhel.tmpl | envsubst > WORKSPACE
export CI_BUILD=1
