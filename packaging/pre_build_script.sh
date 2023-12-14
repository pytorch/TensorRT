#!/bin/bash

# Install dependencies
python3 -m pip install pyyaml
TRT_VERSION=$(python3 -c "import versions; versions.tensorrt_version()")
CUDA_MAJOR_VERSION=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])")

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum check-update
yum install -y ninja-build gettext tensorrt-${TRT_VERSION}.*.cuda${CUDA_MAJOR_VERSION}.*
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")

cat toolchains/ci_workspaces/WORKSPACE.x86_64.release.rhel.tmpl | envsubst > WORKSPACE
export CI_BUILD=1

echo $LD_LIBRARY_PATH
