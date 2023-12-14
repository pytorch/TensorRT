#!/bin/bash

# Install dependencies
python3 -m pip install pyyaml
TRT_VERSION=$(python3 -c "import versions; versions.tensorrt_version()")
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
CUDA_MAJOR_VERSION=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])")

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum check-update
yum install -y ninja-build gettext tensorrt-${TRT_VERSION}.*.cuda${CUDA_MAJOR_VERSION}.*
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
rm -fr /usr/local/cuda
ln -s /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda

export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")

cat toolchains/ci_workspaces/WORKSPACE.x86_64.release.rhel.tmpl | envsubst > WORKSPACE
export CI_BUILD=1

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
