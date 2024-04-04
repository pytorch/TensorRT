#!/bin/bash

# Install dependencies
python3 -m pip install pyyaml
TRT_VERSION=$(python3 -c "import versions; versions.tensorrt_version()")
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
yum check-update
yum install -y ninja-build gettext tensorrt-${TRT_VERSION}.*
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")

cat toolchains/ci_workspaces/WORKSPACE.x86_64.release.rhel.tmpl | envsubst > WORKSPACE
export CI_BUILD=1
