#!/bin/bash

# Install dependencies
TRT_VERSION=$(python3 -c "import versions; versions.tensorrt_version()")
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum check-update
yum install -y ninja-build tensorrt-${TRT_VERSION}.*
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

cp toolchains/ci_workspaces/WORKSPACE.x86_64.${VERSION_SUFFIX#*+}.release.rhel WORKSPACE
export CI_BUILD=1
