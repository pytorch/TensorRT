#!/bin/bash
set -o nounset
set -o errexit
set -o pipefail
set -e

post=${1:-""}

# fetch bazel executable
BAZEL_VERSION=4.2.1
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then ARCH="arm64"; fi
wget -q https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${ARCH} -O /usr/bin/bazel
chmod a+x /usr/bin/bazel
export NVIDIA_TF32_OVERRIDE=0

cd /opt/pytorch/torch_tensorrt
cp /opt/pytorch/torch_tensorrt/docker/WORKSPACE.docker  /opt/pytorch/torch_tensorrt/WORKSPACE

pip install --user --upgrade nox
TOP_DIR=/opt/pytorch/torch_tensorrt nox
