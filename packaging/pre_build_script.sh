#!/bin/bash

set -x

# Install dependencies
python3 -m pip install pyyaml

install -y ninja-build gettext

PLATFORM="amd64"
PLATFORM=x86_64
BAZEL_PLATFORM=amd64
if [[ $(uname -m) == "aarch64" ]]; then
    PLATFORM=aarch64
    BAZEL_PLATFORM=arm64

    rm -rf /opt/openssl # Not sure whats up with the openssl mismatch
fi

wget https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-${BAZEL_PLATFORM} \
    && mv bazelisk-linux-${BAZEL_PLATFORM} /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

TORCH_TORCHVISION=$(grep "^torch" py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}

# Install all the dependencies required for Torch-TensorRT
pip uninstall -y torch torchvision
pip install --force-reinstall --pre ${TORCH_TORCHVISION} --index-url ${INDEX_URL}

export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")
export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")

if [[ ${TENSORRT_VERSION} != "" ]]; then
  # Replace dependencies in the original pyproject.toml with the current TensorRT version. It is used for CI tests of different TensorRT versions.
  # For example, if the current testing TensorRT version is 10.7.0, but the pyproject.toml tensorrt>=10.8.0,<10.9.0, then the following sed command
  # will replace tensorrt>=10.8.0,<10.9.0 with tensorrt==10.7.0
  sed -i -e "s/tensorrt>=.*,<.*\"/tensorrt>=${TENSORRT_VERSION},<$(echo "${TENSORRT_VERSION}" | awk -F. '{print $1"."$2+1".0"}')\"/g" \
         -e "s/tensorrt-cu12>=.*,<.*\"/tensorrt-cu12>=${TENSORRT_VERSION},<$(echo "${TENSORRT_VERSION}" | awk -F. '{print $1"."$2+1".0"}')\"/g" \
         -e "s/tensorrt-cu12-bindings>=.*,<.*\"/tensorrt-cu12-bindings>=${TENSORRT_VERSION},<$(echo "${TENSORRT_VERSION}" | awk -F. '{print $1"."$2+1".0"}')\"/g" \
         -e "s/tensorrt-cu12-libs>=.*,<.*\"/tensorrt-cu12-libs>=${TENSORRT_VERSION},<$(echo "${TENSORRT_VERSION}" | awk -F. '{print $1"."$2+1".0"}')\"/g" \
         pyproject.toml
fi

if [[ "${CU_VERSION::4}" < "cu12" ]]; then
  # replace dependencies from tensorrt-cu12-bindings/libs to tensorrt-cu11-bindings/libs
  sed -i -e "s/tensorrt-cu12/tensorrt-${CU_VERSION::4}/g" \
         -e "s/tensorrt-cu12-bindings/tensorrt-${CU_VERSION::4}-bindings/g" \
         -e "s/tensorrt-cu12-libs/tensorrt-${CU_VERSION::4}-libs/g" \
         pyproject.toml
fi
curl -L  https://github.com/a8m/envsubst/releases/download/v1.4.2/envsubst-Linux-arm64 -o envsubst
chmod +x envsubst
if [[ ${TENSORRT_VERSION} != "" ]]; then
  cat toolchains/ci_workspaces/MODULE_tensorrt.bazel.tmpl | envsubst > MODULE.bazel
else
  cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel
fi

cat MODULE.bazel
export CI_BUILD=1
