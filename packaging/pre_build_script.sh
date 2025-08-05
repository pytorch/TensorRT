#!/bin/bash

set -x

# Install dependencies
python3 -m pip install pyyaml

if [[ $(uname -m) == "aarch64" ]]; then
  IS_AARCH64=true
  BAZEL_PLATFORM=arm64
  os_name=$(cat /etc/os-release | grep -w "ID" | cut -d= -f2)
  if [[ ${os_name} == "ubuntu" ]]; then
      IS_JETPACK=true
      apt-get update
      apt-get install -y ninja-build gettext curl libopenblas-dev
  else
      IS_SBSA=true
      yum install -y ninja-build gettext
  fi
else
  BAZEL_PLATFORM="amd64"
fi


if [[ ${IS_AARCH64} == true ]]; then
    # aarch64 does not have envsubst pre-installed in the image, install it here
    curl -L  https://github.com/a8m/envsubst/releases/download/v1.4.2/envsubst-Linux-arm64 -o envsubst \
    && mv envsubst /usr/bin/envsubst && chmod +x /usr/bin/envsubst
    # install cuda for SBSA
    if [[ ${IS_SBSA} == true ]]; then
        rm -rf /opt/openssl # Not sure whats up with the openssl mismatch
        source .github/scripts/install-cuda-aarch64.sh
        install_cuda_aarch64
    fi
fi

curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-${BAZEL_PLATFORM} \
    -o bazelisk-linux-${BAZEL_PLATFORM} \
    && mv bazelisk-linux-${BAZEL_PLATFORM} /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

pip uninstall -y torch torchvision

if [[ ${IS_JETPACK} == true ]]; then
    # install torch 2.7 for jp6.2
    pip install torch==2.7.0 --index-url=https://pypi.jetson-ai-lab.dev/jp6/cu126/
else
    TORCH=$(grep "^torch>" py/requirements.txt)
    INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}

    # Install all the dependencies required for Torch-TensorRT
    pip install --force-reinstall --pre ${TORCH} --index-url ${INDEX_URL}
fi

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

cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel

if [[ ${TENSORRT_VERSION} != "" ]]; then
    sed -i -e "s/strip_prefix = \"TensorRT-.*\"/strip_prefix = \"${TENSORRT_STRIP_PREFIX}\"/g" MODULE.bazel
    sed -i -e "s#\"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/.*\"#\"${TENSORRT_URLS}\"#g" MODULE.bazel
fi

cat MODULE.bazel
export CI_BUILD=1

if [[ ${USE_RTX} == true ]]; then
    cat pyproject_rtx.toml.temp > pyproject.toml
fi