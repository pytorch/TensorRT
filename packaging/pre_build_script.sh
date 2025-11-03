#!/bin/bash

set -x

# Install dependencies
python3 -m pip install pyyaml

# CU_VERSION: cu130, cu129, etc.
# CU_MAJOR_VERSION: 13, 12, etc.
# CU_MINOR_VERSION: 0, 9, etc.
CU_MAJOR_VERSION=${CU_VERSION:2:4}
CU_MINOR_VERSION=${CU_VERSION:4:5}
ctk_name="cuda-toolkit-${CU_MAJOR_VERSION}-${CU_MINOR_VERSION}"

if [[ $(uname -m) == "aarch64" ]]; then
  IS_AARCH64=true
  BAZEL_PLATFORM=arm64
  os_name=$(cat /etc/os-release | grep -w "ID" | cut -d= -f2)
  if [[ ${os_name} == "ubuntu" ]]; then
      IS_JETPACK=true
      apt-get update
      apt-get install -y ninja-build gettext curl libopenblas-dev zip unzip libfmt-dev ${ctk_name}
  else
      IS_SBSA=true
      yum install -y ninja-build gettext zip unzip
      yum install -y fmt-devel ${ctk_name}
  fi
else
  BAZEL_PLATFORM="amd64"
  yum install -y fmt-devel ${ctk_name}
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
    # install torch 2.8 for jp6.2
    source .github/scripts/install-cuda-dss.sh
    install_cuda_dss_aarch64
    pip install torch==2.8.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126/
else
    TORCH=$(grep "^torch>" py/requirements.txt)
    INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}

    # Install all the dependencies required for Torch-TensorRT
    pip install --force-reinstall --pre ${TORCH} --index-url ${INDEX_URL}
fi

export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")
export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")

# CU_UPPERBOUND eg:13.0 or 12.9
# tensorrt tar for linux and windows are different across cuda version
# for sbsa it is the same tar across cuda version
if [[ ${CU_MAJOR_VERSION} == "13" ]]; then
    export CU_UPPERBOUND="13.0"
else
    export CU_UPPERBOUND="12.9"
fi

cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel

if [[ ${TENSORRT_VERSION} != "" ]]; then
    sed -i -e "s/strip_prefix = \"TensorRT-.*\"/strip_prefix = \"${TENSORRT_STRIP_PREFIX}\"/g" MODULE.bazel
    sed -i -e "s#\"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/.*\"#\"${TENSORRT_URLS}\"#g" MODULE.bazel
fi

cat MODULE.bazel
export CI_BUILD=1

if [[ ${USE_TRT_RTX} == true ]]; then
    source .github/scripts/install-tensorrt-rtx.sh
    install_wheel_or_not=true
    install_tensorrt_rtx ${install_wheel_or_not}
fi