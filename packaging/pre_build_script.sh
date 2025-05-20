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
    # install torch 2.7 torchvision 0.22.0 for jp6.2
    TORCH_URL=https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/6ef/f643c0a7acda9/torch-2.7.0-cp310-cp310-linux_aarch64.whl#sha256=6eff643c0a7acda92734cc798338f733ff35c7df1a4434576f5ff7c66fc97319
    TORCHVISION_URL=https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/daa/bff3a07259968/torchvision-0.22.0-cp310-cp310-linux_aarch64.whl#sha256=daabff3a0725996886b92e4b5dd143f5750ef4b181b5c7d01371a9185e8f0402
    pip install ${TORCH_URL}
    pip install ${TORCHVISION_URL}
else
    TORCH_TORCHVISION=$(grep "^torch" py/requirements.txt)
    INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}

    # Install all the dependencies required for Torch-TensorRT
    pip install --force-reinstall --pre ${TORCH_TORCHVISION} --index-url ${INDEX_URL}
fi

export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")
export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")

if [[ ${IS_JETPACK} == true ]]; then
    # change the torch dependency for jp6.2
    sed -i -e "s/torch>=2.8.0.dev,<2.9.0/torch>=2.7.0,<2.8.0/g" pyproject.toml
    # change the tensorrt dependency for jp6.2
    sed -i -e "s/tensorrt>=10.9.0,<10.10.0/tensorrt>=10.3.0,<10.4.0/g" pyproject.toml
    sed -i -e "s/\"tensorrt-cu12>=10.9.0,<10.10.0\",//g" pyproject.toml
    sed -i -e "s/\"tensorrt-cu12-bindings>=10.9.0,<10.10.0\",//g" pyproject.toml
    sed -i -e "s/\"tensorrt-cu12-libs>=10.9.0,<10.10.0\",//g" pyproject.toml
    # downgrade the numpy dependency for jp6.2
    sed -i -e "s/\"numpy\"/\"numpy==1.26.3\"/g" pyproject.toml
else
    # for non-jetpack, we need to support on cuda 118
    if [[ "${CU_VERSION::4}" < "cu12" ]]; then
        # replace dependencies from tensorrt-cu12-bindings/libs to tensorrt-cu11-bindings/libs
        sed -i -e "s/tensorrt-cu12/tensorrt-${CU_VERSION::4}/g" \
            -e "s/tensorrt-cu12-bindings/tensorrt-${CU_VERSION::4}-bindings/g" \
            -e "s/tensorrt-cu12-libs/tensorrt-${CU_VERSION::4}-libs/g" \
            pyproject.toml
    fi
    # for non-jetpack, we need to support build with different tensorrt version
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
fi

cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel

if [[ ${TENSORRT_VERSION} != "" ]]; then
    sed -i -e "s/strip_prefix = \"TensorRT-.*\"/strip_prefix = \"TensorRT-${TENSORRT_VERSION}\"/g" MODULE.bazel
    sed -i -e "s#\"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/.*\"#\"${TENSORRT_URLS}\"#g" MODULE.bazel
fi
cat MODULE.bazel
cat pyproject.toml
export CI_BUILD=1
