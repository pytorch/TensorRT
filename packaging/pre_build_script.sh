#!/bin/bash

set -x

# Install dependencies
python3 -m pip install pyyaml

yum install -y ninja-build gettext

wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

TORCH_TORCHVISION=$(grep "^torch" py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}

# Install all the dependencies required for Torch-TensorRT
pip uninstall -y torch torchvision
pip install --force-reinstall --pre ${TORCH_TORCHVISION} --index-url ${INDEX_URL}

export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")
export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")

if [[ ${TENSORRT_VERSION} != "" ]]; then
  # Replace both in dependencies and build-system.requires
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

if [[ ${TENSORRT_VERSION} != "" ]]; then
  cat toolchains/ci_workspaces/MODULE_tensorrt.bazel.tmpl | envsubst > MODULE.bazel
else
  cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel
fi

cat MODULE.bazel
export CI_BUILD=1
