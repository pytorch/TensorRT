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
pip install --pre -r tests/py/requirements.txt --use-deprecated legacy-resolver


export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")
export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")

if [[ "${CU_VERSION::4}" < "cu12" ]]; then
  # replace dependencies from tensorrt-cu12-bindings/libs to tensorrt-cu11-bindings/libs
  sed -i -e "s/tensorrt-cu12==/tensorrt-${CU_VERSION::4}==/g" \
         -e "s/tensorrt-cu12-bindings==/tensorrt-${CU_VERSION::4}-bindings==/g" \
         -e "s/tensorrt-cu12-libs==/tensorrt-${CU_VERSION::4}-libs==/g" \
         pyproject.toml
  cat pyproject.toml
fi

cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel

cat MODULE.bazel
export CI_BUILD=1
