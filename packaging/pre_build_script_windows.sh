set -x

pip install -U numpy packaging pyyaml setuptools wheel

choco install bazelisk -y

echo TENSORRT_VERSION=${TENSORRT_VERSION}

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

TORCH_TORCHVISION=$(grep "^torch" py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}

# Install all the dependencies required for Torch-TensorRT
pip uninstall -y torch torchvision
pip install --force-reinstall --pre ${TORCH_TORCHVISION} --index-url ${INDEX_URL}

export CUDA_HOME="$(echo ${CUDA_PATH} | sed -e 's#\\#\/#g')"
export TORCH_INSTALL_PATH="$(python -c "import torch, os; print(os.path.dirname(torch.__file__))" | sed -e 's#\\#\/#g')"


export TENSORRT_NAME=tensorrt_win
export CUDA_NAME=cuda_win
export TENSORRT_STRIP_PREFIX=TensorRT-10.9.0.34
export TENSORRT_URLS=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/zip/TensorRT-10.9.0.34.Windows.win10.cuda-12.8.zip

cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel


cat MODULE.bazel
echo "RELEASE=1" >> ${GITHUB_ENV}
