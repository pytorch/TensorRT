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

TORCH=$(grep "^torch>" py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}

# Install all the dependencies required for Torch-TensorRT
pip uninstall -y torch torchvision
pip install --force-reinstall --pre ${TORCH} --index-url ${INDEX_URL}

export CUDA_HOME="$(echo ${CUDA_PATH} | sed -e 's#\\#\/#g')"
export TORCH_INSTALL_PATH="$(python -c "import torch, os; print(os.path.dirname(torch.__file__))" | sed -e 's#\\#\/#g')"

cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel

if [[ ${TENSORRT_VERSION} != "" ]]; then
    sed -i -e "s/strip_prefix = \"TensorRT-.*\"/strip_prefix = \"${TENSORRT_STRIP_PREFIX}\"/g" MODULE.bazel
    sed -i -e "s#\"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/.*\"#\"${TENSORRT_URLS}\"#g" MODULE.bazel
fi

cat MODULE.bazel
echo "RELEASE=1" >> ${GITHUB_ENV}

if [[ ${USE_TRT_RTX} == true ]]; then
    source .github/scripts/install-tensorrt-rtx.sh
    install_wheel_or_not=true
    install_tensorrt_rtx ${install_wheel_or_not}
fi