set -x

pip install -U numpy packaging pyyaml setuptools wheel

choco install bazelisk -y

echo TENSORRT_VERSION=${TENSORRT_VERSION}

if [[ ${TENSORRT_VERSION} != "" ]]; then
  # this is the upgraded TensorRT version, replace current tensorrt version to the upgrade tensorRT version in the pyproject.toml
  # example: __tensorrt_version__: ">=10.3.0,<=10.6.0"
  # replace: tensorrt-cu12>=10.3.0,<=10.6.0 to tensorrt-cu12==10.8.0
  current_version=$(cat dev_dep_versions.yml | grep __tensorrt_version__ | sed 's/__tensorrt_version__: //g' | sed 's/"//g')
  sed -i -e "s/tensorrt-cu12${current_version}/tensorrt-cu12==${TENSORRT_VERSION}/g" \
         -e "s/tensorrt-cu12-bindings${current_version}/tensorrt-cu12-bindings==${TENSORRT_VERSION}/g" \
         -e "s/tensorrt-cu12-libs${current_version}/tensorrt-cu12-libs==${TENSORRT_VERSION}/g" \
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


if [[ ${TENSORRT_VERSION} != "" ]]; then
  cat toolchains/ci_workspaces/MODULE_tensorrt.bazel.tmpl | envsubst > MODULE.bazel
else
  cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel
fi

cat MODULE.bazel
echo "RELEASE=1" >> ${GITHUB_ENV}
