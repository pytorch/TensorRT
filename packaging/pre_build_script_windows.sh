set -exou pipefail

pip install -U numpy packaging pyyaml setuptools wheel

choco install bazelisk -y

if [[ "${CU_VERSION::4}" < "cu12" ]]; then
  # replace dependencies from tensorrt-cu12-bindings/libs to tensorrt-cu11-bindings/libs
  sed -i -e "s/tensorrt-cu12>=/tensorrt-${CU_VERSION::4}>=/g" \
         -e "s/tensorrt-cu12-bindings>=/tensorrt-${CU_VERSION::4}-bindings>=/g" \
         -e "s/tensorrt-cu12-libs>=/tensorrt-${CU_VERSION::4}-libs>=/g" \
         pyproject.toml
fi

TORCH_TORCHVISION=$(grep "^torch" py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}

# Install all the dependencies required for Torch-TensorRT
pip uninstall -y torch torchvision
pip install --force-reinstall --pre ${TORCH_TORCHVISION} --index-url ${INDEX_URL}

export CUDA_HOME="$(echo ${CUDA_PATH} | sed -e 's#\\#\/#g')"
export TORCH_INSTALL_PATH="$(python -c "import torch, os; print(os.path.dirname(torch.__file__))" | sed -e 's#\\#\/#g')"

cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel

cat MODULE.bazel
echo "RELEASE=1" >> ${GITHUB_ENV}
