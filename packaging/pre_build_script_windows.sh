set -exou pipefail

pip install -U numpy packaging pyyaml setuptools wheel

# Install TRT from PyPI
TRT_VERSION=$(python -c "import yaml; print(yaml.safe_load(open('dev_dep_versions.yml', 'r'))['__tensorrt_version__'])")
pip install tensorrt==${TRT_VERSION} tensorrt-${CU_VERSION::4}==${TRT_VERSION} tensorrt-${CU_VERSION::4}-bindings==${TRT_VERSION} tensorrt-${CU_VERSION::4}-libs==${TRT_VERSION} --extra-index-url https://pypi.nvidia.com

choco install bazelisk -y

if [[ "${CU_VERSION::4}" < "cu12" ]]; then
  # replace dependencies from tensorrt-cu12-bindings/libs to tensorrt-cu11-bindings/libs
  sed -i -e "s/tensorrt-cu12==/tensorrt-${CU_VERSION::4}==/g" \
         -e "s/tensorrt-cu12-bindings==/tensorrt-${CU_VERSION::4}-bindings==/g" \
         -e "s/tensorrt-cu12-libs==/tensorrt-${CU_VERSION::4}-libs==/g" \
         pyproject.toml
  cat pyproject.toml
fi

#curl -Lo TensorRT.zip https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.0.1/zip/TensorRT-10.0.1.6.Windows10.win10.cuda-12.4.zip
#unzip -o TensorRT.zip -d C:/
TORCH_TORCHVISION=$(grep "^torch" py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}

# Install all the dependencies required for Torch-TensorRT
pip uninstall -y torch torchvision
pip install --force-reinstall --pre ${TORCH_TORCHVISION} --index-url ${INDEX_URL}
pip install --pre -r tests/py/requirements.txt --use-deprecated legacy-resolver

export CUDA_HOME="$(echo ${CUDA_PATH} | sed -e 's#\\#\/#g')"
export TORCH_INSTALL_PATH="$(python -c "import torch, os; print(os.path.dirname(torch.__file__))" | sed -e 's#\\#\/#g')"

cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel

cat MODULE.bazel
echo "RELEASE=1" >> ${GITHUB_ENV}
