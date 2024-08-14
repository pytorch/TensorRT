set -exou pipefail

pip install -U numpy packaging pyyaml setuptools wheel

# Install TRT from PyPI
TRT_VERSION=$(python -c "import yaml; print(yaml.safe_load(open('dev_dep_versions.yml', 'r'))['__tensorrt_version__'])")
pip install tensorrt==${TRT_VERSION} tensorrt-${CU_VERSION::4}-bindings==${TRT_VERSION} tensorrt-${CU_VERSION::4}-libs==${TRT_VERSION} --extra-index-url https://pypi.nvidia.com

choco install bazelisk -y

#curl -Lo TensorRT.zip https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.0.1/zip/TensorRT-10.0.1.6.Windows10.win10.cuda-12.4.zip
#unzip -o TensorRT.zip -d C:/

export CUDA_HOME="$(echo ${CUDA_PATH} | sed -e 's#\\#\/#g')"
export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")

cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel

cat MODULE.bazel

echo "RELEASE=1" >> ${GITHUB_ENV}
