set -eou pipefail

python -m pip install -U numpy packaging pyyaml setuptools wheel

# Install TRT from PyPI
TRT_VERSION=$(python -c "import yaml; print(yaml.safe_load(open('dev_dep_versions.yml', 'r'))['__tensorrt_version__'])")
python -m pip install tensorrt==${TRT_VERSION} tensorrt-${CU_VERSION::4}==${TRT_VERSION} tensorrt-${CU_VERSION::4}-bindings==${TRT_VERSION} tensorrt-${CU_VERSION::4}-libs==${TRT_VERSION} --extra-index-url https://pypi.nvidia.com

choco install bazelisk -y

if [ ${CU_VERSION} = cu118 ]; then
    TRT_DOWNLOAD_LINK=https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.0.1/zip/TensorRT-10.0.1.6.Windows10.win10.cuda-11.8.zip
elif [ ${CU_VERSION} = cu121 ] || [ ${CU_VERSION} = cu124 ]; then
    TRT_DOWNLOAD_LINK=https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.0.1/zip/TensorRT-10.0.1.6.Windows10.win10.cuda-12.4.zip
else
    echo "Unsupported CU_VERSION"
    exit 1
fi

curl -Lo TensorRT.zip ${TRT_DOWNLOAD_LINK}
unzip -o TensorRT.zip -d C:/

export CUDA_HOME="$(echo ${CUDA_PATH} | sed -e 's#\\#\/#g')"

cat toolchains/ci_workspaces/WORKSPACE.win.release.tmpl | envsubst > WORKSPACE

echo "RELEASE=1" >> ${GITHUB_ENV}
