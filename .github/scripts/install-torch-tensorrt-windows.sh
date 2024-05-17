set -eou pipefail
source "${BUILD_ENV_FILE}"

# Install test index version of Torch and Torchvision
${CONDA_RUN} ${PIP_INSTALL_TORCH} torchvision
${CONDA_RUN} pip install pyyaml mpmath==1.3.0

# Install TRT from PyPi
TRT_VERSION=$(${CONDA_RUN} python -c "import yaml; print(yaml.safe_load(open('dev_dep_versions.yml', 'r'))['__tensorrt_version__'])")
${CONDA_RUN} pip install tensorrt==${TRT_VERSION} tensorrt-${CU_VERSION::4}==${TRT_VERSION} tensorrt-${CU_VERSION::4}-bindings==${TRT_VERSION} tensorrt-${CU_VERSION::4}-libs==${TRT_VERSION} --extra-index-url https://pypi.nvidia.com

# Install pre-built Torch-TRT
${CONDA_RUN} pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl

echo -e "Running test script";
