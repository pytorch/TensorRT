set -eou pipefail
source "${BUILD_ENV_FILE}"

# Install test index version of Torch and Torchvision
${CONDA_RUN} pip install torch torchvision --index-url https://download.pytorch.org/whl/test/${CU_VERSION}
${CONDA_RUN} pip install pyyaml mpmath==1.3.0

# Install TRT 10 from PyPi
${CONDA_RUN} pip install tensorrt==10.0.0b6 tensorrt-${CU_VERSION::4}-bindings==10.0.0b6 tensorrt-${CU_VERSION::4}-libs==10.0.0b6 --extra-index-url https://pypi.nvidia.com

# Install pre-built Torch-TRT
${CONDA_RUN} pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl

echo -e "Running test script";
