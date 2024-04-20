source "${BUILD_ENV_FILE}"
# ${CONDA_RUN} ${PIP_INSTALL_TORCH} torchvision --extra-index-url https://download.pytorch.org/whl/test/cu121 --extra-index-url https://download.pytorch.org/whl/test/cu118
${CONDA_RUN} pip install pyyaml mpmath==1.3.0
${CONDA_RUN} pip install tensorrt==10.0.0b6 --extra-index-url https://pypi.nvidia.com
${CONDA_RUN} pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl torch torchvision --extra-index-url https://download.pytorch.org/whl/test/cu121 --extra-index-url https://download.pytorch.org/whl/test/cu118

echo -e "Running test script";
