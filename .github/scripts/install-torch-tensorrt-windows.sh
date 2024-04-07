source "${BUILD_ENV_FILE}"
${CONDA_RUN} ${PIP_INSTALL_TORCH} torchvision
${CONDA_RUN} pip install pyyaml mpmath==1.3.0
${CONDA_RUN} pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl tensorrt==9.3.0.post12.dev1 --extra-index-url https://pypi.nvidia.com

echo -e "Running test script";
