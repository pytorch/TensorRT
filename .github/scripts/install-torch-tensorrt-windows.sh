#!/usr/bin/env bash
set -eou pipefail
# Source conda so it's available to the script environment
source ${BUILD_ENV_FILE}

# Install PyTorch and torchvision from index
${CONDA_RUN} ${PIP_INSTALL_TORCH} torchvision

# Install all the dependencies required for Torch-TensorRT
${CONDA_RUN} pip install --pre -r ${PWD}/tests/py/requirements.txt --use-deprecated=legacy-resolver

# Install Torch-TensorRT via pre-built wheels. On windows, the location of wheels is not fixed.
${CONDA_RUN} pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl

echo -e "Running test script";