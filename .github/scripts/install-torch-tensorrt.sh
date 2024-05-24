#!/usr/bin/env bash
set -eou pipefail
# Source conda so it's available to the script environment
source ${BUILD_ENV_FILE}
${CONDA_RUN} ${PIP_INSTALL_TORCH} torchvision==0.18.0
${CONDA_RUN} python -m pip install pyyaml mpmath==1.3.0
export TRT_VERSION=$(${CONDA_RUN} python -c "import versions; versions.tensorrt_version()")

# Install Torch-TensorRT
${CONDA_RUN} python -m pip install /opt/torch-tensorrt-builds/torch_tensorrt*+${CU_VERSION}*.whl tensorrt~=${TRT_VERSION} --extra-index-url=https://pypi.ngc.nvidia.com

echo -e "Running test script";
