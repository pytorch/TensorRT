#!/usr/bin/env bash
set -eou pipefail
# Source conda so it's available to the script environment
source ${BUILD_ENV_FILE}
${CONDA_RUN} ${PIP_INSTALL_TORCH} torchvision --extra-index-url https://pypi.python.org/simple
${CONDA_RUN} python -m pip install pyyaml mpmath==1.3.0
export TRT_VERSION=$(${CONDA_RUN} python -c "import versions; versions.tensorrt_version()")
${CONDA_RUN} python -m pip install /opt/torch-tensorrt-builds/torch_tensorrt*+${CU_VERSION}*.whl tensorrt~=${TRT_VERSION} tensorrt-bindings~=${TRT_VERSION} --extra-index-url=https://pypi.ngc.nvidia.com

echo -e "Running test script";