#!/usr/bin/env bash
set -eou pipefail
# Source conda so it's available to the script environment
source ${BUILD_ENV_FILE}
${CONDA_RUN} ${PIP_INSTALL_TORCH} torch==2.1.2 torchvision==0.16.2 pyyaml
export TRT_VERSION=$(${CONDA_RUN} python -c "import versions; versions.tensorrt_version()")
${CONDA_RUN} python -m pip install /opt/torch-tensorrt-builds/torch_tensorrt*+${CU_VERSION}*.whl tensorrt~=${TRT_VERSION} tensorrt-bindings~=${TRT_VERSION} --extra-index-url=https://pypi.ngc.nvidia.com

echo -e "Running test script";
