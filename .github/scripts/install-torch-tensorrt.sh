#!/usr/bin/env bash
set -eou pipefail
# Source conda so it's available to the script environment
source ${BUILD_ENV_FILE}

# Install stable versions of Torch and Torchvision since this is a release branch.
# For main branches, the INDEX_URL will always point to nightly and for RC releases,
# it will contain test tag in the url.
export TORCH_VERSION_CI="2.3.0"
export TORCHVISION_VERSION_CI="0.18.0"
export TENSORRT_VERSION_CI="10.0.1"
export INDEX_URL="https://download.pytorch.org/whl/${CU_VERSION}"


${CONDA_RUN} python -m pip install pyyaml mpmath==1.3.0

# Install appropriate torch and torchvision versions for the platform
${CONDA_RUN} pip install torch==${TORCH_VERSION_CI} torchvision==${TORCHVISION_VERSION_CI} --index-url ${INDEX_URL}

# Install TRT 10 from PyPi
${CONDA_RUN} pip install tensorrt==${TENSORRT_VERSION_CI}

# Install Torch-TensorRT
${CONDA_RUN} python -m pip install /opt/torch-tensorrt-builds/torch_tensorrt*+${CU_VERSION}*.whl

echo -e "Running test script";
