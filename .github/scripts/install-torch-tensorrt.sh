#!/usr/bin/env bash
set -eou pipefail
# Source conda so it's available to the script environment
source ${BUILD_ENV_FILE}
export EXTRA_INDEX_URL="https://download.pytorch.org/whl/nightly/${CU_VERSION}"
export PLATFORM=$(${CONDA_RUN} python -c "import sys; print(sys.platform)")

# Install all the dependencies required for Torch-TensorRT
${CONDA_RUN} pip install --pre -r ${PWD}/tests/py/requirements.txt --use-deprecated=legacy-resolver --extra-index-url=${EXTRA_INDEX_URL}

# Install Torch-TensorRT
if [[ "$PLATFORM" == "win32" ]]; then
    ${CONDA_RUN} pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl
else
    ${CONDA_RUN} pip install /opt/torch-tensorrt-builds/torch_tensorrt*.whl
fi

echo -e "Running test script";