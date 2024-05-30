set -eou pipefail

EXTRA_INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}
PLATFORM=$(python -c "import sys; print(sys.platform)")

# Install all the dependencies required for Torch-TensorRT
pip install --pre -r ${PWD}/tests/py/requirements.txt --use-deprecated=legacy-resolver --extra-index-url=${EXTRA_INDEX_URL}

# Install Torch-TensorRT
if [[ ${PLATFORM} == win32 ]]; then
    pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl
else
    pip install /opt/torch-tensorrt-builds/torch_tensorrt*.whl
fi

echo -e "Running test script";
