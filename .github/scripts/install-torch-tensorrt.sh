set -eou pipefail

TORCH_TORCHVISION=$(grep "^torch" ${PWD}/py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}
PLATFORM=$(python -c "import sys; print(sys.platform)")

echo -e "TORCH_TORCHVISION=${TORCH_TORCHVISION}"
echo -e "INDEX_URL=${INDEX_URL}"
echo -e "PLATFORM=${PLATFORM}"

# Install all the dependencies required for Torch-TensorRT
pip install --pre ${TORCH_TORCHVISION} --index-url ${INDEX_URL}
pip install --pre -r ${PWD}/tests/py/requirements.txt --use-deprecated legacy-resolver

# Install Torch-TensorRT
if [[ ${PLATFORM} == win32 ]]; then
    echo -e "RUNNER_ARTIFACT_DIR=${RUNNER_ARTIFACT_DIR}"
    pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl
else
    pip install /opt/torch-tensorrt-builds/torch_tensorrt*.whl
fi

echo -e "Running test script";
