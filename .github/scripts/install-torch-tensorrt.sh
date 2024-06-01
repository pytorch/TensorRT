set -eou pipefail

INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}
PLATFORM=$(python -c "import sys; print(sys.platform)")

# Install all the dependencies required for Torch-TensorRT
pip install --pre "torch>=2.4.0.dev,<2.5.0" "torchvision>=0.19.0.dev,<0.20.0" --index-url ${INDEX_URL}
pip install --pre -r ${PWD}/tests/py/requirements.txt --use-deprecated legacy-resolver

# Install Torch-TensorRT
if [[ ${PLATFORM} == win32 ]]; then
    pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl
else
    pip install /opt/torch-tensorrt-builds/torch_tensorrt*.whl
fi

echo -e "Running test script";
