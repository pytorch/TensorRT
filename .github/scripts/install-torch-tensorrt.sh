#set -exou pipefail
set -x

TORCH=$(grep "^torch>" ${PWD}/py/requirements.txt)
TORCHVISION=$(grep "^torchvision" ${PWD}/py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}
PLATFORM=$(python -c "import sys; print(sys.platform)")

if [[ $(uname -m) == "aarch64" ]]; then
    # install cuda for aarch64
    source .github/scripts/install-cuda-aarch64.sh
    install_cuda_aarch64
fi

# Install all the dependencies required for Torch-TensorRT
pip install --pre -r ${PWD}/tests/py/requirements.txt
pip uninstall -y torch torchvision
pip install --force-reinstall --pre ${TORCH} --index-url ${INDEX_URL}
pip install --force-reinstall --pre ${TORCHVISION} --index-url ${INDEX_URL}


# Install Torch-TensorRT
if [[ ${PLATFORM} == win32 ]]; then
    pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl
else
    pip install /opt/torch-tensorrt-builds/torch_tensorrt*.whl
fi

echo -e "Running test script";
