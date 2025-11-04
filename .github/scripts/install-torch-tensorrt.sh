#set -exou pipefail
set -x

TORCH=$(grep "^torch>" ${PWD}/py/requirements.txt)
TORCHVISION=$(grep "^torchvision>" ${PWD}/tests/py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}
PLATFORM=$(python -c "import sys; print(sys.platform)")

if [[ $(uname -m) == "aarch64" ]]; then
    # install cuda for aarch64
    source .github/scripts/install-cuda-aarch64.sh
    install_cuda_aarch64
fi

# Install all the dependencies required for Torch-TensorRT
pip install --pre -r ${PWD}/tests/py/requirements.txt
# dependencies in the tests/py/requirements.txt might install a different version of torch or torchvision
# eg. timm will install the latest torchvision, however we want to use the torchvision from nightly
# reinstall torch torchvisionto make sure we have the correct version
pip uninstall -y torch torchvision
pip install --force-reinstall --pre ${TORCHVISION} --index-url ${INDEX_URL}
pip install --force-reinstall --pre ${TORCH} --index-url ${INDEX_URL}

if [[ ${USE_TRT_RTX} == true ]]; then
    source .github/scripts/install-tensorrt-rtx.sh
    # tensorrt-rtx is not publicly available, so we need to install the wheel from the tar ball
    install_wheel_or_not=true
    install_tensorrt_rtx ${install_wheel_or_not}
fi

# Install Torch-TensorRT
if [[ ${PLATFORM} == win32 ]]; then
    pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl
else
    pip install /opt/torch-tensorrt-builds/torch_tensorrt*.whl
fi

echo -e "Running test script";
