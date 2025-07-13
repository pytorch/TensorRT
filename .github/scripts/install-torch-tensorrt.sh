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

# tensorrt-rtx is not publicly available, so we need to install it from the local path
if [[ ${USE_RTX} == true ]]; then
    echo "It is the tensorrt-rtx build, install tensorrt-rtx"
    # python version is like 3.11, we need to convert it to cp311
    CPYTHON_TAG="cp${PYTHON_VERSION//./}"
    if [[ ${PLATFORM} == win32 ]]; then
        curl -L http://cuda-repo/release-candidates/Libraries/TensorRT/v10.12/10.12.0.35-51f47a12/12.9-r575/Windows10-x64-winjit/zip/TensorRT-RTX-1.0.0.21.Windows.win10.cuda-12.9.zip -o TensorRT-RTX-1.0.0.21.Windows.win10.cuda-12.9.zip
        unzip TensorRT-RTX-1.0.0.21.Windows.win10.cuda-12.9.zip
        pip install TensorRT-RTX-1.0.0.21/python/tensorrt_rtx-1.0.0.21-${CPYTHON_TAG}-none-win_amd64.whl
    else
        curl -L http://cuda-repo/release-candidates/Libraries/TensorRT/v10.12/10.12.0.35-51f47a12/12.9-r575/Linux-x64-manylinux_2_28-winjit/tar/TensorRT-RTX-1.0.0.21.Linux.x86_64-gnu.cuda-12.9.tar.gz -o TensorRT-RTX-1.0.0.21.Linux.x86_64-gnu.cuda-12.9.tar.gz
        tar -xzf TensorRT-RTX-1.0.0.21.Linux.x86_64-gnu.cuda-12.9.tar.gz
        pip install TensorRT-RTX-1.0.0.21/python/tensorrt_rtx-1.0.0.21-${CPYTHON_TAG}-none-linux_x86_64.whl
    fi
else
    echo "It is the standard tensorrt build"
fi


# Install Torch-TensorRT
if [[ ${PLATFORM} == win32 ]]; then
    pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl
else
    pip install /opt/torch-tensorrt-builds/torch_tensorrt*.whl
fi

echo -e "Running test script";
