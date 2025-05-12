#set -exou pipefail
set -x
TORCH_TORCHVISION=$(grep "^torch" ${PWD}/py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}
PLATFORM=$(python -c "import sys; print(sys.platform)")

if [[ $(uname -m) == "aarch64" ]]; then
    # install cuda 12.8 only for aarch64 for now
    VER="12-8"
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
    dnf -y install cuda-compiler-${VER}.aarch64 \
                cuda-libraries-${VER}.aarch64 \
                cuda-libraries-devel-${VER}.aarch64
    dnf clean all
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ls -lart /usr/local/
    nvcc --version
    echo "cuda installed successfully"
fi

# Install all the dependencies required for Torch-TensorRT
pip install --pre ${TORCH_TORCHVISION} --index-url ${INDEX_URL}
pip install --pre -r ${PWD}/tests/py/requirements.txt

# Install Torch-TensorRT
if [[ ${PLATFORM} == win32 ]]; then
    pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl
else
    pip install /opt/torch-tensorrt-builds/torch_tensorrt*.whl
fi

echo -e "Running test script";
