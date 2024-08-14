set -euxo pipefail

TORCH_TORCHVISION=$(grep "^torch" ${PWD}/py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}
PLATFORM=$(python -c "import sys; print(sys.platform)")

echo "PLATFORM=${PLATFORM}"
# Install all the dependencies required for Torch-TensorRT
pip install --pre ${TORCH_TORCHVISION} --index-url ${INDEX_URL}
pip install --pre -r ${PWD}/tests/py/requirements.txt --use-deprecated legacy-resolver

export CUDA_HOME="$(echo ${CUDA_PATH} | sed -e 's#\\#\/#g')"
export TORCH_INSTALL_PATH="$(python -c "import torch, os; print(os.path.dirname(torch.__file__))" | sed -e 's#\\#\/#g')"

cat ${PWD}/toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > ${PWD}/MODULE.bazel

cat ${PWD}/MODULE.bazel

# Install Torch-TensorRT
if [[ ${PLATFORM} == win32 ]]; then
    ls -lart ${RUNNER_ARTIFACT_DIR}
    pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl
else
    pip install /opt/torch-tensorrt-builds/torch_tensorrt*.whl
fi

echo -e "Running test script";
