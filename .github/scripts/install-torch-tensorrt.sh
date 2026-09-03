#set -exou pipefail
set -x

TORCH=$(grep "^torch>" ${PWD}/py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}
PLATFORM=$(python -c "import sys; print(sys.platform)")

if [[ $(uname -m) == "aarch64" ]]; then
    # install cuda for aarch64
    source .github/scripts/install-cuda-aarch64.sh
    install_cuda_aarch64
fi

# Install all the dependencies required for Torch-TensorRT
python -m pip install --upgrade "pip>=25.1" "tomli>=1.1.0; python_version < '3.11'"
python -m pip install \
    --pre \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION} \
    --group test \
    --group test-ext \
    --group quantization
TORCHVISION=$(python - <<'PY'
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

with open("pyproject.toml", "rb") as f:
    deps = tomllib.load(f)["dependency-groups"]["test-ext"]

for dep in deps:
    if dep.startswith("torchvision"):
        print(dep)
        break
else:
    raise SystemExit("torchvision was not found in dependency group test-ext")
PY
)
# test dependencies might install a different version of torch or torchvision
# eg. timm will install the latest torchvision, however we want to use the torchvision from nightly
# reinstall torch torchvision to make sure we have the correct version
python -m pip uninstall -y torch torchvision
python -m pip install --force-reinstall --pre ${TORCHVISION} --index-url ${INDEX_URL} --extra-index-url https://pypi.org/simple
python -m pip install --force-reinstall --pre ${TORCH} --index-url ${INDEX_URL} --extra-index-url https://pypi.org/simple
# dynamo-torchao full/nightly suite
python -m pip install torchao

# Prepend the venv's NVIDIA CUDA runtime libs to LD_LIBRARY_PATH. The two majors ship
# different layouts: nvidia-cuda-runtime-cu12 installs nvidia/cuda_runtime/lib/libcudart.so.12
# while the CUDA 13 line installs nvidia/cu13/lib/libcudart.so.13. Naming only the cu13 path
# left every CUDA 12 row without a CUDA runtime on the search path, so a binary linked against
# libcudart died at startup with "libcudart.so.12: cannot open shared object file" even though
# the package was installed.
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
case "${CU_VERSION}" in
cu13*) CUDA_RUNTIME_LIB_DIR="${SITE_PACKAGES}/nvidia/cu13/lib" ;;
cu12*) CUDA_RUNTIME_LIB_DIR="${SITE_PACKAGES}/nvidia/cuda_runtime/lib" ;;
*) CUDA_RUNTIME_LIB_DIR="" ;;
esac
if [[ -n "${CUDA_RUNTIME_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Install Torch-TensorRT
if [[ ${PLATFORM} == win32 ]]; then
    python -m pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl
else
    python -m pip install /opt/torch-tensorrt-builds/torch_tensorrt*.whl --use-deprecated=legacy-resolver
fi

echo -e "Running test script";
