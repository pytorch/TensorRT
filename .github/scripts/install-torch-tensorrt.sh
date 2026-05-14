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
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
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

# If CUDA 13 (cu13), prepend venv's NVIDIA CUDA 13 libs to LD_LIBRARY_PATH
if [[ "${CU_VERSION}" == cu13* ]]; then
    SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
    export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cu13/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Install Torch-TensorRT
if [[ ${PLATFORM} == win32 ]]; then
    python -m pip install ${RUNNER_ARTIFACT_DIR}/torch_tensorrt*.whl
else
    python -m pip install /opt/torch-tensorrt-builds/torch_tensorrt*.whl --use-deprecated=legacy-resolver
fi

echo -e "Running test script";
