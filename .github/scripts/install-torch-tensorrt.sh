# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -ex

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

# Prepend the venv's NVIDIA CUDA runtime libs to LD_LIBRARY_PATH.
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
case "${CU_VERSION}" in
cu13*) CUDA_RUNTIME_LIB_DIR="${SITE_PACKAGES}/nvidia/cu13/lib" ;;
*) CUDA_RUNTIME_LIB_DIR="" ;;
esac
if [[ -n "${CUDA_RUNTIME_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Install Torch-TensorRT
if [[ ${PLATFORM} == win32 ]]; then
    # Same exclusion as the Linux branch below, and for the same reason: the wheel's name varies by
    # variant, so anchoring on a prefix leaves the pattern unexpanded and pip reads it literally.
    wheels=""
    for wheel in "${RUNNER_ARTIFACT_DIR}"/torch_tensorrt*.whl; do
        case "${wheel}" in
            *executorch_runtime*) continue ;;
        esac
        wheels="${wheels} ${wheel}"
    done
    # pin-check: no-nightly -- Windows installs only the main wheel, without the Linux companion.
    python -m pip install ${wheels} || exit 1
else
    # Every built wheel except the companion. Installing the companion here is what forced a
    # nightly index onto release jobs; the ExecuTorch workflow installs it instead, naming the
    # channel it wants. Selecting by exclusion rather than by prefix, because the main wheel's name
    # varies by variant and a prefix guess leaves the glob unexpanded and pip reading it literally.
    wheels=""
    for wheel in /opt/torch-tensorrt-builds/torch_tensorrt*.whl; do
        case "${wheel}" in
            *executorch_runtime*) continue ;;
        esac
        wheels="${wheels} ${wheel}"
    done
    # Exit explicitly: the caller appends its test script and this file does not use set -e.
    # pin-check: no-nightly -- the main wheel alone, which needs no ExecuTorch channel.
    python -m pip install ${wheels} --use-deprecated=legacy-resolver || exit 1
fi

echo -e "Running test script";
