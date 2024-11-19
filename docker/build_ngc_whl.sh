#!/bin/bash
build_wheel() {
    cd /opt/pytorch/torch_tensorrt && \
    cp ./docker/MODULE.bazel.ngc  MODULE.bazel && \
    cp ./docker/pyproject.toml.ngc  pyproject.toml && \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs && \
    MAX_JOBS=1 LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 && \    
    BUILD_VERSION=$(cat version.txt) python setup.py bdist_wheel --use-cxx11-abi --release --ci
}
patch_wheel() {
    python -m auditwheel repair  $(cat py/ci/soname_excludes.params) --only-plat dist/torch_tensorrt-*-$3-linux_x86_64.whl
    pip install --no-cache-dir wheelhouse/*.whl
}
py38() {
    PY_BUILD_CODE=cp38-cp38
    PY_SINGLE_BUILD_CODE=cp38
    PY_VERSION=3.8
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR} ${PY_SINGLE_BUILD_CODE}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}
py39() {
    PY_BUILD_CODE=cp39-cp39
    PY_VERSION=3.9
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}
py310() {
    PY_BUILD_CODE=cp310-cp310
    PY_VERSION=3.10
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}
py311() {
    PY_BUILD_CODE=cp311-cp311
    PY_VERSION=3.11
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}
py312() {
    PY_BUILD_CODE=cp312-cp312
    PY_VERSION=3.12
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}
py_version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$py_version" = "3.8" ]; then
    py38
elif [ "$py_version" = "3.9" ]; then
    py39
elif [ "$py_version" = "3.10" ]; then
    py310
elif [ "$py_version" = "3.11" ]; then
    py311
else
    py312
fi