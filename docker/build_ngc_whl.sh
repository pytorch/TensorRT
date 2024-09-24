#!/bin/bash
build_wheel() {
    cd /opt/pytorch/torch_tensorrt && \
    cp ./docker/MODULE.bazel.ngc  MODULE.bazel && \
    MAX_JOBS=1 LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 && \
    BUILD_VERSION=$(cat version.txt) python setup.py bdist_wheel --use-cxx11-abi --release && \
    pip install --no-cache-dir dist/*.whl
}

patch_wheel() {
    $2/bin/python -m pip install auditwheel
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TENSERRT_DIR}/lib:$1/torch/lib:$1/tensorrt/:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs $2/bin/python -m auditwheel repair  $(cat ${PROJECT_DIR}/py/ci/soname_excludes.params) --plat manylinux_2_34_x86_64 dist/torch_tensorrt-*-$3-linux_x86_64.whl
}

py38() {
    cd ${TORCHTRT_DIR}
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
    cd ${TORCHTRT_DIR}
    PY_BUILD_CODE=cp39-cp39
    PY_VERSION=3.9
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

py310() {
    cd ${TORCHTRT_DIR}
    PY_BUILD_CODE=cp310-cp310
    PY_VERSION=3.10
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

py311() {
    cd ${TORCHTRT_DIR}
    PY_BUILD_CODE=cp311-cp311
    PY_VERSION=3.11
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

py312() {
    cd ${TORCHTRT_DIR}
    PY_BUILD_CODE=cp312-cp312
    PY_VERSION=3.12
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

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
