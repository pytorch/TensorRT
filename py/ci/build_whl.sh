#!/bin/bash

# Example usage: docker run -it -v$(pwd):/workspace/TensorRT build_torch_tensorrt_wheel /bin/bash /workspace/TensorRT/py/ci/build_whl.sh

export CXX=g++
export CUDA_HOME=/usr/local/cuda-12.8
export PROJECT_DIR=/workspace/TensorRT

rm -rf /usr/local/cuda

if [[ $CUDA_HOME == "/usr/local/cuda-12.8" ]]; then
    cp -r /usr/local/cuda-11.8 /usr/local/cuda
    cp -r /usr/local/cuda-12.0/ /usr/local/cuda/
    rsync -a /usr/local/cuda-12.8/ /usr/local/cuda/
    export CUDA_HOME=/usr/local/cuda
else
    ln -s $CUDA_HOME /usr/local/cuda
fi

build_wheel() {
    $1/bin/python -m pip install --upgrade pip setuptools
    $1/bin/python -m pip install ${TENSORRT_DIR}/python/tensorrt-${TENSORRT_VERSION}-${2}-none-linux_x86_64.whl

    $1/bin/python -m pip install -r py/requirements.txt
    #$1/bin/python -m pip wheel . -w dist
    export BUILD_VERSION=$(cd ${PROJECT_DIR} && $1/bin/python3 -c "import versions; versions.torch_tensorrt_version_release()")
    CI_BUILD=1 RELEASE=1 $1/bin/python setup.py bdist_wheel
}

patch_wheel() {
    $2/bin/python -m pip install auditwheel
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TENSERRT_DIR}/lib:$1/torch/lib:$1/tensorrt/:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs $2/bin/python -m auditwheel repair  $(cat ${PROJECT_DIR}/py/ci/soname_excludes.params) --plat manylinux_2_34_x86_64 dist/torch_tensorrt-*-$3-linux_x86_64.whl
}

py39() {
    cd ${PROJECT_DIR}
    PY_BUILD_CODE=cp39-cp39
    PY_VERSION=3.9
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

py310() {
    cd ${PROJECT_DIR}
    PY_BUILD_CODE=cp310-cp310
    PY_VERSION=3.10
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

py311() {
    cd ${PROJECT_DIR}
    PY_BUILD_CODE=cp311-cp311
    PY_VERSION=3.11
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

py312() {
    cd ${PROJECT_DIR}
    PY_BUILD_CODE=cp312-cp312
    PY_VERSION=3.12
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

libtorchtrt() {
    cd ${PROJECT_DIR}
    mkdir -p ${PROJECT_DIR}/py/wheelhouse
    PY_BUILD_CODE=cp310-cp310
    PY_VERSION=3.10
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    ${PY_DIR}/bin/python -m pip install --upgrade pip
    ${PY_DIR}/bin/python -m pip install -r py/requirements.txt
    ${PY_DIR}/bin/python -m pip install setuptools wheel auditwheel
    bazel build //:libtorchtrt --platforms //toolchains:ci_rhel_x86_64_linux -c opt --noshow_progress
    CUDA_VERSION=$(cd ${PROJECT_DIR} && ${PY_DIR}/bin/python3 -c "import versions; versions.cuda_version()")
    TORCHTRT_VERSION=$(cd ${PROJECT_DIR} && ${PY_DIR}/bin/python3 -c "import versions; versions.torch_tensorrt_version_release()")
    TRT_VERSION=$(cd ${PROJECT_DIR} && ${PY_DIR}/bin/python3 -c "import versions; versions.tensorrt_version()")
    TORCH_VERSION=$(${PY_DIR}/bin/python -c "from torch import __version__;print(__version__.split('+')[0])")
    cp ${PROJECT_DIR}/bazel-bin/libtorchtrt.tar.gz ${PROJECT_DIR}/py/wheelhouse/libtorchtrt-${TORCHTRT_VERSION}-tensorrt${TRT_VERSION}-cuda${CUDA_VERSION}-libtorch${TORCH_VERSION}-x86_64-linux.tar.gz
}

libtorchtrt_pre_cxx11_abi() {
    cd ${PROJECT_DIR}/py
    mkdir -p ${PROJECT_DIR}/py/wheelhouse
    PY_BUILD_CODE=cp310-cp310
    PY_VERSION=3.10
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    ${PY_DIR}/bin/python -m pip install --upgrade pip
    ${PY_DIR}/bin/python -m pip install -r ${PROJECT_DIR}/py/requirements.txt
    ${PY_DIR}/bin/python -m pip install setuptools wheel auditwheel
    bazel build //:libtorchtrt --config pre_cxx11_abi --platforms //toolchains:ci_rhel_x86_64_linux -c opt --noshow_progress
    CUDA_VERSION=$(cd ${PROJECT_DIR} && ${PY_DIR}/bin/python3 -c "import versions; versions.cuda_version()")
    TORCHTRT_VERSION=$(cd ${PROJECT_DIR} && ${PY_DIR}/bin/python3 -c "import versions; versions.torch_tensorrt_version_release()")
    TRT_VERSION=$(cd ${PROJECT_DIR} && ${PY_DIR}/bin/python3 -c "import versions; versions.tensorrt_version()")
    TORCH_VERSION=$(${PY_DIR}/bin/python -c "from torch import __version__;print(__version__.split('+')[0])")
    cp ${PROJECT_DIR}/bazel-bin/libtorchtrt.tar.gz ${PROJECT_DIR}/py/wheelhouse/libtorchtrt-${TORCHTRT_VERSION}-pre-cxx11-abi-tensorrt${TRT_VERSION}-cuda${CUDA_VERSION}-libtorch${TORCH_VERSION}-x86_64-linux.tar.gz
}
