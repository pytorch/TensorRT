#!/bin/bash

# Example usage: docker run -it -v$(pwd)/..:/workspace/TRTorch build_trtorch_wheel /bin/bash /workspace/TRTorch/py/build_whl.sh

export CXX=g++
export CUDA_HOME=/usr/local/cuda-11.7
export PROJECT_DIR=/workspace/project

cp -r $CUDA_HOME /usr/local/cuda

py37() {
    cd /workspace/project/py
    PY_BUILD_CODE=cp37-cp37m
    PY_VERSION=3.7
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    ${PY_DIR}/bin/python -m pip install --upgrade pip
    ${PY_DIR}/bin/python -m pip install -r requirements.txt
    ${PY_DIR}/bin/python -m pip install setuptools wheel auditwheel
    ${PY_DIR}/bin/python setup.py bdist_wheel --release --ci
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PY_PKG_DIR}/torch/lib:${PY_PKG_DIR}/tensorrt/:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs ${PY_DIR}/bin/python -m auditwheel repair $(cat ${PROJECT_DIR}/py/ci/soname_excludes.params) --plat manylinux_2_17_x86_64 dist/torch_tensorrt-*-${PY_BUILD_CODE}-linux_x86_64.whl
}

py38() {
    cd /workspace/project/py
    PY_BUILD_CODE=cp38-cp38
    PY_VERSION=3.8
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    ${PY_DIR}/bin/python -m pip install --upgrade pip
    ${PY_DIR}/bin/python -m pip install -r requirements.txt
    ${PY_DIR}/bin/python -m pip install setuptools wheel auditwheel
    ${PY_DIR}/bin/python setup.py bdist_wheel --release --ci
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PY_PKG_DIR}/torch/lib:${PY_PKG_DIR}/tensorrt/:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs ${PY_DIR}/bin/python -m auditwheel repair  $(cat ${PROJECT_DIR}/py/ci/soname_excludes.params) --plat manylinux_2_17_x86_64 dist/torch_tensorrt-*-${PY_BUILD_CODE}-linux_x86_64.whl
}

py39() {
    cd /workspace/project/py
    PY_BUILD_CODE=cp39-cp39
    PY_VERSION=3.9
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    ${PY_DIR}/bin/python -m pip install --upgrade pip
    ${PY_DIR}/bin/python -m pip install -r requirements.txt
    ${PY_DIR}/bin/python -m pip install setuptools wheel auditwheel
    ${PY_DIR}/bin/python setup.py bdist_wheel --release --ci
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PY_PKG_DIR}/torch/lib:${PY_PKG_DIR}/tensorrt/:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs ${PY_DIR}/bin/python -m auditwheel repair  $(cat ${PROJECT_DIR}/py/ci/soname_excludes.params) --plat manylinux_2_17_x86_64 dist/torch_tensorrt-*-${PY_BUILD_CODE}-linux_x86_64.whl
}

py310() {
    cd /workspace/project/py
    PY_BUILD_CODE=cp310-cp310
    PY_VERSION=3.10
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    ${PY_DIR}/bin/python -m pip install --upgrade pip
    ${PY_DIR}/bin/python -m pip install -r requirements.txt
    ${PY_DIR}/bin/python -m pip install setuptools wheel auditwheel
    ${PY_DIR}/bin/python setup.py bdist_wheel --release --ci
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PY_PKG_DIR}/torch/lib:${PY_PKG_DIR}/tensorrt/:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs ${PY_DIR}/bin/python -m auditwheel repair  $(cat ${PROJECT_DIR}/py/ci/soname_excludes.params) --plat manylinux_2_17_x86_64 dist/torch_tensorrt-*-${PY_BUILD_CODE}-linux_x86_64.whl
}

#build_py311() {
#    /opt/python/cp311-cp311/bin/python -m pip install -r requirements.txt
#    /opt/python/cp311-cp311/bin/python setup.py bdist_wheel --release --ci
#    #auditwheel repair --plat manylinux_2_17_x86_64
#}

libtorchtrt() {
    cd /workspace/project/py
    mkdir -p /workspace/project/py/wheelhouse
    PY_BUILD_CODE=cp310-cp310
    PY_VERSION=3.10
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    ${PY_DIR}/bin/python -m pip install --upgrade pip
    ${PY_DIR}/bin/python -m pip install -r requirements.txt
    ${PY_DIR}/bin/python -m pip install setuptools wheel auditwheel
    bazel build //:libtorchtrt --platforms //toolchains:ci_rhel_x86_64_linux -c opt --noshow_progress
    CUDA_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "from versions import __cuda_version__;print(__cuda_version__)")
    TORCHTRT_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "from versions import __version__;print(__version__)")
    TRT_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "from versions import __tensorrt_version__;print(__tensorrt_version__)")
    CUDNN_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "from versions import __cudnn_version__;print(__cudnn_version__)")
    TORCH_VERSION=$(${PY_DIR}/bin/python -c "from torch import __version__;print(__version__.split('+')[0])")
    cp ${PROJECT_DIR}/bazel-bin/libtorchtrt.tar.gz ${PROJECT_DIR}/py/wheelhouse/libtorchtrt-${TORCHTRT_VERSION}-cudnn${CUDNN_VERSION}-tensorrt${TRT_VERSION}-cuda${CUDA_VERSION}-libtorch${TORCH_VERSION}-x86_64-linux.tar.gz
}

libtorchtrt_pre_cxx11_abi() {
    cd /workspace/project/py
    mkdir -p /workspace/project/py/wheelhouse
    PY_BUILD_CODE=cp310-cp310
    PY_VERSION=3.10
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    ${PY_DIR}/bin/python -m pip install --upgrade pip
    ${PY_DIR}/bin/python -m pip install -r requirements.txt
    ${PY_DIR}/bin/python -m pip install setuptools wheel auditwheel
    bazel build //:libtorchtrt --config pre_cxx11_abi --platforms //toolchains:ci_rhel_x86_64_linux -c opt --noshow_progress
    CUDA_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "from versions import __cuda_version__;print(__cuda_version__)")
    TORCHTRT_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "from versions import __version__;print(__version__)")
    TRT_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "from versions import __tensorrt_version__;print(__tensorrt_version__)")
    CUDNN_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "from versions import __cudnn_version__;print(__cudnn_version__)")
    TORCH_VERSION=$(${PY_DIR}/bin/python -c "from torch import __version__;print(__version__.split('+')[0])")
    cp ${PROJECT_DIR}/bazel-bin/libtorchtrt.tar.gz ${PROJECT_DIR}/py/wheelhouse/libtorchtrt-${TORCHTRT_VERSION}-pre-cxx11-abi-cudnn${CUDNN_VERSION}-tensorrt${TRT_VERSION}-cuda${CUDA_VERSION}-libtorch${TORCH_VERSION}-x86_64-linux.tar.gz
}