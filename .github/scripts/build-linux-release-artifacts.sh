#!/bin/bash

set -x

source "${BUILD_ENV_FILE}"

export CXX=g++
CURRENT_DIR=`pwd`

if [[ "${PYTHON_VERSION}" == "3.8" ]]; then
  PY_SINGLE_BUILD_CODE=cp38
  PY_BUILD_CODE=cp38-cp38
elif [[ "${PYTHON_VERSION}" == "3.9" ]]; then
  PY_SINGLE_BUILD_CODE=cp39
  PY_BUILD_CODE=cp39-cp39
elif [[ "${PYTHON_VERSION}" == "3.10" ]]; then
  PY_SINGLE_BUILD_CODE=cp310
  PY_BUILD_CODE=cp310-cp310
elif [[ "${PYTHON_VERSION}" == "3.11" ]]; then
  PY_SINGLE_BUILD_CODE=cp311
  PY_BUILD_CODE=cp311-cp311
elif [[ "${PYTHON_VERSION}" == "3.12" ]]; then
  PY_SINGLE_BUILD_CODE=cp312
  PY_BUILD_CODE=cp312-cp312
else
   echo "python version: ${PYTHON_VERSION} is not supported"
   exit
fi

python -m pip install --upgrad pip setuptools
python -m pip install auditwheel pyyaml

# Setup Bazel via Bazelisk
wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 -O /usr/bin/bazel &&\
    chmod a+x /usr/bin/bazel

which bazel

# download TensorRT tarball
TRT_VERSION=10.0.1
wget -q https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz \
&& gunzip TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz \
&& tar -xvf TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar \
&& rm TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar

TENSERRT_DIR=${CURRENT_DIR}/TensorRT-10.0.1.6/

python -m pip install ${TENSERRT_DIR}/python/tensorrt-${TRT_VERSION}-${PY_SINGLE_BUILD_CODE}-none-linux_x86_64.whl

SITE_PKG_DIR=`python -c  'import sysconfig; print(sysconfig.get_paths()["purelib"])'`

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TENSERRT_DIR}/lib:${SITE_PKG_DIR}/torch/lib:${SITE_PKG_DIR}/tensorrt/:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs \

if [[ ! -d dist ]]; then
  mkdir dist
fi

# BUILD_VERSION is 2.3.0+cu121
# but we don't want the +cu121 in the wheel file name
TORCHTRT_VERSION=${BUILD_VERSION%+*}

libtorchtrt() {
    pre_cxx11_abi=${1}
    PROJECT_DIR=${CURRENT_DIR}
    cd ${PROJECT_DIR}
    # mkdir -p ${PROJECT_DIR}/py/wheelhouse

    PY_NAME=python${PYTHON_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}

    #python -m pip install -r ${PROJECT_DIR}/py/requirements.txt

    if [[ '${pre_cxx11_abi}' == 'true' ]]; then
      bazel build //:libtorchtrt --config pre_cxx11_abi --platforms //toolchains:ci_rhel_x86_64_linux -c opt --noshow_progress
      cp ${PROJECT_DIR}/bazel-bin/libtorchtrt.tar.gz \
      ${PROJECT_DIR}/dist/libtorchtrt-${TORCHTRT_VERSION}-pre-cxx11-abi-tensorrt${TRT_VERSION}-cuda${CU_VERSION}-libtorch${PYTORCH_VERSION}-x86_64-linux.tar.gz
    else
      bazel build //:libtorchtrt --platforms //toolchains:ci_rhel_x86_64_linux -c opt --noshow_progress
      cp ${PROJECT_DIR}/bazel-bin/libtorchtrt.tar.gz \
      ${PROJECT_DIR}/dist/libtorchtrt-${TORCHTRT_VERSION}-tensorrt${TRT_VERSION}-cuda${CU_VERSION}-libtorch${PYTORCH_VERSION}-x86_64-linux.tar.gz
    fi
}

# # build pre_cxx11_abi
# libtorchtrt true
# # build cxx11_abi
# libtorchtrt false

# auditwheel repair
python -m auditwheel repair \
 $(cat py/ci/soname_excludes.params) \
 --plat manylinux_2_34_x86_64 \
 /opt/torch-tensorrt-builds/torch_tensorrt-*-${PY_BUILD_CODE}-linux_x86_64.whl

cp wheelhouse/torch_tensorrt*x86_64.whl dist/


BUILD_VERSION=${TORCHTRT_VERSION} CI_BUILD=1 RELEASE=1 python setup.py bdist_wheel --release