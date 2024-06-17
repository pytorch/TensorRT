#!/bin/bash

set -x

CURRENT_DIR=`pwd`

ls

python -m pip list | grep torch

python -m pip list | grep tensorrt

if [[ "${PYTHON_VERSION}" == "3.8" ]]; then
  PY_BUILD_CODE=cp38-cp38
elif [[ "${PYTHON_VERSION}" == "3.9" ]]; then
  PY_BUILD_CODE=cp39-cp39
elif [[ "${PYTHON_VERSION}" == "3.10" ]]; then
  PY_BUILD_CODE=cp310-cp310
elif [[ "${PYTHON_VERSION}" == "3.11" ]]; then
  PY_BUILD_CODE=cp38-cp38
else
   echo "python version: ${PYTHON_VERSION} is not supported"
   exit
fi

python -m pip install auditwheel

# download TensorRT tarball
wget -q https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz \
&& gunzip TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz \
&& tar -xvf TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar \
&& rm TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar

TENSERRT_DIR=${CURRENT_DIR}/TensorRT-10.0.1.6

SITE_PKG_DIR=`python -c  'import sysconfig; print(sysconfig.get_paths()["purelib"])'`

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TENSERRT_DIR}/lib:${SITE_PKG_DIR}/torch/lib:${SITE_PKG_DIR}/tensorrt/:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs \

python -m auditwheel repair \
 $(cat py/ci/soname_excludes.params) \
 --plat manylinux_2_34_x86_64 \
 /opt/torch-tensorrt-builds/torch_tensorrt-*-${PY_BUILD_CODE}-linux_x86_64.whl

if [[ ! -d dist ]]; then
  mkdir dist
fi
cp wheelhouse/torch_tensorrt*x86_64.whl dist/

CUDA_VERSION=$(python -c "import versions; versions.cuda_version()")
TORCHTRT_VERSION=$(python -c "import versions; versions.torch_tensorrt_version_release()")
TRT_VERSION=$(python -c "import versions; versions.tensorrt_version()")
TORCH_VERSION=$(python -c "from torch import __version__;print(__version__.split('+')[0])")

libtorchtrt() {
    pre_cxx11_abi=${1}
    PROJECT_DIR=${CURRENT_DIR}
    cd ${PROJECT_DIR}
    # mkdir -p ${PROJECT_DIR}/py/wheelhouse

    PY_NAME=python${PYTHON_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}

    python -m pip install -r ${PROJECT_DIR}/py/requirements.txt
    if [[ '${pre_cxx11_abi}' == 'true' ]]; then
      bazel build //:libtorchtrt --config pre_cxx11_abi --platforms //toolchains:ci_rhel_x86_64_linux -c opt --noshow_progress
      cp ${PROJECT_DIR}/bazel-bin/libtorchtrt.tar.gz \
      ${PROJECT_DIR}/dist/libtorchtrt-${TORCHTRT_VERSION}-pre-cxx11-abi-tensorrt${TRT_VERSION}-cuda${CUDA_VERSION}-libtorch${TORCH_VERSION}-x86_64-linux.tar.gz
    else
      bazel build //:libtorchtrt --platforms //toolchains:ci_rhel_x86_64_linux -c opt --noshow_progress
      cp ${PROJECT_DIR}/bazel-bin/libtorchtrt.tar.gz \
      ${PROJECT_DIR}/dist/libtorchtrt-${TORCHTRT_VERSION}-tensorrt${TRT_VERSION}-cuda${CUDA_VERSION}-libtorch${TORCH_VERSION}-x86_64-linux.tar.gz
    fi
}

# build pre_cxx11_abi
libtorchtrt true
# build cxx11_abi
libtorchtrt false
