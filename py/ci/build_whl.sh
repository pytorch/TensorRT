#!/bin/bash

# Example usage: docker run -it -v$(pwd)/..:/workspace/TRTorch build_trtorch_wheel /bin/bash /workspace/TRTorch/py/build_whl.sh

cd /workspace/Torch-TensorRT/py

export CXX=g++
export CUDA_HOME=/usr/local/cuda-11.3

build_py37() {
    /opt/python/cp37-cp37m/bin/python -m pip install -r requirements.txt
    /opt/python/cp37-cp37m/bin/python setup.py bdist_wheel --release --ci
    #auditwheel repair --plat manylinux2014_x86_64
}

build_py38() {
    /opt/python/cp38-cp38/bin/python -m pip install -r requirements.txt
    /opt/python/cp38-cp38/bin/python setup.py bdist_wheel --release --ci
    #auditwheel repair --plat manylinux2014_x86_64
}

build_py39() {
    /opt/python/cp39-cp39/bin/python -m pip install -r requirements.txt
    /opt/python/cp39-cp39/bin/python setup.py bdist_wheel --release --ci
    #auditwheel repair --plat manylinux2014_x86_64
}

build_py310() {
    /opt/python/cp310-cp310/bin/python -m pip install -r requirements.txt
    /opt/python/cp310-cp310/bin/python setup.py bdist_wheel --release --ci
    #auditwheel repair --plat manylinux2014_x86_64
}

build_libtorchtrt() {
    bazel clean
    bazel build //:libtorchtrt --platforms //toolchains:ci_rhel_x86_64_linux -c opt
    CUDA_VERSION=$(cd torch_tensorrt && python3 -c "from _version import __cuda_version__;print(__cuda_version__)")
    TORCHTRT_VERSION=$(cd torch_tensorrt && python3 -c "from _version import __version__;print(__version__)")
    TRT_VERSION=$(cd torch_tensorrt && python3 -c "from _version import __tensorrt_version__;print(__tensorrt_version__)")
    CUDNN_VERSION=$(cd torch_tensorrt && python3 -c "from _version import __cudnn_version__;print(__cudnn_version__)")
    TORCH_VERSION=$(/opt/python/cp310-cp310/bin/python -c "from torch import __version__;print(__version__.split('+')[0])")
    cp ../bazel-bin/libtorchtrt.tar.gz dist/libtorchtrt-${TORCHTRT_VERSION}-cudnn${CUDNN_VERSION}-tensorrt${TRT_VERSION}-cuda${CUDA_VERSION}-libtorch-${TORCH_VERSION}.tar.gz
}

build_libtorchtrt_pre_cxx11_abi() {
    bazel build //:libtorchtrt --config pre_cxx11_abi --platforms //toolchains:ci_rhel_x86_64_linux -c opt
    CUDA_VERSION=$(cd torch_tensorrt && python3 -c "from _version import __cuda_version__;print(__cuda_version__)")
    TORCHTRT_VERSION=$(cd torch_tensorrt && python3 -c "from _version import __version__;print(__version__)")
    TRT_VERSION=$(cd torch_tensorrt && python3 -c "from _version import __tensorrt_version__;print(__tensorrt_version__)")
    CUDNN_VERSION=$(cd torch_tensorrt && python3 -c "from _version import __cudnn_version__;print(__cudnn_version__)")
    TORCH_VERSION=$(/opt/python/cp310-cp310/bin/python -c "from torch import __version__;print(__version__.split('+')[0])")
    cp ../bazel-bin/libtorchtrt.tar.gz dist/libtorchtrt-${TORCHTRT_VERSION}-pre-cxx11-abi-cudnn${CUDNN_VERSION}-tensorrt${TRT_VERSION}-cuda${CUDA_VERSION}-libtorch-${TORCH_VERSION}.tar.gz

}

build_py37
build_py38
build_py39
build_py310
build_libtorchtrt_pre_cxx11_abi
build_libtorchtrt
