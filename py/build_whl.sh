#!/bin/bash

# Example usage: docker run -it -v$(pwd)/..:/workspace/TRTorch build_trtorch_wheel /bin/bash /workspace/TRTorch/py/build_whl.sh

cd /workspace/Torch-TensorRT/py

export CXX=g++
export CUDA_HOME=/usr/local/cuda-11.3

build_py37() {
    /opt/python/cp37-cp37m/bin/python -m pip install -r requirements.txt
    /opt/python/cp37-cp37m/bin/python setup.py bdist_wheel --release
    #auditwheel repair --plat manylinux2014_x86_64
}

build_py38() {
    /opt/python/cp38-cp38/bin/python -m pip install -r requirements.txt
    /opt/python/cp38-cp38/bin/python setup.py bdist_wheel --release
    #auditwheel repair --plat manylinux2014_x86_64
}

build_py39() {
    /opt/python/cp39-cp39/bin/python -m pip install -r requirements.txt
    /opt/python/cp39-cp39/bin/python setup.py bdist_wheel --release
    #auditwheel repair --plat manylinux2014_x86_64
}

build_py310() {
    /opt/python/cp310-cp310/bin/python -m pip install -r requirements.txt
    /opt/python/cp310-cp310/bin/python setup.py bdist_wheel --release
    #auditwheel repair --plat manylinux2014_x86_64
}

#build_py311() {
#    /opt/python/cp311-cp311/bin/python -m pip install -r requirements.txt
#    /opt/python/cp311-cp311/bin/python setup.py bdist_wheel --release
    #auditwheel repair --plat manylinux2014_x86_64
#}

build_py37
build_py38
build_py39
build_py310
#build_py311
