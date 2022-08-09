#!/bin/bash

TOP_DIR=$(cd $(dirname $0); pwd)/..

if [[ -z "${USE_CXX11}" ]]; then
    BUILD_CMD="python3 setup.py bdist_wheel"
else
    BUILD_CMD="python3 setup.py bdist_wheel  --use-cxx11-abi"
fi

cd ${TOP_DIR} \
    && mkdir -p dist && cd py \
    && pip install -r requirements.txt \
    && MAX_JOBS=1 LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 \
        ${BUILD_CMD} $* || exit 1

pip3 install ipywidgets --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org
jupyter nbextension enable --py widgetsnbextension
pip3 install timm

# test install
pip3 uninstall -y torch_tensorrt && pip3 install ${TOP_DIR}/py/dist/*.whl
