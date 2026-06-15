#!/bin/bash

set -x

TOP_DIR=$(cd $(dirname $0); pwd)/..

BUILD_CMD="python -m pip wheel .  --extra-index-url https://download.pytorch.org/whl/test/cu130 -w dist"

# TensorRT restricts our pip version
cd ${TOP_DIR} \
    && python -m pip install "pip<=23.1" wheel \
    && python -m pip install -r py/requirements.txt

# Build Torch-TRT
MAX_JOBS=4 LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 ${BUILD_CMD} $* || exit 1

python -m pip install ipywidgets --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org
jupyter nbextension enable --py widgetsnbextension
python -m pip install timm

# test install
python -m pip uninstall -y torch_tensorrt && python -m pip install ${TOP_DIR}/dist/*.whl
