#!/bin/bash

mkdir -p dist

cd py && MAX_JOBS=1 LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 python3 setup.py bdist_wheel --use-cxx11-abi

cd ..

cp py/dist/* dist/

pip3 install ipywidgets --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org
jupyter nbextension enable --py widgetsnbextension

pip3 install timm

# test install
mkdir -p /opt/torch_tensorrt && pip3 uninstall -y torch_tensorrt && pip3 install dist/*.whl

