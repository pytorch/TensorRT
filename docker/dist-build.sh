#!/bin/bash

mkdir -p dist

bazel build //:libtrtorch --compilation_mode opt

cd py && MAX_JOBS=1 LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 python3 setup.py bdist_wheel --use-cxx11-abi

cd ..

cp bazel-bin/libtrtorch.tar.gz dist/
cp py/dist/* dist/

pip3 install ipywidgets --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org
jupyter nbextension enable --py widgetsnbextension

pip3 install timm

# test install
mkdir -p /opt/trtorch && tar xvf dist/libtrtorch.tar.gz --strip-components 2 -C /opt/trtorch --exclude=LICENSE && pip3 uninstall -y trtorch && pip3 install dist/*.whl

