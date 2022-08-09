#!/bin/bash

pip3 install timm --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org
# Build and run unit tests
cd tests/modules && python3 ./hub.py
cd ../..

bazel test //tests:tests //tests:python_api_tests --compilation_mode=opt --jobs=4 --define=torchtrt_src=prebuilt
