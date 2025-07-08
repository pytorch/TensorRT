#!/bin/bash

# This script installs all the external dependencies required for Torch-TensorRT
# It defines the required variables and installs the dependencies

set -x

# Set default values for CHANNEL and CU_VERSION if not defined
CHANNEL=${CHANNEL:-nightly}
CU_VERSION=${CU_VERSION:-cu129}

# Define variables for torch and torchvision versions
TORCH=$(grep "^torch>" ${PWD}/py/requirements.txt)
TORCHVISION=$(grep "^torchvision>" ${PWD}/tests/py/requirements.txt)
INDEX_URL=https://download.pytorch.org/whl/${CHANNEL}/${CU_VERSION}

# Install all the dependencies required for Torch-TensorRT
pip install --pre -r ${PWD}/tests/py/requirements.txt
# dependencies in the tests/py/requirements.txt might install a different version of torch or torchvision
# eg. timm will install the latest torchvision, however we want to use the torchvision from nightly
# reinstall torch torchvision to make sure we have the correct version
pip uninstall -y torch torchvision
pip install --force-reinstall --pre ${TORCHVISION} --index-url ${INDEX_URL}
pip install --force-reinstall --pre ${TORCH} --index-url ${INDEX_URL}