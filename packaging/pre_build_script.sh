#!/bin/bash

set -exou pipefail

# Install dependencies
python3 -m pip install pyyaml
yum install -y ninja-build gettext
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

export TORCH_BUILD_NUMBER=$(python -c "import torch, urllib.parse as ul; print(ul.quote_plus(torch.__version__))")
export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")

cat toolchains/ci_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel
export CI_BUILD=1
