# syntax=docker/dockerfile:1

# Base image starts with CUDA
ARG BASE_IMG=nvidia/cuda:12.1.1-devel-ubuntu22.04
FROM ${BASE_IMG} as base
ENV BASE_IMG=nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG TENSORRT_VERSION
ENV TENSORRT_VERSION=${TENSORRT_VERSION}
RUN test -n "$TENSORRT_VERSION" || (echo "No tensorrt version specified, please use --build-arg TENSORRT_VERSION=x.y to specify a version." && exit 1)

ARG PYTHON_VERSION=3.10
ENV PYTHON_VERSION=${PYTHON_VERSION}

ARG USE_CXX11_ABI
ENV USE_CXX11=${USE_CXX11_ABI}
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update
RUN apt install -y build-essential manpages-dev wget zlib1g software-properties-common git libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget ca-certificates curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev mecab-ipadic-utf8

# Install PyEnv and desired Python version
ENV HOME="/root"
ENV PYENV_DIR="$HOME/.pyenv"
ENV PATH="$PYENV_DIR/shims:$PYENV_DIR/bin:$PATH"
RUN wget -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer &&\
    chmod 755 pyenv-installer &&\
    bash pyenv-installer &&\
    eval "$(pyenv init -)"

RUN pyenv install -v ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

# Install TensorRT + dependencies
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
RUN apt-get update

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
RUN apt-get update

RUN apt-get install -y libnvinfer8=${TENSORRT_VERSION}.* libnvinfer-plugin8=${TENSORRT_VERSION}.* libnvinfer-dev=${TENSORRT_VERSION}.* libnvinfer-plugin-dev=${TENSORRT_VERSION}.* libnvonnxparsers8=${TENSORRT_VERSION}.* libnvonnxparsers-dev=${TENSORRT_VERSION}.* libnvparsers8=${TENSORRT_VERSION}.*  libnvparsers-dev=${TENSORRT_VERSION}.* libnvinfer-headers-dev=${TENSORRT_VERSION}.* libnvinfer-headers-plugin-dev=${TENSORRT_VERSION}.*

# Setup Bazel via Bazelisk
RUN wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 -O /usr/bin/bazel &&\
    chmod a+x /usr/bin/bazel

# Build Torch-TensorRT in an auxillary container
FROM base as torch-tensorrt-builder-base

ARG ARCH="x86_64"
ARG TARGETARCH="amd64"

RUN apt-get update
RUN apt-get install -y python3-setuptools
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

RUN apt-get update &&\
    apt-get install -y --no-install-recommends locales ninja-build &&\
    rm -rf /var/lib/apt/lists/* &&\
    locale-gen en_US.UTF-8

FROM torch-tensorrt-builder-base as torch-tensorrt-builder

COPY . /workspace/torch_tensorrt/src
WORKDIR /workspace/torch_tensorrt/src
RUN cp ./docker/WORKSPACE.docker WORKSPACE

# Symlink the path pyenv is using for python with the /opt directory for package sourcing
RUN mkdir -p "/opt/python3/" &&\
    ln -s "`pyenv which python | xargs dirname | xargs dirname`/lib/python$PYTHON_VERSION/site-packages" "/opt/python3/"

# Extract base image cuda version (everything after :, before -, before final ., in BASE_IMG)
# Ensure the default cuda folder agrees with the version in the base image
RUN CUDA_BASE_IMG_VERSION_INTERMEDIATE=`echo ${BASE_IMG#*:}` &&\
    CUDA_BASE_IMG_VERSION=`echo ${CUDA_BASE_IMG_VERSION_INTERMEDIATE%%-*}` &&\
    CUDA_MAJOR_MINOR_VERSION=`echo ${CUDA_BASE_IMG_VERSION%.*}` &&\
    rm -fr /usr/local/cuda &&\
    ln -s /usr/local/cuda-${CUDA_MAJOR_MINOR_VERSION} /usr/local/cuda

ENV CUDA_HOME=/usr/local/cuda

# This script builds both libtorchtrt bin/lib/include tarball and the Python wheel, in dist/
RUN bash ./docker/dist-build.sh

# Copy and install Torch-TRT into the main container
FROM base as torch-tensorrt

COPY . /opt/torch_tensorrt

# Symlink the path pyenv is using for python with the /opt directory for package sourcing
RUN mkdir -p "/opt/python3/" &&\
    ln -s "`pyenv which python | xargs dirname | xargs dirname`/lib/python$PYTHON_VERSION/site-packages" "/opt/python3/"

COPY --from=torch-tensorrt-builder  /workspace/torch_tensorrt/src/dist/ .

RUN cp /opt/torch_tensorrt/docker/WORKSPACE.docker /opt/torch_tensorrt/WORKSPACE &&\
    pip install -r /opt/torch_tensorrt/py/requirements.txt &&\
    # Install all dependency wheel files and user-specified TensorRT
    pip install *.whl &&\
    pip install tensorrt==${TENSORRT_VERSION}.* &&\
    # Add the Torch-TensorRT wheel file to the dist directory and delete all other .whl files
    rm -fr /workspace/torch_tensorrt/dist/* &&\
    mkdir -p /opt/torch_tensorrt/dist/ && mv torch_tensorrt*.whl /opt/torch_tensorrt/dist/ &&\
    rm -fr *.whl &&\
    # Remove other cache files if present
    pip cache purge && rm -rf /opt/torch_tensorrt/.mypy_cache

WORKDIR /opt/torch_tensorrt

ENV LD_LIBRARY_PATH /opt/python3/site-packages/torch/lib:/opt/python3/site-packages/torch_tensorrt/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV PATH /opt/python3/site-packages/torch_tensorrt/bin:${PATH}

CMD /bin/bash
