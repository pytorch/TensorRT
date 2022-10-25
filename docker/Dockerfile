# Base image starts with CUDA
ARG BASE_IMG=nvidia/cuda:11.7.1-devel-ubuntu18.04
FROM ${BASE_IMG} as base

# Install basic dependencies
RUN apt-get update
RUN apt install -y build-essential
RUN apt-get install -y manpages-dev
RUN apt install -y wget zlib1g software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.8 python3.8-distutils python3.8-dev
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py
RUN pip3 install wheel

# Install Pytorch
RUN pip3 install torch==1.13.0.dev20221006+cu117 torchvision==0.15.0.dev20221006+cu117 --extra-index-url https://download.pytorch.org/whl/nightly/cu117

# Install CUDNN + TensorRT
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 536F8F1DE80F6A35
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
RUN apt-get update
RUN apt-get install -y libcudnn8=8.4.1* libcudnn8-dev=8.4.1*

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
RUN apt-get update

RUN apt-get install -y libnvinfer8=8.4.3* libnvinfer-plugin8=8.4.3* libnvinfer-dev=8.4.3* libnvinfer-plugin-dev=8.4.3* libnvonnxparsers8=8.4.3-1* libnvonnxparsers-dev=8.4.3-1* libnvparsers8=8.4.3-1*  libnvparsers-dev=8.4.3-1*

# Build Torch-TensorRT in an auxillary container
FROM base as torch-tensorrt-builder-base

ARG ARCH="x86_64"
ARG TARGETARCH="amd64"
ARG BAZEL_VERSION=5.2.0

RUN apt-get install -y python3-setuptools
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN  apt-get update && apt-get install -y wget && apt-get install -y git

RUN wget -q https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64 -O /usr/bin/bazel \
 && chmod a+x /usr/bin/bazel

RUN apt-get update && apt-get install -y --no-install-recommends locales ninja-build && rm -rf /var/lib/apt/lists/* && locale-gen en_US.UTF-8

FROM torch-tensorrt-builder-base as torch-tensorrt-builder

COPY . /workspace/torch_tensorrt/src
WORKDIR /workspace/torch_tensorrt/src
RUN cp ./docker/WORKSPACE.docker WORKSPACE

# Remove cuda symlink to avoid bazel circle symlink errors
RUN rm /usr/local/cuda-11.7/cuda-11.7

# This script builds both libtorchtrt bin/lib/include tarball and the Python wheel, in dist/
RUN ./docker/dist-build.sh

# Copy and install Torch-TRT into the main container
FROM base as torch-tensorrt

COPY . /opt/torch_tensorrt
COPY --from=torch-tensorrt-builder  /workspace/torch_tensorrt/src/py/dist/ .

RUN cp /opt/torch_tensorrt/docker/WORKSPACE.docker /opt/torch_tensorrt/WORKSPACE
RUN pip3 install *.whl && rm -fr /workspace/torch_tensorrt/py/dist/* *.whl

# Install native tensorrt python package required by torch_tensorrt whl file
RUN pip install nvidia-pyindex
RUN pip install nvidia-tensorrt

WORKDIR /opt/torch_tensorrt
ENV LD_LIBRARY_PATH /usr/local/lib/python3.8/dist-packages/torch/lib:/usr/local/lib/python3.8/dist-packages/torch_tensorrt/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV PATH /usr/local/lib/python3.8/dist-packages/torch_tensorrt/bin:${PATH}

CMD /bin/bash
