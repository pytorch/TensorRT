# Base image starts with CUDA
ARG BASE_IMG=nvidia/cuda:11.7.1-devel-ubuntu20.04
FROM ${BASE_IMG} as base

# Install basic dependencies
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt install -y build-essential manpages-dev wget zlib1g software-properties-common git
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.8 python3.8-distutils python3.8-dev
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN python get-pip.py
RUN pip3 install wheel

# Install Pytorch
RUN pip3 install torch==2.0.0.dev20230103+cu117 torchvision==0.15.0.dev20230103+cu117 --extra-index-url https://download.pytorch.org/whl/nightly/cu117

# Install CUDNN + TensorRT
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 536F8F1DE80F6A35
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update
RUN apt-get install -y libcudnn8=8.5.0* libcudnn8-dev=8.5.0*

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update

RUN apt-get install -y libnvinfer8=8.5.1* libnvinfer-plugin8=8.5.1* libnvinfer-dev=8.5.1* libnvinfer-plugin-dev=8.5.1* libnvonnxparsers8=8.5.1-1* libnvonnxparsers-dev=8.5.1-1* libnvparsers8=8.5.1-1*  libnvparsers-dev=8.5.1-1*

# Setup Bazel
ARG BAZEL_VERSION=5.2.0
RUN wget -q https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64 -O /usr/bin/bazel \
 && chmod a+x /usr/bin/bazel

# Build Torch-TensorRT in an auxillary container
FROM base as torch-tensorrt-builder-base

ARG ARCH="x86_64"
ARG TARGETARCH="amd64"

RUN apt-get install -y python3-setuptools
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN  apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends locales ninja-build && rm -rf /var/lib/apt/lists/* && locale-gen en_US.UTF-8

FROM torch-tensorrt-builder-base as torch-tensorrt-builder

COPY . /workspace/torch_tensorrt/src
WORKDIR /workspace/torch_tensorrt/src
RUN cp ./docker/WORKSPACE.docker WORKSPACE

# This script builds both libtorchtrt bin/lib/include tarball and the Python wheel, in dist/
RUN ./docker/dist-build.sh

# Copy and install Torch-TRT into the main container
FROM base as torch-tensorrt

COPY . /opt/torch_tensorrt
COPY --from=torch-tensorrt-builder  /workspace/torch_tensorrt/src/py/dist/ .

RUN cp /opt/torch_tensorrt/docker/WORKSPACE.docker /opt/torch_tensorrt/WORKSPACE
RUN pip3 install *.whl && rm -fr /workspace/torch_tensorrt/py/dist/* *.whl

# Install native tensorrt python package required by torch_tensorrt whl file
RUN pip install tensorrt==8.5.1.7

WORKDIR /opt/torch_tensorrt
ENV LD_LIBRARY_PATH /usr/local/lib/python3.8/dist-packages/torch/lib:/usr/local/lib/python3.8/dist-packages/torch_tensorrt/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV PATH /usr/local/lib/python3.8/dist-packages/torch_tensorrt/bin:${PATH}

CMD /bin/bash
