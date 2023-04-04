# Base image starts with CUDA
ARG BASE_IMG=nvidia/cuda:11.7.1-devel-ubuntu22.04
FROM ${BASE_IMG} as base

ARG TENSORRT_VERSION
RUN test -n "$TENSORRT_VERSION" || (echo "No tensorrt version specified, please use --build-arg TENSORRT_VERSION=x.y.z to specify a version." && exit 1)
ARG CUDNN_VERSION
RUN test -n "$CUDNN_VERSION" || (echo "No cudnn version specified, please use --build-arg CUDNN_VERSION=x.y.z to specify a version." && exit 1)

ARG USE_CXX11_ABI
ENV USE_CXX11=${USE_CXX11_ABI}
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update
RUN apt install -y build-essential manpages-dev wget zlib1g software-properties-common git
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.8 python3.8-distutils python3.8-dev
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN python get-pip.py
RUN pip3 install wheel

# Install CUDNN + TensorRT
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/7fa2af80.pub
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 536F8F1DE80F6A35
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
RUN apt-get update
RUN apt-get install -y libcudnn8=${CUDNN_VERSION}* libcudnn8-dev=${CUDNN_VERSION}*

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
RUN apt-get update

RUN apt-get install -y libnvinfer8=${TENSORRT_VERSION}* libnvinfer-plugin8=${TENSORRT_VERSION}* libnvinfer-dev=${TENSORRT_VERSION}* libnvinfer-plugin-dev=${TENSORRT_VERSION}* libnvonnxparsers8=${TENSORRT_VERSION}-1* libnvonnxparsers-dev=${TENSORRT_VERSION}-1* libnvparsers8=${TENSORRT_VERSION}-1*  libnvparsers-dev=${TENSORRT_VERSION}-1*

# Setup Bazel
RUN wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64 -O /usr/bin/bazel \
 && chmod a+x /usr/bin/bazel

# Build Torch-TensorRT in an auxillary container
FROM base as torch-tensorrt-builder-base

ARG ARCH="x86_64"
ARG TARGETARCH="amd64"

RUN apt-get install -y python3-setuptools
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

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
RUN pip install -r /opt/torch_tensorrt/py/requirements.txt
RUN pip install tensorrt==${TENSORRT_VERSION}.*
RUN pip3 install *.whl && rm -fr /workspace/torch_tensorrt/py/dist/* *.whl

WORKDIR /opt/torch_tensorrt
ENV LD_LIBRARY_PATH /usr/local/lib/python3.8/dist-packages/torch/lib:/usr/local/lib/python3.8/dist-packages/torch_tensorrt/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV PATH /usr/local/lib/python3.8/dist-packages/torch_tensorrt/bin:${PATH}

CMD /bin/bash
