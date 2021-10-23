ARG BASE=21.09
ARG BASE_IMG=nvcr.io/nvidia/pytorch:${BASE}-py3
FROM ${BASE_IMG} as base

FROM base as torch-tensorrt-builder-base

# Removing any bazel or torch-tensorrt pre-installed from the base image

RUN rm -rf /opt/torch-tensorrt /usr/bin/bazel

RUN apt-get update && apt-get install --no-install-recommends -y curl gnupg
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > /etc/apt/trusted.gpg.d/bazel.gpg
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

RUN apt-get update && apt-get install -y --no-install-recommends bazel-4.0.0
RUN ln -s /usr/bin/bazel-4.0.0 /usr/bin/bazel

# Workaround for bazel expecting both static and shared versions, we only use shared libraries inside container
RUN cp /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer_static.a

RUN apt-get update && apt-get install -y --no-install-recommends locales ninja-build && rm -rf /var/lib/apt/lists/* && locale-gen en_US.UTF-8

FROM torch-tensorrt-builder-base as torch-tensorrt-builder

COPY . /workspace/torch_tensorrt/src
WORKDIR /workspace/torch_tensorrt/src
RUN cp ./docker/WORKSPACE.docker WORKSPACE

# This script builds both libtorchtrt bin/lib/include tarball and the Pythin wheel, in dist/
RUN ./docker/dist-build.sh

FROM base as torch-tensorrt

# copy source repo
COPY . /workspace/torch_tensorrt
COPY --from=torch-tensorrt-builder  /workspace/torch_tensorrt/src/dist/ .
RUN patch -u /opt/conda/lib/python3.8/site-packages/pytorch_quantization/nn/modules/tensor_quantizer.py -i /workspace/torch_tensorrt/docker/qat.patch
RUN conda init bash

RUN pip3 install ipywidgets --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org
RUN jupyter nbextension enable --py widgetsnbextension

RUN pip3 install *.whl && rm -fr /workspace/torch_tensorrt/dist/* *.whl

ENV LD_LIBRARY_PATH /opt/conda/lib/python3.8/site-packages/torch/lib:/opt/conda/lib/python3.8/site-packages/torch_tensorrt/lib:${LD_LIBRARY_PATH}
ENV PATH /opt/conda/lib/python3.8/site-packages/torch_tensorrt/bin:${PATH}

WORKDIR /workspace
RUN mv /workspace/torch_tensorrt /opt/pytorch/torch_tensorrt
RUN mkdir torch_tensorrt
RUN ln -s /opt/pytorch/torch_tensorrt/notebooks /workspace/torch_tensorrt/notebooks

CMD /bin/bash
