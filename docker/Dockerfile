ARG BASE=21.07
FROM nvcr.io/nvidia/pytorch:${BASE}-py3 as base

FROM base as trtorch-builder-base

RUN apt-get update && apt-get install -y curl gnupg
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > /etc/apt/trusted.gpg.d/bazel.gpg
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

RUN apt-get update && apt-get install -y bazel-4.0.0
RUN ln -s /usr/bin/bazel-4.0.0 /usr/bin/bazel

# Workaround for bazel expecting both static and shared versions, we only use shared libraries inside container
RUN cp /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer_static.a

RUN apt-get update && apt-get install -y locales ninja-build && rm -rf /var/lib/apt/lists/* && locale-gen en_US.UTF-8

FROM trtorch-builder-base as trtorch-builder

COPY . /workspace/trtorch/src
WORKDIR /workspace/trtorch/src
RUN cp ./docker/WORKSPACE.cu.docker  WORKSPACE

RUN ./docker/dist-build.sh

FROM base as trtorch

# copy source repo
COPY . /workspace/trtorch
COPY --from=trtorch-builder  /workspace/trtorch/src/dist/ .

RUN conda init bash

RUN pip install ipywidgets --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org
RUN jupyter nbextension enable --py widgetsnbextension

RUN mkdir -p /opt/trtorch && tar xvf libtrtorch.tar.gz --strip-components 2 -C /opt/trtorch --exclude=LICENSE && pip3 install *.whl && rm -fr /workspace/trtorch/dist/*

ENV LD_LIBRARY_PATH /opt/conda/lib/python3.8/site-packages/torch/lib:/opt/trtorch/lib:${LD_LIBRARY_PATH}
ENV PATH /opt/trtorch/bin:${PATH}

WORKDIR /workspace/trtorch/
CMD /bin/bash
