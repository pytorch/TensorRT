FROM pytorch/manylinux-builder:cuda11.3

RUN yum install -y ninja-build

RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

RUN mkdir /workspace

WORKDIR /workspace
