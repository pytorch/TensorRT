FROM pytorch/manylinux-cuda102

RUN yum install -y ninja-build

RUN wget https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo \
    && mv vbatts-bazel-epel-7.repo /etc/yum.repos.d/

RUN yum install -y bazel3

RUN mv /usr/bin/ninja-build /usr/bin/ninja

RUN mkdir /workspace

WORKDIR /workspace