FROM pytorch/manylinux-cuda113

RUN yum install -y ninja-build

RUN wget --no-check-certificate https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo \
    && mv vbatts-bazel-epel-7.repo /etc/yum.repos.d/

RUN yum install -y bazel4 --nogpgcheck

RUN mkdir /workspace

WORKDIR /workspace