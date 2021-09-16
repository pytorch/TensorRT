#!/bin/bash

mkdir -p dist

bazel build //:libtrtorch //:bin --compilation_mode opt

cd py && MAX_JOBS=1 LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 python3 setup.py bdist_wheel --use-cxx11-abi

cd ..

mv bazel-bin/libtrtorch.tar.gz dist/
mv py/dist/* dist/
