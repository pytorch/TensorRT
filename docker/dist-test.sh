#!/bin/bash

# Build and run unit tests
cd tests/modules && python3 ./hub.py
cd ../..

bazel test //tests:tests //tests:python_api_tests --compilation_mode opt --jobs 4

