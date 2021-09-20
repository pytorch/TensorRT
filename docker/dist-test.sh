#!/bin/bash

# Build and run unit tests
bazel test //tests:tests //tests:python_api_tests --compilation_mode opt --jobs 4

