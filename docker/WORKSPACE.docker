workspace(name = "Torch-TensorRT")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
        name = "rules_python",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.2.0/rules_python-0.2.0.tar.gz",
        sha256 = "778197e26c5fbeb07ac2a2c5ae405b30f6cb7ad1f5510ea6fdac03bded96cc6f",
    )

load("@rules_python//python:pip.bzl", "pip_install")

http_archive(
    name = "rules_pkg",
    urls = [
    	"https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.4.0/rules_pkg-0.4.0.tar.gz",
	"https://github.com/bazelbuild/rules_pkg/releases/download/0.4.0/rules_pkg-0.4.0.tar.gz",
    ],
    sha256 = "038f1caa773a7e35b3663865ffb003169c6a71dc995e39bf4815792f385d837d",
)
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

git_repository(
    name = "googletest",
    remote = "https://github.com/google/googletest",
    commit = "703bd9caab50b139428cea1aaff9974ebee5742e",
    shallow_since = "1570114335 -0400"
)

# External dependency for torch_tensorrt if you already have precompiled binaries.
# This is currently used in pytorch NGC container CI testing.
local_repository(
    name = "torch_tensorrt",
    path = "/opt/conda/lib/python3.8/site-packages/torch_tensorrt"
)

# CUDA should be installed on the system locally
new_local_repository(
    name = "cuda",
    path = "/usr/local/cuda",
    build_file = "@//third_party/cuda:BUILD",
)

new_local_repository(
    name = "cublas",
    build_file = "@//third_party/cublas:BUILD",
    path = "/usr",
)

####################################################################################
# Locally installed dependencies (use in cases of custom dependencies or aarch64)
####################################################################################

new_local_repository(
    name = "libtorch",
    path = "/opt/conda/lib/python3.8/site-packages/torch",
    build_file = "third_party/libtorch/BUILD"
)

new_local_repository(
    name = "libtorch_pre_cxx11_abi",
    path = "/opt/conda/lib/python3.8/site-packages/torch",
    build_file = "third_party/libtorch/BUILD"
)

new_local_repository(
    name = "cudnn",
    path = "/usr/",
    build_file = "@//third_party/cudnn/local:BUILD"
)

new_local_repository(
    name = "tensorrt",
    path = "/usr/",
    build_file = "@//third_party/tensorrt/local:BUILD"
)

#########################################################################
# Testing Dependencies (optional - comment out on aarch64)
#########################################################################
pip_install(
    name = "torch_tensorrt_py_deps",
    requirements = "//py:requirements.txt",
)

pip_install(
    name = "py_test_deps",
    requirements = "//tests/py:requirements.txt",
)

pip_install(
    name = "pylinter_deps",
    requirements = "//tools/linter:requirements.txt",
)
