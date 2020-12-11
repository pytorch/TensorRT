workspace(name = "TRTorch")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    commit = "4fcc24fd8a850bdab2ef2e078b1de337eea751a6",
    shallow_since = "1589292086 -0400"
)

load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

load("@rules_python//python:pip.bzl", "pip_repositories", "pip3_import")
pip_repositories()

http_archive(
    name = "rules_pkg",
    url = "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
    sha256 = "4ba8f4ab0ff85f2484287ab06c0d871dcb31cc54d439457d28fd4ae14b18450a",
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

git_repository(
    name = "googletest",
    remote = "https://github.com/google/googletest",
    commit = "703bd9caab50b139428cea1aaff9974ebee5742e",
    shallow_since = "1570114335 -0400"
)

# CUDA should be installed on the system locally
new_local_repository(
    name = "cuda",
    path = "/usr/local/cuda-11.0/",
    build_file = "@//third_party/cuda:BUILD",
)

new_local_repository(
    name = "cublas",
    path = "/usr",
    build_file = "@//third_party/cublas:BUILD",
)

#############################################################################################################
# Tarballs and fetched dependencies (default - use in cases when building from precompiled bin and tarballs)
#############################################################################################################

http_archive(
    name = "libtorch",
    build_file = "@//third_party/libtorch:BUILD",
    strip_prefix = "libtorch",
    sha256 = "656db919c00e99dac81bc21598845ba9231e57cbc8e1570bea017dfb9298236d",
    urls = ["https://download.pytorch.org/libtorch/cu110/libtorch-cxx11-abi-shared-with-deps-1.7.0%2Bcu110.zip"],
)

http_archive(
    name = "libtorch_pre_cxx11_abi",
    build_file = "@//third_party/libtorch:BUILD",
    strip_prefix = "libtorch",
    sha256 = "b3976f3294a6c20be4807191042d1e74162a41fb1bed0919f637afc3517d76be",
    urls = ["https://download.pytorch.org/libtorch/cu110/libtorch-shared-with-deps-1.7.0%2Bcu110.zip"],
)

# Download these tarballs manually from the NVIDIA website
# Either place them in the distdir directory in third_party and use the --distdir flag
# or modify the urls to "file:///<PATH TO TARBALL>/<TARBALL NAME>.tar.gz

http_archive(
    name = "cudnn",
    urls = ["https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.0_20201106/cudnn-11.0-linux-x64-v8.0.5.39.tgz",],
    build_file = "@//third_party/cudnn/archive:BUILD",
    sha256 = "4e16ee7895deb4a8b1c194b812ba49586ef7d26902051401d3717511898a9b73",
    strip_prefix = "cuda"
)

http_archive(
    name = "tensorrt",
    urls = ["https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.1/tars/TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz",],
    build_file = "@//third_party/tensorrt/archive:BUILD",
    sha256 = "8def6b03b0c8c3751f560df21b3e99668ae05aab5140b1d38b8e51e4a0ffbbb8",
    strip_prefix = "TensorRT-7.2.1.6"
)


####################################################################################
# Locally installed dependencies (use in cases of custom dependencies or aarch64)
####################################################################################

# NOTE: In the case you are using just the pre-cxx11-abi path or just the cxx11 abi path
# with your local libtorch, just point deps at the same path to satisfy bazel.

# NOTE: NVIDIA's aarch64 PyTorch (python) wheel file uses the CXX11 ABI unlike PyTorch's standard
# x86_64 python distribution. If using NVIDIA's version just point to the root of the package
# for both versions here and do not use --config=pre-cxx11-abi

#new_local_repository(
#    name = "libtorch",
#    path = "/usr/local/lib/python3.6/dist-packages/torch",
#    build_file = "third_party/libtorch/BUILD"
#)

#new_local_repository(
#    name = "libtorch_pre_cxx11_abi",
#    path = "/usr/local/lib/python3.6/dist-packages/torch",
#    build_file = "third_party/libtorch/BUILD"
#)

#new_local_repository(
#    name = "cudnn",
#    path = "/usr/",
#    build_file = "@//third_party/cudnn/local:BUILD"
#)

#new_local_repository(
#   name = "tensorrt",
#   path = "/usr/",
#   build_file = "@//third_party/tensorrt/local:BUILD"
#)

#########################################################################
# Testing Dependencies (optional - comment out on aarch64)
#########################################################################
pip3_import(
    name = "trtorch_py_deps",
    requirements = "//py:requirements.txt"
)

load("@trtorch_py_deps//:requirements.bzl", "pip_install")
pip_install()

pip3_import(
    name = "py_test_deps",
    requirements = "//tests/py:requirements.txt"
)

load("@py_test_deps//:requirements.bzl", "pip_install")
pip_install()

pip3_import(
   name = "pylinter_deps",
   requirements = "//tools/linter:requirements.txt",
)

load("@pylinter_deps//:requirements.bzl", "pip_install")
pip_install()
