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
    path = "/usr/local/cuda-10.2/",
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
    urls = ["https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.5.1.zip"],
    sha256 = "0efdd4e709ab11088fa75f0501c19b0e294404231442bab1d1fb953924feb6b5"
)

http_archive(
    name = "libtorch_pre_cxx11_abi",
    build_file = "@//third_party/libtorch:BUILD",
    strip_prefix = "libtorch",
    sha256 = "818977576572eadaf62c80434a25afe44dbaa32ebda3a0919e389dcbe74f8656",
    urls = ["https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.5.1.zip"],
)

# Download these tarballs manually from the NVIDIA website
# Either place them in the distdir directory in third_party and use the --distdir flag
# or modify the urls to "file:///<PATH TO TARBALL>/<TARBALL NAME>.tar.gz

http_archive(
    name = "cudnn",
    urls = ["https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.1.13/10.2_20200626/cudnn-10.2-linux-x64-v8.0.1.13.tgz",],
    build_file = "@//third_party/cudnn/archive:BUILD",
    sha256 = "0c106ec84f199a0fbcf1199010166986da732f9b0907768c9ac5ea5b120772db",
    strip_prefix = "cuda"
)

http_archive(
    name = "tensorrt",
    urls = ["https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.1/tars/TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz",],
    build_file = "@//third_party/tensorrt/archive:BUILD",
    sha256 = "9205bed204e2ae7aafd2e01cce0f21309e281e18d5bfd7172ef8541771539d41",
    strip_prefix = "TensorRT-7.1.3.4"
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

