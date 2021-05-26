workspace(name = "TRTorch")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "rules_python",
    sha256 = "778197e26c5fbeb07ac2a2c5ae405b30f6cb7ad1f5510ea6fdac03bded96cc6f",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.2.0/rules_python-0.2.0.tar.gz",
)

load("@rules_python//python:pip.bzl", "pip_install")

http_archive(
    name = "rules_pkg",
    sha256 = "038f1caa773a7e35b3663865ffb003169c6a71dc995e39bf4815792f385d837d",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.4.0/rules_pkg-0.4.0.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.4.0/rules_pkg-0.4.0.tar.gz",
    ],
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

git_repository(
    name = "googletest",
    commit = "703bd9caab50b139428cea1aaff9974ebee5742e",
    remote = "https://github.com/google/googletest",
    shallow_since = "1570114335 -0400",
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
#############################################################################################################
# Tarballs and fetched dependencies (default - use in cases when building from precompiled bin and tarballs)
#############################################################################################################

http_archive(
    name = "libtorch",
    build_file = "@//third_party/libtorch:BUILD",
    sha256 = "edc12091193ba772db77a6ec14e05cef6da881288fca0dfc89a031f631601f60",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu111.zip"],
)

http_archive(
    name = "libtorch_pre_cxx11_abi",
    build_file = "@//third_party/libtorch:BUILD",
    sha256 = "af9435fa4b44bb395c1a7645391c00228a72af4305f43a61e9300c0abdbe0819",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.9.0%2Bcu111.zip"],
)

# Download these tarballs manually from the NVIDIA website
# Either place them in the distdir directory in third_party and use the --distdir flag
# or modify the urls to "file:///<PATH TO TARBALL>/<TARBALL NAME>.tar.gz

#http_archive(
#    name = "cudnn",
#    urls = ["file:///home/boris/git/TRTorch/third_party/dist_dir/x86_64-linux-gnu/cudnn-11.2-linux-x64-v8.1.1.33.tgz",],
#    # "https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz",],
#    build_file = "@//third_party/cudnn/archive:BUILD",
#    sha256 = "98a8784e92862f20018d20c281b30d4a0cd951f93694f6433ccf4ae9c502ba6a",
#    strip_prefix = "cuda"
#)

#http_archive(
#    name = "tensorrt",
#    urls = ["file:///home/boris/git/TRTorch/third_party/dist_dir/x86_64-linux-gnu/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz"],
# "https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.3/tars/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz",],
#    build_file = "@//third_party/tensorrt/archive:BUILD",
#    strip_prefix = "TensorRT-7.2.3.4",
#    sha256 = "d3a1f478e304b48878604fac70ce7920fece71f9cac62f925c9c59c197f5d087"
#)

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
    name = "trtorch_py_deps",
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
