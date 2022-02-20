workspace(name = "Torch-TensorRT")

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

# External dependency for torch_tensorrt if you already have precompiled binaries.
local_repository(
    name = "torch_tensorrt",
    path = "/opt/conda/lib/python3.8/site-packages/torch_tensorrt"
)

# CUDA should be installed on the system locally
new_local_repository(
    name = "cuda",
    build_file = "@//third_party/cuda:BUILD",
    path = "/usr/local/cuda-11.3/",
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
    # sha256 = "190e963e739d5f7c2dcf94b3994de8fcd335706a4ebb333812ea7d8c841beb06",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/nightly/cu113/libtorch-cxx11-abi-shared-with-deps-latest.zip"],
)

http_archive(
    name = "libtorch_pre_cxx11_abi",
    build_file = "@//third_party/libtorch:BUILD",
    # sha256 = "0996a6a4ea8bbc1137b4fb0476eeca25b5efd8ed38955218dec1b73929090053",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/nightly/cu113/libtorch-shared-with-deps-latest.zip"],
)

# Download these tarballs manually from the NVIDIA website
# Either place them in the distdir directory in third_party and use the --distdir flag
# or modify the urls to "file:///<PATH TO TARBALL>/<TARBALL NAME>.tar.gz

http_archive(
    name = "cudnn",
    build_file = "@//third_party/cudnn/archive:BUILD",
    sha256 = "0e5d2df890b9967efa6619da421310d97323565a79f05a1a8cb9b7165baad0d7",
    strip_prefix = "cuda",
    urls = [
        "https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.4/11.4_20210831/cudnn-11.4-linux-x64-v8.2.4.15.tgz",
    ],
)

http_archive(
    name = "tensorrt",
    build_file = "@//third_party/tensorrt/archive:BUILD",
    sha256 = "da130296ac6636437ff8465812eb55dbab0621747d82dc4fe9b9376f00d214af",
    strip_prefix = "TensorRT-8.2.2.1",
    urls = [
        "https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.2.1/tars/tensorrt-8.2.2.1.linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz",
    ],
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
