workspace(name = "Torch-TensorRT")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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
    path = "/opt/conda/lib/python3.8/site-packages/torch_tensorrt",
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
    sha256 = "80f089939de20e68e3fcad4dfa72a26c8bf91b5e77b11042f671f39ebac35865",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcu113.zip"],
)

http_archive(
    name = "libtorch_pre_cxx11_abi",
    build_file = "@//third_party/libtorch:BUILD",
    sha256 = "8e35371403f7052d9e9b43bcff383980dbde4df028986dc1dab539953481d55f",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.12.0%2Bcu113.zip"],
)

http_archive(
    name = "libcudacxx",
    build_file = "@third_party/cuda:BUILD",
    sha256 = "ea74852b7ef808cc34b0bc4af3e0bd093bf50848ee1793d1abcaf9eb36506fc7",
    strip_prefix = "libcudacxx-1.8.0-post-release",
    urls = ["https://github.com/NVIDIA/libcudacxx/archive/refs/tags/1.8.0-post-release.zip"]
)

# Download these tarballs manually from the NVIDIA website
# Either place them in the distdir directory in third_party and use the --distdir flag
# or modify the urls to "file:///<PATH TO TARBALL>/<TARBALL NAME>.tar.gz

http_archive(
    name = "cudnn",
    build_file = "@//third_party/cudnn/archive:BUILD",
    sha256 = "7f3fbe6201708de409532a32d647af6b4bdb10d7f045d557270549e286487289",
    strip_prefix = "cudnn-linux-x86_64-8.4.1.114_cuda11.4-archive",
    urls = [
        "https://developer.nvidia.com/compute/cudnn/secure/8.4.1/local_installers/11.6/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz",
    ],
)

http_archive(
    name = "tensorrt",
    build_file = "@//third_party/tensorrt/archive:BUILD",
    sha256 = "8107861af218694130f170e071f49814fa3e27f1386ce7cb6d807ac05a7fcf0e",
    strip_prefix = "TensorRT-8.4.1.5",
    urls = [
        "https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.4.1/tars/tensorrt-8.4.1.5.linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz",
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
# Development Dependencies (optional - comment out on aarch64)
#########################################################################

pip_install(
    name = "devtools_deps",
    requirements = "//:requirements-dev.txt",
)
