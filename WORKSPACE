workspace(name = "Torch-TensorRT")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    sha256 = "863ba0fa944319f7e3d695711427d9ad80ba92c6edd0b7c7443b84e904689539",
    strip_prefix = "rules_python-0.22.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.22.0/rules_python-0.22.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

http_archive(
    name = "rules_pkg",
    sha256 = "8f9ee2dc10c1ae514ee599a8b42ed99fa262b757058f65ad3c384289ff70c4b8",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
    ],
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

http_archive(
    name = "googletest",
    sha256 = "755f9a39bc7205f5a0c428e920ddad092c33c8a1b46997def3f1d4a82aded6e1",
    strip_prefix = "googletest-5ab508a01f9eb089207ee87fd547d290da39d015",
    urls = ["https://github.com/google/googletest/archive/5ab508a01f9eb089207ee87fd547d290da39d015.zip"],
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
    path = "/usr/local/cuda-12.4/",
)

new_local_repository(
    name = "cuda_win",
    build_file = "@//third_party/cuda:BUILD",
    path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/",
)

#############################################################################################################
# Tarballs and fetched dependencies (default - use in cases when building from precompiled bin and tarballs)
#############################################################################################################

# http_archive(
#     name = "libtorch",
#     build_file = "@//third_party/libtorch:BUILD",
#     strip_prefix = "libtorch",
#     urls = ["https://download.pytorch.org/libtorch/nightly/cu124/libtorch-cxx11-abi-shared-with-deps-latest.zip"],
# )

# http_archive(
#     name = "libtorch_pre_cxx11_abi",
#     build_file = "@//third_party/libtorch:BUILD",
#     strip_prefix = "libtorch",
#     urls = ["https://download.pytorch.org/libtorch/nightly/cu124/libtorch-shared-with-deps-latest.zip"],
# )

http_archive(
    name = "libtorch_win",
    build_file = "@//third_party/libtorch:BUILD",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/nightly/cu124/libtorch-win-shared-with-deps-latest.zip"],
)

# Download these tarballs manually from the NVIDIA website
# Either place them in the distdir directory in third_party and use the --distdir flag
# or modify the urls to "file:///<PATH TO TARBALL>/<TARBALL NAME>.tar.gz

http_archive(
    name = "tensorrt",
    build_file = "@//third_party/tensorrt/archive:BUILD",
    sha256 = "a5cd2863793d69187ce4c73b2fffc1f470ff28cfd91e3640017e53b8916453d5",
    strip_prefix = "TensorRT-10.0.1.6",
    urls = [
        "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz",
    ],
)

http_archive(
    name = "tensorrt_win",
    build_file = "@//third_party/tensorrt/archive:BUILD",
    sha256 = "d667bd10b178e239b621a8929008ef3e27967d181bf07a39845a0f99edeec47a",
    strip_prefix = "TensorRT-10.0.1.6",
    urls = [
        "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/zip/TensorRT-10.0.1.6.Windows10.win10.cuda-12.4.zip",
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

new_local_repository(
    name = "libtorch",
    build_file = "third_party/libtorch/BUILD",
    path = "/usr/local/lib/python3.6/dist-packages/torch",
)

new_local_repository(
    name = "libtorch_pre_cxx11_abi",
    build_file = "third_party/libtorch/BUILD",
    path = "/usr/local/lib/python3.6/dist-packages/torch",
)

#new_local_repository(
#   name = "tensorrt",
#   path = "/usr/",
#   build_file = "@//third_party/tensorrt/local:BUILD"
#)

#########################################################################
# Development Dependencies (optional - comment out on aarch64)
#########################################################################

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "devtools_deps",
    requirements = "//:requirements-dev.txt",
)

load("@devtools_deps//:requirements.bzl", "install_deps")

install_deps()
