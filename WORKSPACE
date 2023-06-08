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
    strip_prefix = "googletest-1.13.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip"],
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
    path = "/usr/local/cuda-11.8/",
)

#############################################################################################################
# Tarballs and fetched dependencies (default - use in cases when building from precompiled bin and tarballs)
#############################################################################################################

http_archive(
    name = "libtorch",
    build_file = "@//third_party/libtorch:BUILD",
    sha256 = "999becce82b73e566d0ffe010cd21fea8cf3a33f90f09dcc6b01150b820ae063",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/nightly/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0.dev20230605%2Bcu118.zip"],
)

http_archive(
    name = "libtorch_pre_cxx11_abi",
    build_file = "@//third_party/libtorch:BUILD",
    sha256 = "786cc728c63ea69c40bd8fb535cf8e5e1dfff1d43eaad3eb5256b9ed89c1b268",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/nightly/cu118/libtorch-shared-with-deps-2.1.0.dev20230605%2Bcu118.zip"],
)

# Download these tarballs manually from the NVIDIA website
# Either place them in the distdir directory in third_party and use the --distdir flag
# or modify the urls to "file:///<PATH TO TARBALL>/<TARBALL NAME>.tar.gz

http_archive(
    name = "cudnn",
    build_file = "@//third_party/cudnn/archive:BUILD",
    sha256 = "62046476076d1b455fb13fb78b38d3496bacafbfdbb0494fb3392240a04466ea",
    strip_prefix = "cudnn-linux-x86_64-8.8.0.121_cuda11-archive",
    urls = [
        "http://cuda-repo/release-candidates/kitpicks/cudnn-v8-8-cuda-11-8/8.8.0.121/001/redist/cudnn/cudnn/linux-x86_64/cudnn-linux-x86_64-8.8.0.121_cuda11-archive.tar.xz",
        #"https://developer.download.nvidia.com/compute/cudnn/secure/8.8.0/local_installers/11.8/cudnn-linux-x86_64-8.8.0.121_cuda11-archive.tar.xz",
    ],
)

http_archive(
    name = "tensorrt",
    build_file = "@//third_party/tensorrt/archive:BUILD",
    sha256 = "15bfe6053d45feec45ecc7123a9106076b0b43fa0435f242d89dca0778337759",
    strip_prefix = "TensorRT-8.6.1.6",
    urls = [
        "file:///home/narens/Developer/opensource/pytorch_org/tensorrt/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz",
        #"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz",
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

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "devtools_deps",
    requirements = "//:requirements-dev.txt",
)

load("@devtools_deps//:requirements.bzl", "install_deps")

install_deps()
