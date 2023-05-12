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
    path = "/usr/local/cuda-11.8/",
)

#############################################################################################################
# Tarballs and fetched dependencies (default - use in cases when building from precompiled bin and tarballs)
#############################################################################################################

http_archive(
    name = "libtorch",
    build_file = "@//third_party/libtorch:BUILD",
    sha256 = "1a526a9cd19c1015674d26921dbb94bcd2d632a6f9c431a21c43f4e24768d834",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/nightly/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0.dev20230419%2Bcu118.zip"],
)

http_archive(
    name = "libtorch_pre_cxx11_abi",
    build_file = "@//third_party/libtorch:BUILD",
    sha256 = "60c5912a5085a6a7073b3804b10d41d6cc054693bbeb7a45e0247050c2837bac",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/nightly/cu118/libtorch-shared-with-deps-2.1.0.dev20230419%2Bcu118.zip"],
)

# Download these tarballs manually from the NVIDIA website
# Either place them in the distdir directory in third_party and use the --distdir flag
# or modify the urls to "file:///<PATH TO TARBALL>/<TARBALL NAME>.tar.gz

http_archive(
    name = "cudnn",
    build_file = "@//third_party/cudnn/archive:BUILD",
    sha256 = "36fff137153ef73e6ee10bfb07f4381240a86fb9fb78ce372414b528cbab2293",
    strip_prefix = "cudnn-linux-x86_64-8.8.0.121_cuda11-archive",
    urls = [
        "https://developer.download.nvidia.com/compute/cudnn/secure/8.8.0/local_installers/11.8/cudnn-linux-x86_64-8.8.0.121_cuda11-archive.tar.xz",
    ],
)

http_archive(
    name = "tensorrt",
    build_file = "@//third_party/tensorrt/archive:BUILD",
    sha256 = "c1732a1093c57ab79fa0b687f061be369e449c9c17792b660f3663ecd8fa7b63",
    strip_prefix = "TensorRT-8.6.0.12",
    urls = [
        "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.0/tars/TensorRT-8.6.0.12.Linux.x86_64-gnu.cuda-11.8.tar.gz",
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
