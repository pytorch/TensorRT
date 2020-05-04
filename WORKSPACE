workspace(name = "TRTorch")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
    sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
)

load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()
# Only needed if using the packaging rules.
load("@rules_python//python:pip.bzl", "pip_repositories", "pip_import")
pip_repositories()

http_archive(
    name = "rules_pkg",
    url = "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
    sha256 = "4ba8f4ab0ff85f2484287ab06c0d871dcb31cc54d439457d28fd4ae14b18450a",
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

# CUDA should be installed on the system locally
new_local_repository(
    name = "cuda",
    path = "/usr/local/cuda-10.2/targets/x86_64-linux/",
    build_file = "@//third_party/cuda:BUILD",
)

http_archive(
    name = "libtorch",
    build_file = "@//third_party/libtorch:BUILD",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.5.0.zip"],
    sha256 = "0efdd4e709ab11088fa75f0501c19b0e294404231442bab1d1fb953924feb6b5"
)

# Downloaded distributions to use with --distdir
http_archive(
    name = "cudnn",
    urls = ["https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/cudnn-10.2-linux-x64-v7.6.5.32.tgz",],

    build_file = "@//third_party/cudnn/archive:BUILD",
    sha256 = "600267f2caaed2fd58eb214ba669d8ea35f396a7d19b94822e6b36f9f7088c20",
    strip_prefix = "cuda"
)

http_archive(
    name = "tensorrt",
    urls = ["https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.0/7.0.0.11/tars/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz",],

    build_file = "@//third_party/tensorrt/archive:BUILD",
    sha256 = "c7d73b2585b18aae68b740249efa8c8ba5ae852abe9a023720595432a8eb4efd",
    strip_prefix = "TensorRT-7.0.0.11"
)

## Locally installed dependencies
# new_local_repository(
#    name = "cudnn",
#    path = "/usr/",
#    build_file = "@//third_party/cudnn/local:BUILD"
#)

# new_local_repository(
#   name = "tensorrt",
#   path = "/usr/",
#   build_file = "@//third_party/tensorrt/local:BUILD"
#)

git_repository(
    name = "googletest",
    remote = "https://github.com/google/googletest",
    commit = "703bd9caab50b139428cea1aaff9974ebee5742e",
    shallow_since = "1570114335 -0400"
)
