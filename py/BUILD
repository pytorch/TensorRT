package(default_visibility = ["//visibility:public"])

load("@trtorch_py_deps//:requirements.bzl", "requirement")

# Exposes the library for testing
py_library(
    name = "trtorch",
    srcs = [
        "trtorch/__init__.py",
        "trtorch/_compiler.py",
        "trtorch/_extra_info.py",
        "trtorch/_types.py",
        "trtorch/_version.py",
        "trtorch/logging.py",
    ],
    data = [
        "trtorch/lib/libtrtorch.so",
    ] + glob([
        "trtorch/_C.cpython*.so",
    ]),
    deps = [
        requirement("torch"),
    ],
)
