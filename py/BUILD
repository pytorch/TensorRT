package(default_visibility = ["//visibility:public"])

load("@torch_tensorrt_py_deps//:requirements.bzl", "requirement")

# Exposes the library for testing
py_library(
    name = "torch_tensorrt",
    srcs = [
        "torch_tensorrt/__init__.py",
        "torch_tensorrt/_compile_spec.py",
        "torch_tensorrt/_compiler.py",
        "torch_tensorrt/_types.py",
        "torch_tensorrt/_version.py",
        "torch_tensorrt/logging.py",
        "torch_tensorrt/ptq.py",
    ],
    data = [
        "torch_tensorrt/lib/libtrtorch.so",
    ] + glob([
        "torch_tensorrt/_C.cpython*.so",
    ]),
    deps = [
        requirement("torch"),
    ],
)
