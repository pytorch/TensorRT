package(default_visibility = ["//visibility:public"])

load("@py_test_deps//:requirements.bzl", "requirement")

py_test(
    name = "test_api",
    srcs = [
        "test_api.py",
    ],
    deps = [
        requirement("torchvision"),
    ],
)
