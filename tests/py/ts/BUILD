load("@py_test_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64_linux",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
)

py_test(
    name = "test_api",
    srcs = [
        "model_test_case.py",
        "test_api.py",
    ] + select({
        ":aarch64_linux": [
            "test_api_dla.py",
        ],
        "//conditions:default": [],
    }),
    deps = [
        requirement("torchvision"),
    ],
)

# Following multi_gpu test is only targeted for multi-gpu configurations. It is not included in the test suite by default.
py_test(
    name = "test_multi_gpu",
    srcs = [
        "model_test_case.py",
        "test_multi_gpu.py",
    ],
    deps = [
        requirement("torchvision"),
    ],
)

py_test(
    name = "test_to_backend_api",
    srcs = [
        "model_test_case.py",
        "test_to_backend_api.py",
    ],
    deps = [
        requirement("torchvision"),
    ],
)

py_test(
    name = "test_trt_intercompatibility",
    srcs = [
        "model_test_case.py",
        "test_trt_intercompatibility.py",
    ],
    deps = [
        requirement("torchvision"),
    ],
)
