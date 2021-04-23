package(default_visibility = ["//visibility:public"])
load("@py_test_deps//:requirements.bzl", "requirement")

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
        "test_api.py",
        "model_test_case.py"
    ] + select({
    ":aarch64_linux": [
        "test_api_dla.py"
    ],
    "//conditions:default" : []
    }),
    deps = [
        requirement("torchvision")
    ]
)

py_test(
    name = "test_ptq_dataloader_calibrator",
    srcs = [
        "test_ptq_dataloader_calibrator.py",
        "model_test_case.py"
    ],
    deps = [
        requirement("torchvision")
    ]
)

# This test is not included in the main test suite by default. This test checks
# if trtorch can use pre-existing trt calibrators already implemented by users.
py_test(
    name = "test_ptq_trt_calibrator",
    srcs = [
        "test_ptq_trt_calibrator.py",
        "model_test_case.py"
    ],
    deps = [
        requirement("torchvision")
    ]
)

# Following multi_gpu test is only targeted for multi-gpu configurations. It is not included in the test suite by default.
py_test(
    name = "test_multi_gpu",
    srcs = [
        "test_multi_gpu.py",
        "model_test_case.py"
    ],
    deps = [
        requirement("torchvision")
    ]
)

py_test(
    name = "test_to_backend_api",
    srcs = [
        "test_to_backend_api.py",
        "model_test_case.py"
    ],
    deps = [
        requirement("torchvision")
    ]
)

py_test(
    name = "test_trt_intercompatability",
    srcs = [
        "test_trt_intercompatability.py",
        "model_test_case.py"
    ],
    deps = [
        requirement("torchvision")
    ]
)

py_test(
    name = "test_ptq_to_backend",
    srcs = [
        "test_ptq_to_backend.py",
        "model_test_case.py"
    ],
    deps = [
        requirement("torchvision")
    ]
)

test_suite(
    name = "py_calibrator_tests",
    tests = [
        ":test_ptq_to_backend",
        ":test_ptq_trt_calibrator",
        ":test_ptq_dataloader_calibrator"
    ],
)