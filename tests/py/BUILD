package(default_visibility = ["//visibility:public"])
load("@py_test_deps//:requirements.bzl", "requirement")


py_test(
    name = "test_api",
    srcs = [
        "test_api.py",
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
