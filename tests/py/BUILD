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

# Following multi_gpu test is only targeted for multi-gpu configurations. It is not included in the test suite by default.
py_test(
    name = "test_api_multi_gpu",
    srcs = [
        "test_api_multi_gpu.py",
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
    name = "test_to_backend_api",
    srcs = [
        "test_to_backend_api.py",
        "model_test_case.py"
    ],
    deps = [
        requirement("torchvision")
    ]
)
