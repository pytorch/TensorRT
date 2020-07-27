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
        "test_api.py"
    ] + select({
	":aarch64_linux": [
		"test_api_dla.py"
	]
    }),
    deps = [
        requirement("torchvision")
    ]
)
