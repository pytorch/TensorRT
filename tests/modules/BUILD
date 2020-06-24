package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

config_setting(
    name = "use_pre_cxx11_abi_aarch64",
    values = {
        "define": "abi=pre_cxx11_abi",
        "cpu" : "aarch64"
    }
)

config_setting(
    name = "aarch64",
    values = {
        "cpu" : "aarch64"
    }
)

config_setting(
    name = "x86_64",
    values = {
        "cpu" : "x86_64"
    }
)

filegroup(
    name = "jit_models",
    srcs = glob(["**/*.jit.pt"])
)

test_suite(
    name = "test_modules",
    tests = [
        ":test_modules_as_engines",
        ":test_compiled_modules",
        ":test_multiple_registered_engines",
        ":test_serialization"
    ]
)

cc_test(
    name = "test_serialization",
    srcs = ["test_serialization.cpp"],
    deps = [
        ":module_test",
    ],
    data = [
        ":jit_models"
    ]
)

cc_test(
    name = "test_multiple_registered_engines",
    srcs = ["test_multiple_registered_engines.cpp"],
    deps = [
        ":module_test",
    ],
    data = [
        ":jit_models"
    ]
)

cc_test(
    name = "test_modules_as_engines",
    srcs = ["test_modules_as_engines.cpp"],
    deps = [
        ":module_test"
    ],
    data = [
        ":jit_models"
    ]
)

cc_test(
    name = "test_compiled_modules",
    srcs = ["test_compiled_modules.cpp"],
    deps = [
        ":module_test"
    ],
    data = [
        ":jit_models"
    ]
)

cc_library(
    name = "module_test",
    hdrs = ["module_test.h"],
    deps = [
        "//cpp/api:trtorch",
        "//tests/util",
        "@googletest//:gtest_main",
    ] + select({
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        ":use_pre_cxx11_abi_aarch64":  ["@libtorch_pre_cxx11_abi_aarch64//:libtorch"],
        ":aarch64" : ["@libtorch_aarch64//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
)
