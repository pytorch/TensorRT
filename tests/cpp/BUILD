package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)


test_suite(
    name = "api_tests",
    tests = [
        ":test_default_input_types",
        ":test_compiled_modules",
        ":test_modules_as_engines",
        ":test_multiple_registered_engines",
        ":test_serialization",
    ],
)

test_suite(
    name = "aarch64_api_tests",
    tests = [
        ":test_default_input_types",
        ":test_compiled_modules",
        ":test_modules_as_engines",
        ":test_multiple_registered_engines",
        ":test_serialization",
    ],
)

cc_test(
    name = "test_default_input_types",
    srcs = ["test_default_input_types.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        ":cpp_api_test",
    ],
)

cc_test(
    name = "test_serialization",
    srcs = ["test_serialization.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        ":cpp_api_test",
    ],
)

cc_test(
    name = "test_multiple_registered_engines",
    srcs = ["test_multiple_registered_engines.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        "//cpp/api:trtorch",
        "//tests/util",
        "@googletest//:gtest_main",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    })
)

cc_test(
    name = "test_modules_as_engines",
    srcs = ["test_modules_as_engines.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        ":cpp_api_test",
    ],
)

cc_test(
    name = "test_compiled_modules",
    srcs = ["test_compiled_modules.cpp"],
    data = [
        "//tests/modules:jit_models",
    ],
    deps = [
        ":cpp_api_test",
    ],
)

cc_test(
    name = "test_multi_gpu_serdes",
    srcs = ["test_multi_gpu_serdes.cpp"],
    data = [
        ":jit_models",
    ],
    deps = [
        ":module_test",
    ],
)

cc_library(
    name = "cpp_api_test",
    hdrs = ["cpp_api_test.h"],
    deps = [
        "//cpp/api:trtorch",
        "//tests/util",
        "@googletest//:gtest_main",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
)
