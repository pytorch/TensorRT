load("//tests/core/runtime:runtime_test.bzl", "runtime_test")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

runtime_test(
    name = "test_multi_device_safe_mode",
)

test_suite(
    name = "runtime_tests",
    tests = [
        ":test_multi_device_safe_mode",
    ],
)
