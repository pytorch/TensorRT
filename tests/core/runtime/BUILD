load("//tests/core/runtime:runtime_test.bzl", "runtime_test")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl"
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
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
